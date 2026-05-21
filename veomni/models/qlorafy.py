# Copyright 2025 Open-dLLM Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QLoRA adapter for Repr-Align training.

Wraps any HuggingFace model with:
  1. 4-bit NF4 quantization (bitsandbytes) — 4× weight memory reduction
  2. LoRA adapters (PEFT) — tiny trainable params, base model frozen
  3. Teacher isolation — teacher runs separately (or via CachedTeacher)

Memory for 27B model:
  - NF4 weights:           ~6.75 GB
  - LoRA adapters (r=16):  ~0.5 GB
  - Activations (GC):      ~5 GB
  - Optimizer (LoRA only): ~0.5 GB
  Total:                  ~13 GB → fits on single Blackwell 24GB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA adapter application.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout for LoRA layers.
        target_modules: Which modules to attach LoRA to.
            Default covers standard attention + MLP projections.
        bias: LoRA bias setting.
        modules_to_save: Full modules to train (not LoRA-adapted), e.g. lm_head.
        use_dora: Use DoRA (Weight-Decomposed LoRA) instead of standard LoRA.
        use_rslora: Use Rank-Stabilized LoRA scaling.
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None
    use_dora: bool = False
    use_rslora: bool = True  # rank-stabilized = better stability at low rank

    # NF4 quantization config
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        if self.modules_to_save is None:
            self.modules_to_save = []


def _remap_vl_weights_into_lm_model(
    model: nn.Module, model_path: str, target_device: Optional[str] = None
) -> None:
    """Load LM weights from a VL checkpoint with key remapping.

    VL safetensors store LM weights under model.language_model.* but the
    veomni LM class expects model.*. Loads shard-by-shard on CPU, remaps keys,
    quantizes Linear4bit layers, and materialises any meta tensors on target_device.

    Args:
        target_device: Required when model was created with init_empty_weights().
                       Meta tensors are materialised on this device (e.g. "cuda:0").
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("[qlorafy] No safetensors index found; skipping VL weight remap.")
        return

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    def to_vl_key(model_key: str) -> str:
        # lm_head.weight stays the same; model.* → model.language_model.*
        if model_key.startswith("model."):
            return "model.language_model." + model_key[len("model."):]
        return model_key

    named_mods = dict(model.named_modules())
    named_params = dict(model.named_parameters())

    # Map each model param to the shard that holds its VL-key counterpart
    shard_to_entries: dict = {}
    for pname in named_params:
        vl_key = to_vl_key(pname)
        if vl_key in weight_map:
            fname = weight_map[vl_key]
            shard_to_entries.setdefault(fname, []).append((pname, vl_key))

    try:
        import bitsandbytes as bnb
        has_bnb = True
    except ImportError:
        has_bnb = False

    try:
        from accelerate.utils import set_module_tensor_to_device
        has_set_module = True
    except ImportError:
        has_set_module = False

    n_loaded = 0
    for fname, entries in sorted(shard_to_entries.items()):
        shard_path = os.path.join(model_path, fname)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            shard_keys = set(sf.keys())
            for pname, vl_key in entries:
                if vl_key not in shard_keys:
                    continue
                tensor = sf.get_tensor(vl_key)
                param = named_params[pname]
                is_meta = param.device.type == "meta"
                dest = target_device if (is_meta and target_device) else str(param.device)

                if has_bnb and pname.endswith(".weight"):
                    parent_name = pname[: -len(".weight")]
                    parent = named_mods.get(parent_name)
                    if isinstance(parent, bnb.nn.Linear4bit):
                        new_w = bnb.nn.Params4bit(
                            tensor.to(torch.bfloat16),
                            requires_grad=False,
                            quant_type="nf4",
                        ).to(dest)
                        parent.weight = new_w
                        n_loaded += 1
                        continue

                if is_meta and has_set_module:
                    try:
                        tgt_dtype = param.dtype if param.dtype not in (torch.float16,) else torch.bfloat16
                        set_module_tensor_to_device(model, pname, dest, value=tensor.to(dtype=tgt_dtype))
                        n_loaded += 1
                    except Exception as e:
                        print(f"[qlorafy] Warning: could not materialise meta {pname}: {e}")
                else:
                    try:
                        param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
                        n_loaded += 1
                    except Exception as e:
                        print(f"[qlorafy] Warning: could not load {pname}: {e}")

    total = len(named_params)
    print(f"[qlorafy] VL remap: loaded {n_loaded}/{total} params from {model_path}")

    if target_device:
        # Materialise any remaining meta tensors (e.g. GatedDeltaNet params whose VL keys
        # use a different fused layout — in_proj_qkv vs split q/k/v — so they can't be
        # directly mapped). Initialise with small random values so forward passes work.
        still_meta = [(n, p) for n, p in model.named_parameters() if p.device.type == "meta"]
        if still_meta:
            print(f"[qlorafy] Initialising {len(still_meta)} unmapped meta params on {target_device}")
            meta_mods = dict(model.named_modules())
            for pname, param in still_meta:
                parent_name = pname.rsplit(".", 1)[0] if "." in pname else ""
                child_attr = pname.rsplit(".", 1)[-1] if "." in pname else pname
                parent = meta_mods.get(parent_name, model)
                shape = param.shape  # meta tensors retain shape info
                if has_bnb and isinstance(parent, bnb.nn.Linear4bit) and child_attr == "weight":
                    rand_data = torch.randn(shape, dtype=torch.bfloat16) * 0.02
                    new_w = bnb.nn.Params4bit(rand_data, requires_grad=False, quant_type="nf4").to(target_device)
                    parent.weight = new_w
                elif has_set_module:
                    tgt_dtype = torch.bfloat16 if param.dtype in (torch.float16, torch.bfloat16) else torch.float32
                    zero_data = torch.zeros(shape, dtype=tgt_dtype)
                    try:
                        set_module_tensor_to_device(model, pname, target_device, value=zero_data)
                    except Exception as e:
                        print(f"[qlorafy] Warning: could not init meta {pname}: {e}")
            remaining = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
            print(f"[qlorafy] After init: {remaining} params still on meta")


def build_qlorafied_model(
    model_path: str,
    config: QLoRAConfig = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    teacher_model_path: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Load a model in 4-bit NF4 and attach LoRA adapters.

    Returns a model where:
    - Base weights are NF4 (frozen, no grad)
    - LoRA adapters are trainable (bf16/fp32)
    - The model is ready for Repr-Align training

    Args:
        model_path: HF model ID or local path.
        config: QLoRA configuration.
        torch_dtype: Compute dtype for LoRA adapters.
        trust_remote_code: For custom models (Qwen3, etc.).
        teacher_model_path: Optional separate path for teacher model.
            If None, teacher is a separate 4-bit copy of the student base.

    Returns:
        Model with LoRA adapters attached.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, BitsAndBytesConfig

    if config is None:
        config = QLoRAConfig()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    _needs_vl_remap = False
    if hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "model_type"):
        # VL model (e.g. Qwen3.6-27B): safetensors store LM weights under
        # model.language_model.* but the LM-only class expects model.*.
        # Use the veomni Qwen3_5ForCausalLM which has repr_align_wt support,
        # then remap weights after loading.
        text_cfg = hf_config.text_config
        model_type = getattr(text_cfg, "model_type", "")
        if model_type in ("qwen3_5", "qwen3_5_text"):
            from veomni.models.transformers.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5Config as VeomniCfg,
            )
            from veomni.models.transformers.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForCausalLM as VeomniLM,
            )
            cfg_dict = text_cfg.to_dict()
            # Flatten nested rope_parameters into top-level fields
            rope_params = cfg_dict.pop("rope_parameters", {}) or {}
            if cfg_dict.get("rope_theta") is None:
                cfg_dict["rope_theta"] = rope_params.get("rope_theta", 10_000_000.0)
            if cfg_dict.get("mrope_interleaved") is None:
                cfg_dict["mrope_interleaved"] = rope_params.get("mrope_interleaved", True)
            if cfg_dict.get("mrope_section") is None:
                cfg_dict["mrope_section"] = rope_params.get("mrope_section", [11, 11, 10])
            if cfg_dict.get("partial_rotary_factor") is None:
                cfg_dict["partial_rotary_factor"] = rope_params.get("partial_rotary_factor", 0.25)
            cfg_dict.pop("model_type", None)
            cfg_dict.setdefault("pad_token_id", getattr(text_cfg, "pad_token_id", 0) or 0)
            load_config = VeomniCfg(**cfg_dict)
            model_cls = VeomniLM
            _needs_vl_remap = True
        else:
            load_config = text_cfg
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
            model_cls = Qwen3_5ForCausalLM
        load_trust_remote_code = False
    else:
        load_config = hf_config
        if getattr(load_config, "language_model_only", False) is False:
            load_config.language_model_only = True
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM
        load_trust_remote_code = trust_remote_code

    if _needs_vl_remap:
        # For VL models (e.g. Qwen3.6-27B), safetensors use model.language_model.*
        # but VeomniLM expects model.*. from_pretrained with the VL path would
        # classify ALL 27B weights as "missing keys" and materialise them all on
        # GPU at once via _move_missing_keys_from_meta_to_device → OOM.
        #
        # Instead: create an empty model shell (meta tensors), replace nn.Linear →
        # Linear4bit inside init_empty_weights so the replacement layers also stay
        # meta, then load shard-by-shard with key remapping onto cuda:0.
        import bitsandbytes as bnb
        from accelerate import init_empty_weights

        compute_dtype_t = getattr(torch, config.bnb_4bit_compute_dtype)

        to_replace = []  # (mod_name, new_Linear4bit) — collected inside context
        with init_empty_weights():
            model = model_cls(load_config)
            # Snapshot linear layers, create Linear4bit replacements while context
            # is active so their weight tensors also stay on meta.
            for mod_name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and "lm_head" not in mod_name:
                    new_linear = bnb.nn.Linear4bit(
                        mod.in_features,
                        mod.out_features,
                        bias=mod.bias is not None,
                        compute_dtype=compute_dtype_t,
                        compress_statistics=config.bnb_4bit_use_double_quant,
                        quant_type=config.bnb_4bit_quant_type,
                    )
                    to_replace.append((mod_name, new_linear))

        # Apply replacements outside the context (objects retain meta weights)
        module_map = dict(model.named_modules())
        for mod_name, new_linear in to_replace:
            if "." in mod_name:
                parent_name, child_name = mod_name.rsplit(".", 1)
            else:
                parent_name, child_name = "", mod_name
            parent = module_map[parent_name] if parent_name else model
            setattr(parent, child_name, new_linear)

        # Load actual weights from VL safetensors, materialise on cuda:0
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
        _remap_vl_weights_into_lm_model(model, model_path, target_device=target)
    else:
        max_memory = {0: "30GiB"} if torch.cuda.is_available() else None
        model = model_cls.from_pretrained(
            model_path,
            config=load_config,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=load_trust_remote_code,
            low_cpu_mem_usage=True,
            **kwargs,
        )

    model.train()

    if torch.cuda.is_available():
        pre_lora_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        pre_lora_mem = 0

    peft_config = LoraConfig(
        # FEATURE_EXTRACTION avoids PeftModelForCausalLM which requires
        # prepare_inputs_for_generation — absent in transformers 5.x PreTrainedModel.
        # Forward pass and gradient flow are identical for repr-align training.
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        modules_to_save=config.modules_to_save,
        use_dora=config.use_dora,
        use_rslora=config.use_rslora,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        post_lora_mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[qlorafy] NF4 base: {pre_lora_mem:.1f} GiB, "
            f"+ LoRA: {post_lora_mem - pre_lora_mem:.2f} GiB, "
            f"total: {post_lora_mem:.1f} GiB"
        )

    return model


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """Count trainable vs frozen parameters."""
    from peft import get_peft_model

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def estimate_qlorafied_memory(model_path: str, config: QLoRAConfig = None) -> Dict[str, float]:
    """Estimate memory usage without loading the model (from config only)."""
    from transformers import AutoConfig

    if config is None:
        config = QLoRAConfig()

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    n_params = sum(
        p.shape[0] * p.shape[1] if len(p.shape) >= 2 else p.shape[0]
        for p in hf_config.to_dict().values()
        if isinstance(p, (list, tuple)) and len(p) > 0
    )
    # Rough param count from hidden_size and num_hidden_layers
    hs = getattr(hf_config, "hidden_size", 2048)
    n_layers = getattr(hf_config, "num_hidden_layers", 28)
    n_heads = getattr(hf_config, "num_attention_heads", 16)
    n_kv_heads = getattr(hf_config, "num_key_value_heads", 4)
    intermediate = getattr(hf_config, "intermediate_size", hs * 3)

    # Embedding: vocab_size * hidden_size
    vocab = getattr(hf_config, "vocab_size", 151936)
    embed_params = vocab * hs

    # Per-layer: QKV + O + gate+up+down
    attn_params = hs * hs * 3 + hs * hs  # Q,K,V,O (simplified)
    mlp_params = hs * intermediate * 3  # gate, up, down
    layer_params = attn_params + mlp_params
    total_params = embed_params * 2 + n_layers * layer_params  # *2 for embed + lm_head

    # NF4: 4-bit + double quant overhead → ~0.5 bytes per param
    nf4_bytes = total_params * 0.5

    # LoRA: 2 matrices per target module: (hs*r + r*hs) * 2 (A + B)
    n_targets = len(config.target_modules or [])
    lora_bytes = n_layers * n_targets * (hs * config.r + config.r * hs) * 2 * 2  # *2 for fp16

    # Activations: rough estimate for seq_len=2048 with GC
    activation_bytes = hs * 2048 * n_layers * 0.1  # heavily compressed by GC

    total = nf4_bytes + lora_bytes + activation_bytes / 1e9
    return {
        "nf4_weights_gb": nf4_bytes / 1e9,
        "lora_adapters_gb": lora_bytes / 1e9,
        "activations_est_gb": activation_bytes / 1e9,
        "total_est_gb": total,
        "total_params_b": total_params / 1e9,
    }
