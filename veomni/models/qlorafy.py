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
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    if config is None:
        config = QLoRAConfig()

    # ── Step 1: 4-bit quantization config ────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # ── Step 2: Load model in 4-bit ──────────────────────────────
    # Note: device_map='auto' is critical for NF4 — it places
    # quantized modules on the right device.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    model.train()

    # ── Step 3: Measure memory before LoRA ───────────────────────
    if torch.cuda.is_available():
        pre_lora_mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        pre_lora_mem = 0

    # ── Step 4: Attach LoRA adapters ─────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
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

    # ── Step 5: Memory report ────────────────────────────────────
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
