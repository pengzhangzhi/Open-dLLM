# Copyright 2025 Open-dLLM Contributors
# SPDX-License-Identifier: Apache-2.0

"""
NVFP4 QAT converter — replaces nn.Linear with NVFP4FakeQuantizedLinear
for Blackwell 4-bit training.

Integrates with torchao's prototype NVFP4 QAT to store weights in NVFP4
(E2M1, 4-bit) during training while computing gradients in full precision.

Usage in config:
    model:
      enable_nvfp4_qat: true

Requires:
    pip install --pre torch torchao mslk --index-url https://download.pytorch.org/whl/nightly/cu130
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn


# ── Module types to skip during conversion ─────────────────────────

_SKIP_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.Embedding,
    # RMSNorm subclasses
    # Any custom norm will be matched by name patterns below
)

_SKIP_NAME_PATTERNS: List[str] = [
    "norm",
    "embed_tokens",
    "embed_positions",
    "lm_head",          # often weight-tied with embed_tokens
    "rotary_emb",
]


def _should_skip(name: str, mod: nn.Module) -> bool:
    """Return True if *mod* should be left in its original precision."""
    if isinstance(mod, _SKIP_MODULE_TYPES):
        return True
    for pattern in _SKIP_NAME_PATTERNS:
        if pattern in name.lower():
            return True
    return False


# ── Default QAT configs ────────────────────────────────────────────

def _default_nvfp4_qat_config(per_tensor_scale: bool = True, triton: bool = False):
    """Build default NVFP4FakeQuantizeConfig for both weights and activations."""
    from torchao.prototype.qat.nvfp4 import NVFP4FakeQuantizeConfig
    return NVFP4FakeQuantizeConfig(
        use_per_tensor_scale=per_tensor_scale,
        use_swizzled_scales=False,
        use_triton_kernel=triton,
    )


# ── Module-level cache for original dtypes (restore on convert) ────
_original_dtypes: Dict[str, torch.dtype] = {}


def apply_nvfp4_qat_prepare(
    model: nn.Module,
    activation_config=None,
    weight_config=None,
    per_tensor_scale: bool = True,
    use_triton: bool = False,
    skip_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace all trainable ``nn.Linear`` modules in *model* with
    ``NVFP4FakeQuantizedLinear`` (QAT "prepare" step).

    During forward: weights and activations are quantized to NVFP4.
    During backward: dequantized to original dtype for gradient computation.

    Args:
        model: The model to convert (in-place).
        activation_config: Optional ``NVFP4FakeQuantizeConfig`` for activations.
        weight_config: Optional ``NVFP4FakeQuantizeConfig`` for weights.
        per_tensor_scale: Enable two-level per-tensor FP32 scaling.
        use_triton: Use Triton kernels for NVFP4 operations.
        skip_names: Additional module name patterns to skip.

    Returns:
        The converted model (same object, modified in-place).
    """
    from torchao.prototype.qat.nvfp4 import NVFP4FakeQuantizedLinear, NVFP4FakeQuantizeConfig

    if activation_config is None:
        activation_config = NVFP4FakeQuantizeConfig(
            use_per_tensor_scale=per_tensor_scale,
            use_swizzled_scales=False,
            use_triton_kernel=use_triton,
        )
    if weight_config is None:
        weight_config = NVFP4FakeQuantizeConfig(
            use_per_tensor_scale=per_tensor_scale,
            use_swizzled_scales=False,
            use_triton_kernel=use_triton,
        )

    merge_skip = list(_SKIP_NAME_PATTERNS)
    if skip_names:
        merge_skip.extend(skip_names)

    converted = 0
    skipped = 0

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if _should_skip(name, mod):
            skipped += 1
            continue

        # Cannot convert modules on meta device
        if mod.weight.device.type == "meta":
            skipped += 1
            continue

        # Save original dtype for potential restore
        _original_dtypes[name] = mod.weight.dtype

        # Build replacement
        qat_linear = NVFP4FakeQuantizedLinear.from_linear(
            mod,
            activation_config=activation_config,
            weight_config=weight_config,
        )

        # Set replacement in parent
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        setattr(parent, child_name, qat_linear)
        converted += 1

    print(
        f"[nvfp4_qat] prepare: converted {converted} Linear → NVFP4FakeQuantizedLinear, "
        f"skipped {skipped}"
    )
    return model


def apply_nvfp4_qat_convert(model: nn.Module) -> nn.Module:
    """
    Convert ``NVFP4FakeQuantizedLinear`` modules back to plain ``nn.Linear``
    with NVFP4-quantized weights frozen (QAT "convert" step).

    Call after training to produce an inference-ready model.
    """
    from torchao.prototype.qat.nvfp4 import NVFP4FakeQuantizedLinear

    converted = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, NVFP4FakeQuantizedLinear):
            continue
        plain = mod.to_linear()
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        setattr(parent, child_name, plain)
        converted += 1

    print(f"[nvfp4_qat] convert: restored {converted} NVFP4FakeQuantizedLinear → nn.Linear")
    return model


def estimate_nvfp4_memory_savings(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory savings from NVFP4 quantization.

    Returns:
        dict with keys:
          - param_bytes_before: total parameter size in bytes (current dtype)
          - param_bytes_after: estimated size in NVFP4 (4-bit + scales overhead)
          - reduction_ratio: before / after
    """
    total_bytes_before = 0
    linear_bytes_before = 0

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight
            if w.device.type != "meta":
                b = w.numel() * w.element_size()
                linear_bytes_before += b
                total_bytes_before += b
                if mod.bias is not None:
                    total_bytes_before += mod.bias.numel() * mod.bias.element_size()

    # NVFP4: 4-bit per value + 1 FP8 scale per 16 values + 1 FP32 per-tensor scale
    # Effective bits per value ≈ 4 + 8/16 + 32/N ≈ 4.5 per value
    overhead_factor = 4.5 / 4.0  # scales overhead
    param_bytes_after = linear_bytes_before * (4.0 / 16.0) * overhead_factor

    return {
        "param_bytes_before": total_bytes_before,
        "param_bytes_after": int(param_bytes_after),
        "reduction_ratio": total_bytes_before / param_bytes_after if param_bytes_after > 0 else 1.0,
    }
