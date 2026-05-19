# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
CARD (Causal Autoregressive Diffusion) soft-tail variant.

Strictly causal attention + soft-tailed masking. Only the tail of the
sequence is noisy at timestep t, preserving a clean prefix anchor for
stability. Addresses the "unlearnable early positions" problem in pure
causal diffusion.

The tail window is: W = min(L, floor(N * lambda_tail)), where
N = max(1, floor(L * t)). Lambda controls aggressiveness (0.5–1.0).

Loss is computed only on the tail positions.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .block_causal_dit import ColaDLMHead


def make_soft_tail_mask(
    seq_len: int,
    t: torch.Tensor,
    lambda_tail: float = 0.6,
    device: Optional[torch.device] = None,
):
    """
    CARD-style soft-tailed masking.

    Args:
        seq_len: total sequence length (G + L)
        t: scalar timestep in [0, 1]
        lambda_tail: aggressiveness (0.0–1.0)

    Returns:
        causal_mask: (seq_len, seq_len) bool — strictly causal
        tail_start: int — index where the noisy tail begins
    """
    device = device or t.device
    N = torch.clamp((seq_len * t).floor().long(), min=1)
    W = torch.clamp((N * lambda_tail).floor().long(), min=1, max=seq_len)
    tail_start = seq_len - W.item()

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return causal_mask, tail_start


class CardColaDLMHead(ColaDLMHead):
    """
    CARD variant: strictly causal attention + soft-tail loss weighting.

    The diffusion loss is applied only to the tail window, while the
    prefix provides clean context anchors (similar to the AR teacher's
    causal prefix).
    """

    def __init__(self, dim: int, lambda_tail: float = 0.6, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.lambda_tail = lambda_tail

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.text_vae(h)
        z = torch.cat([enc["z_global"], enc["z_local"]], dim=1)  # (B, G+L, dim)

        B, N, D = z.shape
        t = self.sample_timesteps(B, device=z.device, dtype=z.dtype)
        a, b = self.noise_schedule(t)
        noise = torch.randn_like(z)
        z_t = a * z + b * noise

        # Build strictly causal mask + compute tail boundary
        # Use per-sample t (different tail lengths per sample)
        # For efficiency, use the mean t for the mask (same mask across batch)
        t_mean = t.mean()
        causal_mask, tail_start = make_soft_tail_mask(N, t_mean, self.lambda_tail, z.device)

        pred = self.dit(z_t, t, mask=causal_mask)

        target = (noise - z) if self.prediction_type == "v" else z

        # Loss only on tail positions
        if tail_start < N:
            loss = F.mse_loss(pred[:, tail_start:], target[:, tail_start:])
        else:
            loss = F.mse_loss(pred, target)

        return {
            "loss": loss,
            "z": z,
            "z_pred": pred,
            "target": target,
            "t_mean": t.mean(),
            "tail_start": tail_start,
            "tail_ratio": (N - tail_start) / N,
            "z_global": enc["z_global"],
            "z_local": enc["z_local"],
            "prediction_type": self.prediction_type,
        }
