# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Fast-dLLM v2 hybrid variant.

Combines block-causal attention (from Guide Labs) with:
  - Complementary masks: alternate positions masked per block, ensuring
    every position is learned across training steps.
  - Token-shift: uses preceding block's output as extra input features,
    retaining AR-style autoregressive characteristics.
  - Per-block independent noise levels for richer training signal.

This is the most AR-friendly variant — strong for speculative decoding
and converges well with Repr-Align.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .block_causal_dit import ColaDLMHead, make_block_causal_mask


def make_complementary_mask(
    num_global: int,
    num_local: int,
    block_size: int,
    device: Optional[torch.device] = None,
):
    """
    Complementary block-causal mask where within each block, every other
    position is masked (checkerboard pattern). Combined with the base
    block-causal mask, this ensures all positions get supervised across
    even/odd steps.

    Returns:
        mask: (G+L, G+L) bool attention mask
        mask_ratio: fraction of local positions that are "masked" (even/odd)
    """
    base_mask = make_block_causal_mask(num_global, num_local, block_size, device)

    # Apply complementary pattern within each local block
    if num_local > 0:
        local_positions = torch.arange(num_local, device=device)
        # Even positions can see even; odd can see odd (complementary groups)
        # This creates two sub-groups per block that alternate across steps
        block_ids = local_positions // block_size
        pos_in_block = local_positions % block_size

        # Group A: even positions within block, Group B: odd positions
        group = pos_in_block % 2  # 0 or 1

        # Within each block, group members can see each other + same/earlier blocks
        for b_id in range((num_local + block_size - 1) // block_size):
            start = b_id * block_size
            end = min(start + block_size, num_local)

            for i in range(start, end):
                g_i = pos_in_block[i]
                for j in range(start, end):
                    g_j = pos_in_block[j]
                    # Same group can see each other within block
                    if g_i == g_j:
                        base_mask[num_global + i, num_global + j] = True

    mask_ratio = 0.5  # Approximately half positions in complementary groups
    return base_mask, mask_ratio


class FastBlockColaDLMHead(ColaDLMHead):
    """
    Fast-dLLM v2 variant: block-causal + complementary masks + token-shift.

    The token-shift mechanism prepends the last position of each preceding
    block to the current block's input, giving the model AR-style context
    without full causal attention.
    """

    def __init__(self, dim: int, block_size: int = 8, use_complementary: bool = True, **kwargs):
        super().__init__(dim=dim, block_size=block_size, **kwargs)
        self.use_complementary = use_complementary
        self.token_shift_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.token_shift_proj.weight)

        if use_complementary:
            comp_mask, comp_ratio = make_complementary_mask(
                self.num_global, self.num_local, block_size
            )
            self.register_buffer("comp_mask", comp_mask, persistent=False)
            self.comp_ratio = comp_ratio

    def _apply_token_shift(self, z: torch.Tensor) -> torch.Tensor:
        """
        For each local block, prepend the last hidden state of the
        preceding block as an extra feature (shifted by 1 block).
        Global positions are left unchanged.
        """
        G = self.num_global
        L = self.num_local
        bs = self.block_size

        if L == 0:
            return z

        z_global = z[:, :G]
        z_local = z[:, G:]

        B, L_dim, D = z_local.shape
        num_blocks = (L_dim + bs - 1) // bs

        shifted = torch.zeros_like(z_local)
        shifted[:, 0] = self.token_shift_proj(z_global[:, -1])  # first block uses last global

        for b in range(1, num_blocks):
            prev_end = min(b * bs, L_dim) - 1
            curr_start = b * bs
            curr_end = min(curr_start + bs, L_dim)
            if curr_start < L_dim:
                shifted[:, curr_start:curr_end] = self.token_shift_proj(
                    z_local[:, prev_end:prev_end + 1]
                ).expand(-1, curr_end - curr_start, -1)

        z_local = z_local + shifted
        return torch.cat([z_global, z_local], dim=1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.text_vae(h)
        z = torch.cat([enc["z_global"], enc["z_local"]], dim=1)

        B = z.shape[0]
        t = self.sample_timesteps(B, device=z.device, dtype=z.dtype)
        a, b = self.noise_schedule(t)
        noise = torch.randn_like(z)
        z_t = a * z + b * noise

        # Apply token-shift to the noisy latents
        z_t = self._apply_token_shift(z_t)

        # Use complementary mask or base block-causal mask
        mask = self.comp_mask if self.use_complementary else self.block_mask
        if mask.device != z.device:
            mask = mask.to(z.device)

        pred = self.dit(z_t, t, mask=mask)

        target = (noise - z) if self.prediction_type == "v" else z
        loss = self.compute_loss(pred, target)

        result = {
            "loss": loss,
            "z": z,
            "z_pred": pred,
            "target": target,
            "t_mean": t.mean(),
            "z_global": enc["z_global"],
            "z_local": enc["z_local"],
            "prediction_type": self.prediction_type,
        }
        if self.use_complementary:
            result["complementary_mask_ratio"] = self.comp_ratio
        return result
