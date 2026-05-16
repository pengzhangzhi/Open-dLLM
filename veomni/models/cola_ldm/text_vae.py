# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TextVAEEncoder — the compressor half of the Cola DLM Text VAE (arXiv:2605.06548).

In the paper, the Text VAE learns a text → multi-scale continuous latent
mapping (encoder) and a conditional decoder that maps latents back to text.

In Open-dLLM we plug only the **encoder** in as an auxiliary head on
top of Repr-Align: the upstream LM hidden states already carry strong
linguistic structure, and the LM head (already in the base model) plays
the role of the conditional decoder. So the role of `TextVAEEncoder`
here is to compress those hidden states into the two-scale latent
(z_global, z_local) that the BlockCausalDiT then operates on.

Two parallel Perceivers + a linear fusion injects the global summary
into every local latent, mirroring the paper's "global semantic latent +
local token-level latent" split.

Design knobs are kept small by default so a 2-GPU 35B-A3B run is
feasible (~200-400 M trainable params for the head).
"""

from typing import Dict

import torch
import torch.nn as nn

from .modules import PerceiverResampler


class TextVAEEncoder(nn.Module):
    """
    Cola DLM Text VAE encoder (hierarchical Perceiver compressor).

    Args:
        dim:          LM hidden size (e.g. 2048 for Qwen3.6-35B-A3B).
        num_global:   Global semantic latents (small, e.g. 16).
        num_local:    Local detail latents (larger, e.g. 64).
        global_depth: Depth of the global Perceiver.
        local_depth:  Depth of the local Perceiver.
        heads:        Attention heads in both Perceivers (must divide `dim`).
    """

    def __init__(
        self,
        dim: int,
        num_global: int = 16,
        num_local: int = 64,
        global_depth: int = 2,
        local_depth: int = 2,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_global = num_global
        self.num_local = num_local

        self.global_perceiver = PerceiverResampler(
            dim=dim, num_latents=num_global, depth=global_depth, heads=heads
        )
        self.local_perceiver = PerceiverResampler(
            dim=dim, num_latents=num_local, depth=local_depth, heads=heads
        )
        # Fusion: concat(local, mean(global)) -> local
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (B, T, dim) hidden states from the LM (typically detached).
        Returns:
            z_global:    (B, num_global, dim) coarse semantic latents
            z_local:     (B, num_local,  dim) fused local latents (carries global context)
            z_local_raw: (B, num_local,  dim) local latents pre-fusion (useful for analysis)
        """
        z_global = self.global_perceiver(h)
        z_local_raw = self.local_perceiver(h)

        global_summary = z_global.mean(dim=1, keepdim=True).expand(-1, z_local_raw.shape[1], -1)
        z_local = self.fusion(torch.cat([z_local_raw, global_summary], dim=-1))

        return {"z_global": z_global, "z_local": z_local, "z_local_raw": z_local_raw}
