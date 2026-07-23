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
Cola DLM block-causal DiT (arXiv:2605.06548) — auxiliary-head variant.

Operates on the concatenated latent z = [z_global ; z_local] produced
by `TextVAEEncoder`. Attention mask:

  - Global prefix [0, G):  fully bidirectional; any position can attend
                           to it (AR-style summary).
  - Local positions [G, G+L):  partitioned into chunks of `block_size`.
        - Within a chunk: bidirectional (diffusion-style).
        - Across chunks: causal (left → right; mirrors the official
          inference pipeline's KV-cache reuse pattern).

`make_block_causal_mask` is a free function — drop in Swin windows,
periodic global tokens, hybrid AR-prefix patterns without touching the
DiT.

Two prediction objectives are supported:

  - `prediction_type="v"` (default, paper alignment):
      Flow Matching. z_t = (1-t)*z + t*ε,  target velocity u_t = ε - z.
      Loss = MSE(v_pred, u_t). Inference does Euler `z ← z - v*dt`.

  - `prediction_type="x0"`:
      Cosine-like schedule (ᾱ = 1 - t²). x0-prediction MSE. Useful for
      cheap sanity-checks and ablations.

Both schedules / loss forms are overridable in subclasses via
`sample_timesteps`, `noise_schedule`, `compute_loss`.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    AdaLN,
    MaskedSelfAttention,
    MLPGeluTanh,
    TimestepEmbedding,
)
from .text_vae import TextVAEEncoder


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def make_block_causal_mask(
    num_global: int,
    num_local: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the (G+L, G+L) bool mask (True = attend, False = mask out).

    Layout:
        - Any query attends to all global keys [0, G).
        - Globals attend among themselves (bidirectional).
        - Local position i (block id b_i = i // block_size) attends to
          local positions j (block id b_j) iff b_j <= b_i.
    """
    assert num_global >= 0 and num_local >= 0 and block_size > 0
    N = num_global + num_local
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    if num_global > 0:
        mask[:, :num_global] = True

    if num_local > 0:
        local_pos = torch.arange(num_local, device=device)
        block_ids = local_pos // block_size  # (L,)
        local_mask = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)  # (L, L)
        mask[num_global:, num_global:] = local_mask

    return mask


# ---------------------------------------------------------------------------
# Denoiser (Cola DiT block, simplified for fixed-shape latents)
# ---------------------------------------------------------------------------

class ColaDiTBlock(nn.Module):
    """One Cola DiT layer: AdaLN-in → MaskedSelfAttention → AdaLN-out
    (gated residual) → AdaLN-in → MLP(GELU-tanh) → AdaLN-out.

    Layer norms are `elementwise_affine=False` because AdaLN owns the
    scale/shift — matches the official Cola DiT.
    """

    def __init__(self, dim: int, emb_dim: int, heads: int = 8, expand_ratio: int = 4, norm_eps: float = 1e-5):
        super().__init__()
        self.msa_norm = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.msa = MaskedSelfAttention(dim, heads=heads)
        self.mlp_norm = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.mlp = MLPGeluTanh(dim, expand_ratio=expand_ratio)
        self.ada = AdaLN(dim=dim, emb_dim=emb_dim, layers=["msa", "mlp"])

    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        msa_in = self.ada(x, emb=emb, layer="msa", mode="in", norm_layer=self.msa_norm)
        msa_out = self.msa(msa_in, mask=mask)
        x = self.ada(msa_out, emb=emb, layer="msa", mode="out", residual=x)

        mlp_in = self.ada(x, emb=emb, layer="mlp", mode="in", norm_layer=self.mlp_norm)
        mlp_out = self.mlp(mlp_in)
        x = self.ada(mlp_out, emb=emb, layer="mlp", mode="out", residual=x)
        return x


class BlockCausalDiT(nn.Module):
    """Block-causal Diffusion Transformer (denoiser only).

    Inputs:
        z_t  shape (B, N, dim) — noisy latents
        t    shape (B,)        — scalar timesteps in [0, 1]
    Output:
        shape (B, N, dim)      — predicted velocity (v) or clean (x0),
                                  depending on the head's prediction_type
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        emb_dim: Optional[int] = None,
        expand_ratio: int = 4,
        norm_eps: float = 1e-5,
        sinusoidal_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        emb_dim = emb_dim or dim

        self.time_embed = TimestepEmbedding(
            sinusoidal_dim=sinusoidal_dim, hidden_dim=dim, output_dim=emb_dim
        )
        self.blocks = nn.ModuleList(
            [
                ColaDiTBlock(dim=dim, emb_dim=emb_dim, heads=heads, expand_ratio=expand_ratio, norm_eps=norm_eps)
                for _ in range(depth)
            ]
        )
        # Final AdaLN-in shift+scale before the output norm (matches official Cola)
        self.out_norm = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.out_ada = AdaLN(dim=dim, emb_dim=emb_dim, layers=["out"], modes=["in"])
        # Zero-init projection so the freshly-built head starts at output ≈ 0
        # (paper-style, helps stability on the first few steps).
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.shape[0] == 1 and z.shape[0] > 1:
            t = t.expand(z.shape[0])
        emb = self.time_embed(t, dtype=z.dtype)  # (B, emb_dim)

        x = z
        for blk in self.blocks:
            x = blk(x, emb=emb, mask=mask)

        x = self.out_ada(x, emb=emb, layer="out", mode="in", norm_layer=self.out_norm)
        return self.proj_out(x)


# ---------------------------------------------------------------------------
# All-in-one: encoder + denoiser + loss
# ---------------------------------------------------------------------------

class ColaDLMHead(nn.Module):
    """
    End-to-end Cola DLM auxiliary head: TextVAEEncoder → BlockCausalDiT,
    with either Flow Matching (paper default) or x0-prediction MSE.

    forward(h) returns {"loss", "z", "z_pred", "target", "t_mean",
                        "z_global", "z_local", "prediction_type"}.

    Extension hooks (override in a subclass — no need to touch the wrapper):
        sample_timesteps(B, device, dtype)       → t       (default: uniform [0,1])
        noise_schedule(t)                        → (a, b)  coefficients of z and ε
        compute_loss(pred, target)               → scalar  (default: F.mse_loss)
    """

    def __init__(
        self,
        dim: int,
        num_global: int = 16,
        num_local: int = 64,
        block_size: int = 16,
        encoder_depth: int = 2,
        diffusion_depth: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        prediction_type: str = "v",
    ):
        super().__init__()
        assert prediction_type in ("v", "x0"), f"prediction_type must be 'v' or 'x0', got {prediction_type!r}"
        self.dim = dim
        self.num_global = num_global
        self.num_local = num_local
        self.block_size = block_size
        self.prediction_type = prediction_type

        self.text_vae = TextVAEEncoder(
            dim=dim,
            num_global=num_global,
            num_local=num_local,
            global_depth=encoder_depth,
            local_depth=encoder_depth,
            heads=heads,
        )
        self.dit = BlockCausalDiT(
            dim=dim,
            depth=diffusion_depth,
            heads=heads,
            expand_ratio=ff_mult,
        )

        self.register_buffer(
            "block_mask",
            make_block_causal_mask(num_global, num_local, block_size),
            persistent=False,
        )

    # -- extension hooks -------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)

    def noise_schedule(self, t: torch.Tensor):
        """
        Return (a, b) of shape (B, 1, 1) such that z_t = a*z + b*ε.

        - Flow Matching (paper):  a = 1 - t,    b = t
        - Cosine x0 (legacy):     a = sqrt(1 - t^2),  b = t  (i.e. sigma=t under
                                                              ᾱ = 1 - t^2)

        Override for rectified flow, EDM, etc.
        """
        t_ = t.view(-1, 1, 1)
        if self.prediction_type == "v":
            a = (1.0 - t_)
            b = t_
        else:  # "x0"
            alpha_bar = (1.0 - t_.pow(2)).clamp(min=1e-6)
            a = alpha_bar.sqrt()
            b = (1.0 - alpha_bar).clamp(min=1e-6).sqrt()
        return a, b

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    # -- forward ---------------------------------------------------------

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.text_vae(h)
        z = torch.cat([enc["z_global"], enc["z_local"]], dim=1)  # (B, G+L, dim)

        B = z.shape[0]
        t = self.sample_timesteps(B, device=z.device, dtype=z.dtype)
        a, b = self.noise_schedule(t)
        noise = torch.randn_like(z)
        z_t = a * z + b * noise

        if self.block_mask.device != z.device:
            self.block_mask = self.block_mask.to(z.device)

        pred = self.dit(z_t, t, mask=self.block_mask)

        # Target depends on the prediction objective:
        #   v:  paper Eq. 2.1.7  →  u_t = ε - z
        #   x0: x0-prediction    →  target = z
        target = (noise - z) if self.prediction_type == "v" else z
        loss = self.compute_loss(pred, target)

        return {
            "loss": loss,
            "z": z,
            "z_pred": pred,
            "target": target,
            "t_mean": t.mean(),
            "z_global": enc["z_global"],
            "z_local": enc["z_local"],
            "prediction_type": self.prediction_type,
        }
