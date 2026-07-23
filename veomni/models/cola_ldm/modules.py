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
Small, self-contained Perceiver + attention + DiT building blocks.

Perceiver pieces (`PerceiverResampler` etc.) are duplicated from
`veomni.models.ldlm.autoencoder` (rather than imported) so the LDLM
package can remain frozen and this extension can evolve independently.
`MaskedSelfAttention` adds the bool-mask support needed for block-causal
diffusion.

DiT pieces (`TimestepEmbedding`, `AdaLN`, `MLPGeluTanh`) are lifted from
the official Cola DLM release (`cola_dlm/modeling_cola_dit.py`,
arXiv:2605.06548) and trimmed to the bits the Open-dLLM auxiliary head
actually needs (no KV cache, no RoPE, no variable-length NA layout).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNorm(nn.Module):
    """LayerNorm before the wrapped module. Forwards extra kwargs."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if context is not None:
            return self.fn(self.norm(x), context=context, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """GELU MLP with 4x hidden expansion by default."""

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention: queries from x, key/values from context."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads

        q = self.to_q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class MaskedSelfAttention(nn.Module):
    """Self-attention with optional (N, N) or (B, N, N) bool mask (True = attend)."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads

        qkv = self.to_qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # broadcast (N, N) -> (1, 1, N, N); (B, N, N) -> (B, 1, N, N)
            if mask.dim() == 2:
                attn = attn.masked_fill(~mask[None, None, :, :], float("-inf"))
            elif mask.dim() == 3:
                attn = attn.masked_fill(~mask[:, None, :, :], float("-inf"))
            else:
                raise ValueError(f"mask must be 2D or 3D bool, got {mask.shape}")
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver: learnable latent queries cross-attend to a variable-length context."""

    def __init__(
        self,
        dim: int,
        num_latents: int,
        depth: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, MaskedSelfAttention(dim, heads=heads)),
                    PreNorm(dim, CrossAttention(dim, heads=heads)),
                    PreNorm(dim, FeedForward(dim, hidden_mult=ff_mult)),
                ])
            )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        x = self.latents.expand(B, -1, -1).to(dtype=context.dtype, device=context.device)
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


# ===========================================================================
# DiT primitives lifted from the official Cola DLM release
# (cola_dlm/modeling_cola_dit.py, arXiv:2605.06548).
#
# Trimmed: no KV cache, no RoPE, no NA variable-length layout. The
# auxiliary head trains on fixed-shape latents (G + L tokens per
# sample), so the simple form is enough.
# ===========================================================================


def get_sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (diffusers convention,
    `flip_sin_to_cos=False`, `downscale_freq_shift=0`).

    Matches the convention used by the official Cola trainer. Using
    `half_dim - 1` here would shift every frequency slightly and
    silently desync the AdaLN conditioning path.
    """
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal projection → 3-layer SiLU MLP → `output_dim`."""

    def __init__(self, sinusoidal_dim: int = 256, hidden_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, timestep: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if timestep.dim() == 0:
            timestep = timestep[None]
        out_dtype = dtype or timestep.dtype
        emb = get_sinusoidal_embedding(timestep, self.sinusoidal_dim).to(out_dtype)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        emb = self.proj_out(emb)
        return emb  # (B, output_dim)


class AdaLN(nn.Module):
    """AdaLN-Zero conditioning (DiT-style, per the official Cola repo).

    For each named layer (e.g. `"msa"`, `"mlp"`) this module owns two
    sub-projections:
      - `<layer>_in`:  SiLU → Linear(dim, 2*dim) — produces (shift, scale).
                       Applied as `norm(x) * (1 + scale) + shift`.
      - `<layer>_out`: SiLU → Linear(dim, dim) — gates the residual.
                       Applied as `x * gate + residual`.

    The final Linears in both paths are zero-initialised so each block
    starts as an identity (AdaLN-Zero).
    """

    def __init__(self, dim: int, emb_dim: int, layers: List[str], modes: Optional[List[str]] = None):
        super().__init__()
        if modes is None:
            modes = ["in", "out"]
        self.dim = dim
        self.layers = layers
        self.modes = modes
        for layer in layers:
            if "in" in modes:
                self.register_module(f"{layer}_in", nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * dim, bias=True)))
            if "out" in modes:
                self.register_module(f"{layer}_out", nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim, bias=True)))
        self._zero_init()

    def _zero_init(self):
        for layer in self.layers:
            for m in self.modes:
                last = getattr(self, f"{layer}_{m}")[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        hid: torch.Tensor,
        emb: torch.Tensor,
        layer: str,
        mode: str,
        norm_layer: Optional[nn.Module] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hid: (B, N, dim)
        emb: (B, emb_dim)  — one per sample (timestep embedding)
        """
        assert layer in self.layers and mode in self.modes
        emb_proj = getattr(self, f"{layer}_{mode}")(emb)  # (B, 2*dim) or (B, dim)
        emb_proj = emb_proj.unsqueeze(1)  # (B, 1, *)  broadcast over N
        if mode == "in":
            shift, scale = emb_proj.chunk(2, dim=-1)
            return norm_layer(hid) * (1 + scale) + shift
        # mode == "out"
        return hid * emb_proj + residual


class MLPGeluTanh(nn.Module):
    """GELU(tanh approximation) MLP — matches the Cola DiT FFN exactly."""

    def __init__(self, dim: int, expand_ratio: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(x)))
