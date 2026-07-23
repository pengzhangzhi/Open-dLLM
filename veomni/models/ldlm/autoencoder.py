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
LDLM (Latent Diffusion Language Model) autoencoder module.

Implements the Perceiver-based latent encoder/decoder that compresses
Qwen3.6 hidden states into a compact latent space for diffusion modeling.

Reference: LDLM (arXiv:2605.07933)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Perceiver components
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Pre-normalization wrapper for any module."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if context is not None:
            return self.fn(self.norm(x), context=context, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Simple MLP with GELU activation."""
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
    """Cross-attention: queries attend to key-value pairs."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
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

        q = self.to_q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)   # (B, H, N, d)
        k = self.to_k(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention with pre-norm."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads

        qkv = self.to_qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler that maps a variable-length input sequence
    to a fixed set of latent tokens via cross-attention.

    Architecture (from Flamingo / LDLM):
        latents -> self-attention -> cross-attention(latents, context) -> FFN
    """
    def __init__(
        self,
        dim: int,
        num_latents: int = 128,
        depth: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads=heads)),
                PreNorm(dim, CrossAttention(dim, heads=heads)),
                PreNorm(dim, FeedForward(dim, hidden_mult=ff_mult)),
            ]))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, T, dim) input sequence from encoder
        Returns:
            latents: (B, num_latents, dim) compressed representation
        """
        B = context.shape[0]
        x = self.latents.expand(B, -1, -1)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context) + x
            x = ff(x) + x

        return x


# ---------------------------------------------------------------------------
# LDLM Autoencoder
# ---------------------------------------------------------------------------

class LDLMAutoencoder(nn.Module):
    """
    Latent Diffusion Language Model autoencoder.

    Encodes Qwen3.6 hidden states into a compact latent space via a
    Perceiver resampler, then decodes back to token predictions.

    Architecture:
        Qwen3.6 encoder (frozen) -> Perceiver encoder -> latent z
        -> Perceiver decoder -> Transformer decoder -> LM head
    """
    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen3.6-35B-A3B",
        seq_len: int = 128,
        depth: int = 6,
        decoder_input_noise_std: float = 3.0,
        latent_dim: Optional[int] = None,
        encoder_hidden_layer: int = -3,
        decoder_num_layers: int = 3,
        perceiver_heads: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.decoder_input_noise_std = decoder_input_noise_std
        self.encoder_hidden_layer = encoder_hidden_layer

        self.register_buffer("_h_mean", torch.zeros(1))
        self.register_buffer("_h_var", torch.ones(1))
        self.register_buffer("_h_count", torch.tensor(0.0))
        self._normalize_hidden_states = True

        self.token_encoder = AutoModel.from_pretrained(
            encoder_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self._encoder_device_map = None
        for p in self.token_encoder.parameters():
            p.requires_grad = False

        cfg = self.token_encoder.config
        # Handle nested config (MoE variants use text_config)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.dim = cfg.text_config.hidden_size
            vocab_size = cfg.text_config.vocab_size
        else:
            self.dim = cfg.hidden_size
            vocab_size = cfg.vocab_size
        self.latent_dim = latent_dim or self.dim
        self._vocab_size = vocab_size

        # Compute nhead from dim (must divide evenly)
        for nhead_candidate in [16, 12, 8, 4, 2]:
            if self.dim % nhead_candidate == 0:
                decoder_nhead = nhead_candidate
                break
        else:
            decoder_nhead = 8  # safe fallback

        # Perceiver-based latent encoder/decoder
        self.latent_encoder = PerceiverResampler(
            dim=self.dim,
            num_latents=seq_len,
            depth=depth,
            heads=perceiver_heads,
        )
        self.latent_decoder = PerceiverResampler(
            dim=self.dim,
            num_latents=seq_len,
            depth=depth,
            heads=perceiver_heads,
        )

        # Lightweight token decoder (transformer decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.dim, nhead=decoder_nhead, dim_feedforward=self.dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_layers,
            norm=nn.LayerNorm(self.dim),
        )
        self.lm_head = nn.Linear(self.dim, self._vocab_size)

    def move_encoder_to_gpus(self, max_memory=None):
        import transformers.modeling_utils as mu
        _orig_warmup = mu.caching_allocator_warmup
        mu.caching_allocator_warmup = lambda *a, **k: None
        try:
            from transformers import AutoModel
            self.token_encoder = AutoModel.from_pretrained(
                self.token_encoder.config._name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
            for p in self.token_encoder.parameters():
                p.requires_grad = False
            self._encoder_device_map = dict(self.token_encoder.hf_device_map)
        finally:
            mu.caching_allocator_warmup = _orig_warmup

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        device = self.latent_encoder.latents.device
        with torch.no_grad():
            if self._encoder_device_map is not None:
                encoder_device = self.token_encoder.device
                outputs = self.token_encoder(
                    input_ids.to(encoder_device),
                    attention_mask=attention_mask.to(encoder_device) if attention_mask is not None else None,
                    output_hidden_states=True,
                )
                h = outputs.hidden_states[self.encoder_hidden_layer].to(device)
            else:
                outputs = self.token_encoder(
                    input_ids.cpu(),
                    attention_mask=attention_mask.cpu() if attention_mask is not None else None,
                    output_hidden_states=True,
                )
                h = outputs.hidden_states[self.encoder_hidden_layer].to(device)
            del outputs

        if self._normalize_hidden_states and self.training:
            h_mean = self._h_mean.to(h.device)
            h_var = self._h_var.to(h.device)
            h_count = self._h_count.to(h.device)
            batch_mean = h.mean()
            batch_var = h.var(unbiased=False)
            count = h.numel()
            new_count = h_count + count
            h_mean = h_mean * (h_count / new_count) + batch_mean * (count / new_count)
            h_var = h_var * (h_count / new_count) + batch_var * (count / new_count)
            self._h_mean.copy_(h_mean)
            self._h_var.copy_(h_var)
            self._h_count.copy_(new_count)

        if self._normalize_hidden_states and self._h_count > 0:
            h = (h - self._h_mean.to(h.device)) / (self._h_var.to(h.device).sqrt().clamp(min=1e-6))

        z0 = self.latent_encoder(h)
        return {"z0": z0, "h": h}

    def decode(self, z: torch.Tensor, training: bool = True) -> Dict:
        """
        Decode latent back to token predictions.

        Args:
            z: (B, seq_len, dim) latent representation
            training: if True, add noise to latent before decoding
        Returns:
            dict with 'logits', 'h_hat', 'noise'
        """
        if training and self.decoder_input_noise_std > 0:
            noise = torch.randn_like(z) * self.decoder_input_noise_std
            z_noisy = z + noise
        else:
            z_noisy = z
            noise = None

        h_hat = self.latent_decoder(z_noisy)

        # Autoregressive token prediction
        h_detached = h_hat.detach() if training else h_hat
        B, T, D = h_hat.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(h_hat.device)
        token_hidden = self.token_decoder(
            tgt=h_detached,
            memory=h_detached,
            tgt_mask=causal_mask,
        )
        logits = self.lm_head(token_hidden)

        return {
            "logits": logits,
            "h_hat": h_hat,
            "noise": noise,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, training: bool = True) -> Dict:
        """
        Full forward pass: encode -> (noise) -> decode.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: optional (B, T) mask
            training: if True, apply decoder input noise
        Returns:
            dict with 'z0', 'h_hat', 'logits', 'h', 'noise'
        """
        enc_out = self.encode(input_ids, attention_mask)
        dec_out = self.decode(enc_out["z0"], training=training)

        return {
            "z0": enc_out["z0"],
            "h": enc_out["h"],
            "h_hat": dec_out["h_hat"],
            "logits": dec_out["logits"],
            "noise": dec_out["noise"],
        }


# ---------------------------------------------------------------------------
# Diffusion Head (Lightweight DiT)
# ---------------------------------------------------------------------------

class DiffusionHead(nn.Module):
    """
    Lightweight Diffusion Transformer that operates on latent space.

    Can be attached on top of LDLMAutoencoder latents for
    diffusion-based generation in latent space.

    Supports self-conditioning (LDLM Section 4.2): with probability 0.5,
    the denoiser receives its own previous estimate as additional input.
    """
    def __init__(self, dim: int = 2048, depth: int = 12, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.proj_in = nn.Linear(dim * 2, dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SelfAttention(dim, heads=heads),
                nn.LayerNorm(dim),
                FeedForward(dim),
            ]))

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, z_hat_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, dim) noisy latent
            t: (B,) or (B, 1) timestep in [0, 1]
            z_hat_prev: (B, T, dim) optional previous denoised estimate for self-conditioning
        Returns:
            denoised prediction of same shape as z
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.float()).unsqueeze(1)

        if z_hat_prev is not None:
            x = self.proj_in(torch.cat([z, z_hat_prev], dim=-1))
        else:
            x = z
        x = x + t_emb
        for norm1, attn, norm2, ff in self.blocks:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x

        return self.out_norm(x)


ModelClass = LDLMAutoencoder
