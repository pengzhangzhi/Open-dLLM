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
Cola DLM (Continuous Latent Diffusion Language Model) — Open-dLLM port.

Paper: Cola DLM (arXiv:2605.06548). The paper proposes a Text VAE that
compresses tokens into multi-scale continuous latents, plus a block-causal
DiT that models the global semantic prior over those latents.

This package adapts that recipe as an **opt-in auxiliary head on top of
Repr-Align** (rather than as a stand-alone model):

    Frozen / Repr-Aligned LM hidden states
                 │
                 ▼
        ┌────────────────┐
        │ TextVAEEncoder │  global + local Perceivers, fused
        └────────┬───────┘
                 │ z = [z_global ; z_local]
                 ▼
       ┌──────────────────┐
       │  BlockCausalDiT  │  bidirectional within blocks, causal across
       └──────────────────┘
                 │
                 ▼
           L_cola (MSE)

`ColaDLMHead` packages encoder + DiT + loss. `ColaReprAlignWrapper` bundles
a base Repr-Align LM with the head as a drop-in `nn.Module` so the
existing trainer in `tasks/train_torch.py` only needs an opt-in flag
(`train.cola_wt > 0`) to enable it.

LDLM (`veomni.models.ldlm`) is intentionally not touched.
"""

from .block_causal_dit import BlockCausalDiT, ColaDLMHead, make_block_causal_mask
from .card_head import CardColaDLMHead, make_soft_tail_mask
from .fast_block_head import FastBlockColaDLMHead, make_complementary_mask
from .modules import CrossAttention, FeedForward, MaskedSelfAttention, PerceiverResampler, PreNorm
from .text_vae import TextVAEEncoder
from .wrapper import ColaReprAlignWrapper


def build_cola_head(dim: int, variant: str = "block_causal", **kwargs):
    """Factory: build a ColaDLM head for the given variant.

    Args:
        dim: hidden size of the parent LM.
        variant: one of 'block_causal', 'card', 'fast_block'.
        **kwargs: forwarded to the head constructor (num_global, block_size, etc.)
    """
    if variant == "block_causal":
        return ColaDLMHead(dim=dim, **kwargs)
    elif variant == "card":
        lambda_tail = kwargs.pop("lambda_tail", 0.6)
        return CardColaDLMHead(dim=dim, lambda_tail=lambda_tail, **kwargs)
    elif variant == "fast_block":
        use_complementary = kwargs.pop("use_complementary", True)
        return FastBlockColaDLMHead(dim=dim, use_complementary=use_complementary, **kwargs)
    else:
        raise ValueError(f"Unknown cola_variant: {variant!r}. Choose from: block_causal, card, fast_block")


__all__ = [
    "CrossAttention",
    "FeedForward",
    "MaskedSelfAttention",
    "PerceiverResampler",
    "PreNorm",
    "TextVAEEncoder",
    "BlockCausalDiT",
    "ColaDLMHead",
    "ColaReprAlignWrapper",
    "CardColaDLMHead",
    "FastBlockColaDLMHead",
    "make_block_causal_mask",
    "make_soft_tail_mask",
    "make_complementary_mask",
    "build_cola_head",
]
