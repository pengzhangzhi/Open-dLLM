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

"""Qwen3.5/Qwen3.6 model implementation with hybrid linear/full attention.

Implements the Gated DeltaNet linear attention path alongside standard
multi-head attention, with per-layer routing controlled by
`full_attention_interval`. Supports masked diffusion model (MDM) training,
representation alignment distillation, and liger kernel integration.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    replace_return_docstrings,
)

from veomni.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from veomni.models.transformers.qwen3_5.delta_rule import (
    chunk_gated_delta_rule_pytorch,
    fused_recurrent_gated_delta_rule_pytorch,
)
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationMixin

from ....data.constants import IGNORE_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    reduce_sequence_parallel_loss,
    slice_position_embedding,
)
from ....ops.loss import causallm_loss_function
from ....utils import logging
from ....utils.import_utils import is_fla_available, is_liger_kernel_available
from ...module_utils import GradientCheckpointingLayer


if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.rope import liger_rotary_pos_emb


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3.6-27B"
_CONFIG_FOR_DOC = "Qwen3_5Config"


def repr_align_loss_fn(z1, z2):
    z1_norm = nn.functional.normalize(z1, p=2, dim=-1)
    z2_norm = nn.functional.normalize(z2, p=2, dim=-1)
    cosine_sim = (z1_norm * z2_norm).sum(dim=-1)
    return 1.0 - cosine_sim.mean()


# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------

class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling. Weights are zero-initialized so
    the initial forward is effectively identity."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # (1 + weight) — zero-init => no-op at start
        return (torch.tensor(1.0, dtype=input_dtype, device=hidden_states.device) + self.weight) * hidden_states.to(input_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm used in the linear attention path.
    Returns: norm(hidden) * silu(gate(hidden))."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = Qwen3_5RMSNorm(hidden_size, eps)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.norm(hidden_states) * torch.silu(self.gate(hidden_states))


# ---------------------------------------------------------------------------
# Standard MLP (SwiGLU)
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# ---------------------------------------------------------------------------
# Rotary embedding with partial RoPE support
# ---------------------------------------------------------------------------

class Qwen3_5RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5Config, device=None):
        super().__init__()
        self.config = config
        self.partial_rotary_factor = config.partial_rotary_factor
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        # Standard inv_freq for the rotary dim portion
        self.inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim))
        self.max_seq_len_cached = config.max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim, position_ids=None):
    """Apply partial rotary position embedding — only first `rotary_dim` dims."""
    q_partial = q[..., :rotary_dim]
    k_partial = k[..., :rotary_dim]

    # rotate_half inline
    q1 = q_partial[..., : q_partial.shape[-1] // 2]
    q2 = q_partial[..., q_partial.shape[-1] // 2 :]
    k1 = k_partial[..., : k_partial.shape[-1] // 2]
    k2 = k_partial[..., k_partial.shape[-1] // 2 :]

    cos_partial = cos[..., :rotary_dim]
    sin_partial = sin[..., :rotary_dim]

    # cos/sin shapes: (batch, rotary_dim, seq_len) → (batch, seq_len, rotary_dim)
    # for proper broadcasting with (batch, seq_len, features)
    cos_partial = cos_partial.transpose(1, 2)
    sin_partial = sin_partial.transpose(1, 2)

    q_rotated = torch.cat([q1 * cos_partial - q2 * sin_partial, q1 * sin_partial + q2 * cos_partial], dim=-1)
    k_rotated = torch.cat([k1 * cos_partial - k2 * sin_partial, k1 * sin_partial + k2 * cos_partial], dim=-1)

    # Concatenate unrotated remainder
    q_full = torch.cat([q_rotated, q[..., rotary_dim:]], dim=-1)
    k_full = torch.cat([k_rotated, k[..., rotary_dim:]], dim=-1)
    return q_full, k_full


# ---------------------------------------------------------------------------
# Standard Multi-Head Attention with output gating
# ---------------------------------------------------------------------------

class Qwen3_5Attention(nn.Module):
    """Multi-head attention with query-side sigmoid gating.

    q_proj output is doubled: split into query (head_dim * num_heads) and
    gate (head_dim * num_heads). Forward applies `attn_output * sigmoid(gate)`.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # q_proj outputs query+gate concatenated
        q_all = self.q_proj(hidden_states)
        q_proj, gate = q_all.chunk(2, dim=-1)

        query_states = q_proj.view(B, S, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(B, S, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(B, S, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, self.rotary_dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": cos, "cos": sin, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # SDPA or eager
        if self.config._attn_implementation == "flash_attention_2":
            attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask)
        elif self.config._attn_implementation == "sdpa":
            attn_output = self._sdpa_forward(query_states, key_states, value_states, attention_mask)
        else:
            attn_output = self._eager_forward(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.contiguous().view(B, S, -1)

        # Output gating: attn_output * sigmoid(gate)
        gate_sigmoid = torch.sigmoid(gate)
        attn_output = attn_output * gate_sigmoid

        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _flash_attention_forward(self, query, key, value, attention_mask):
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
        return _flash_attention_forward(
            query, key, value, attention_mask,
            query.shape[1], head_dim=self.head_dim,
            softmax_scale=self.scaling, dropout=0.0,
            is_causal=self.is_causal, sliding_window=None,
        )

    def _sdpa_forward(self, query, key, value, attention_mask):
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        return sdpa(query, key, value, attn_mask=attention_mask, dropout=0.0, is_causal=self.is_causal).transpose(1, 2).contiguous()

    def _eager_forward(self, query, key, value, attention_mask):
        B, H, S, D = query.shape
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = F.softmax(scores.float(), dim=-1).to(query.dtype)
        scores = F.dropout(scores, p=0.0, training=self.training)
        output = torch.matmul(scores, value)
        return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Gated DeltaNet — linear attention path
# ---------------------------------------------------------------------------

class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention layer.

    Uses `fla.ops.gated_delta_rule` when available, falls back to
    pure-PyTorch chunk_gated_delta_rule_pytorch.

    The layer applies a 1-D convolution to key and value before the delta
    rule recurrence. This matches the Qwen3.6 architecture where
    `linear_conv_kernel_dim` controls the conv window.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim
        self.linear_num_key_heads = config.linear_num_key_heads
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_conv_kernel_dim = config.linear_conv_kernel_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.num_key_value_groups = self.linear_num_key_heads // self.linear_num_key_heads if self.linear_num_key_heads > 0 else 1
        self.scaling = self.linear_key_head_dim ** -0.5
        self.use_fla = is_fla_available()

        self.q_proj = nn.Linear(config.hidden_size, self.linear_num_key_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.linear_num_key_heads * self.linear_key_head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.linear_num_value_heads * self.linear_value_head_dim, bias=False)
        self.o_proj = nn.Linear(self.linear_num_value_heads * self.linear_value_head_dim, config.hidden_size, bias=False)

        # Input gate for delta rule
        self.gate_proj = nn.Linear(config.hidden_size, self.linear_num_key_heads * self.head_dim, bias=False)
        # Beta (forget gate) projection — learns the decay rate
        self.beta_proj = nn.Linear(config.hidden_size, self.linear_num_key_heads * self.head_dim, bias=False)

        # Conv projections for key and value
        if self.linear_conv_kernel_dim > 1:
            self.k_conv_proj = nn.Linear(self.linear_num_key_heads * self.linear_key_head_dim, self.linear_num_key_heads * self.linear_key_head_dim * self.linear_conv_kernel_dim, bias=False)
            self.v_conv_proj = nn.Linear(self.linear_num_value_heads * self.linear_value_head_dim, self.linear_num_value_heads * self.linear_value_head_dim * self.linear_conv_kernel_dim, bias=False)
            self.k_conv = nn.Conv1d(
                in_channels=self.linear_num_key_heads * self.linear_key_head_dim,
                out_channels=self.linear_num_key_heads * self.linear_key_head_dim,
                kernel_size=self.linear_conv_kernel_dim,
                padding=self.linear_conv_kernel_dim // 2,
                groups=self.linear_num_key_heads,
            )
            self.v_conv = nn.Conv1d(
                in_channels=self.linear_num_value_heads * self.linear_value_head_dim,
                out_channels=self.linear_num_value_heads * self.linear_value_head_dim,
                kernel_size=self.linear_conv_kernel_dim,
                padding=self.linear_conv_kernel_dim // 2,
                groups=self.linear_num_value_heads,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Project
        q = self.q_proj(hidden_states)                       # (B, S, n_key_heads * head_dim)
        k = self.k_proj(hidden_states)                       # (B, S, n_key_heads * key_head_dim)
        v = self.v_proj(hidden_states)                       # (B, S, n_val_heads * val_head_dim)
        g = self.gate_proj(hidden_states)                    # (B, S, n_key_heads * head_dim)
        beta = self.beta_proj(hidden_states)                 # (B, S, n_key_heads * head_dim)

        # Partial RoPE on q and k
        q, k = apply_rotary_pos_emb_partial(q, k, cos, sin, self.rotary_dim)

        # Reshape
        q = q.view(B, S, self.linear_num_key_heads, self.head_dim).transpose(1, 2)     # (B, H, S, D)
        k = k.view(B, S, self.linear_num_key_heads, self.linear_key_head_dim).transpose(1, 2)  # (B, H, S, Dk)
        v = v.view(B, S, self.linear_num_value_heads, self.linear_value_head_dim).transpose(1, 2)  # (B, H, S, Dv)
        g = g.view(B, S, self.linear_num_key_heads, self.head_dim).transpose(1, 2)     # (B, H, S, D)
        beta = beta.view(B, S, self.linear_num_key_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)

        # Apply conv to k and v
        if self.linear_conv_kernel_dim > 1:
            k_reshaped = k.transpose(1, 2).reshape(B * self.linear_num_key_heads, S, self.linear_key_head_dim)
            v_reshaped = v.transpose(1, 2).reshape(B * self.linear_num_value_heads, S, self.linear_value_head_dim)
            k_reshaped = self.k_conv(k_reshaped.transpose(1, 2)).transpose(1, 2).reshape(B, self.linear_num_key_heads, S, self.linear_key_head_dim)
            v_reshaped = self.v_conv(v_reshaped.transpose(1, 2)).transpose(1, 2).reshape(B, self.linear_num_value_heads, S, self.linear_value_head_dim)
            k = k_reshaped
            v = v_reshaped

        # Handle recurrent cache for decode
        if past_key_value is not None:
            state = past_key_value.get(self.layer_idx)
            if state is not None:
                # Cache stores (q_acc, kv_prod) for delta rule state
                q_acc, kv_prod = state
                # Extend sequences
                q = torch.cat([q_acc, q], dim=2)
                k = torch.cat([kv_prod[0], k], dim=2)
                v = torch.cat([kv_prod[1], v], dim=2)
                g = torch.cat([kv_prod[2][:, :, :q_acc.shape[2]], g], dim=2)
                beta = torch.cat([kv_prod[3][:, :, :q_acc.shape[2]], beta], dim=2)

        # Apply delta rule
        if self.use_fla:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_delta
                output, new_state = fla_delta(q, k, v, beta, g)
            except Exception:
                output, new_state = chunk_gated_delta_rule_pytorch(q, k, v, beta, g)
        else:
            output, new_state = chunk_gated_delta_rule_pytorch(q, k, v, beta, g)

        # Reshape output
        output = output.transpose(1, 2).contiguous().view(B, S, self.linear_num_value_heads * self.linear_value_head_dim)
        output = self.o_proj(output)

        # Store cache for decode
        cache_tuple = None
        if past_key_value is not None:
            cache_tuple = (q, (k, v, g, beta))
            past_key_value.update(self.layer_idx, cache_tuple, layer_type="linear")

        return output, cache_tuple


# ---------------------------------------------------------------------------
# Dynamic cache for dual attention types
# ---------------------------------------------------------------------------

class Qwen3_5DynamicCache(DynamicCache):
    """Extends DynamicCache to store linear attention recurrent states."""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
        layer_type: str = "full_attention",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = cache_kwargs or {}

        # For linear attention layers, store recurrent state tuple
        if layer_type == "linear_attention":
            # key_states here is actually (q, (k, v, g, beta))
            new_state = key_states  # already the tuple
            if layer_idx not in self.key_cache:
                self.key_cache[layer_idx] = new_state
            return key_states[1][1], key_states[1][0]  # return v, k for interface compat

        # Standard KV-cache update for full attention
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def get(self, layer_idx):
        """Retrieve cached state for a layer."""
        return self.key_cache.get(layer_idx, None)


# ---------------------------------------------------------------------------
# Decoder layer — dispatches between linear/full attention
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') else "full_attention"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.self_attn = Qwen3_5GatedDeltaNet(config, layer_idx)

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            # Linear attention path uses gated RMSNorm
            self.post_attention_layernorm = Qwen3_5RMSNormGated(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_causal=is_causal,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                cache_position=cache_position,
                is_causal=is_causal,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # delta net doesn't produce attention weights

        return outputs


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3_5RMSNorm):
            module.weight.data.zero_()  # zero-init => (1+0) = identity

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask, sequence_length, target_length, dtype, cache_position, batch_size, config,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@add_start_docstrings(
    "The bare Qwen3_5 Model outputting raw hidden-states.",
    "",
)
class Qwen3_5Model(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward("")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = Qwen3_5DynamicCache()
            elif not isinstance(past_key_values, Qwen3_5DynamicCache):
                new_cache = Qwen3_5DynamicCache()
                new_cache.key_cache = dict(past_key_values.key_cache)
                new_cache._seen_tokens = past_key_values._seen_tokens
                past_key_values = new_cache

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Compute causal mask
        if attention_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        # Position embeddings shared across layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids if position_ids.dim() > 1 else position_ids.squeeze(0))
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_causal=is_causal,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if past_key_values is not None:
            target_length = past_key_values.get_max_cache_shape() if hasattr(past_key_values, 'get_max_cache_shape') else cache_position[-1].item() + sequence_length
        else:
            target_length = attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item() + sequence_length

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask.dim() == 2:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------

class KwargsForCausalLM: ...


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, MDMGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_function = causallm_loss_function
        self.teacher_model = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mask_ratio: Optional[torch.FloatTensor] = None,
        casual_input_ids: Optional[torch.LongTensor] = None,
        casual_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        is_causal: bool = True,
        repr_align_wt: Optional[float] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Forward pass for the Qwen3_5ForCausalLM model.

        Args:
            input_ids: Input tokens
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for generation
            inputs_embeds: Embedded inputs
            labels: Labels for loss computation
            mask_ratio: Mask ratio for diffusion training
            casual_input_ids: Causal input IDs for diffusion
            casual_labels: Causal labels for diffusion
            use_cache: Whether to use cache
            output_attentions: Output attention weights
            output_hidden_states: Output hidden states
            cache_position: Cache positions
            logits_to_keep: Logits to keep for generation
            is_causal: Whether attention is causal
            repr_align_wt: Weight for representation alignment loss

        Returns:
            CausalLMOutputWithPast model output
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        loss_components = {}

        if not get_parallel_state().sp_enabled and labels is not None:
            labels = labels[..., 1:].contiguous()
            labels = labels.view(-1)
            if position_ids is not None and position_ids.size(0) == 1:
                if not (torch.diff(position_ids, dim=-1) >= 0).all():
                    position_ids_ = position_ids.flatten()
                    indices_q = torch.arange(position_ids_.size(0), device=position_ids_.device, dtype=torch.int32)
                    cu_seq_lens = torch.cat(
                        (indices_q[position_ids_ == 0], torch.tensor(position_ids_.size(), device=position_ids_.device, dtype=torch.int32))
                    )
                    labels[cu_seq_lens[1:-1] - 1] = IGNORE_INDEX

        if mask_ratio is not None:
            is_causal = False
            mask_ratio = mask_ratio[..., 1:].contiguous()

        if (self.teacher_model is not None and repr_align_wt is not None and repr_align_wt > 0 and self.training):
            output_hidden_states = True

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            is_causal=is_causal,
        )

        hidden_states = outputs[0]

        # Representation alignment loss
        repr_align_loss = None
        teacher_outputs = None
        if (self.teacher_model is not None and repr_align_wt is not None and repr_align_wt > 0
                and self.training and labels is not None and mask_ratio is not None):
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=casual_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    labels=casual_labels,
                    is_causal=True,
                )

            student_hidden_states = outputs.hidden_states
            teacher_hidden_states = teacher_outputs.hidden_states

            loss_mask = (labels != IGNORE_INDEX)
            if loss_mask.any():
                student_stacked = torch.cat([h[..., :-1, :] for h in student_hidden_states], dim=0).permute(1, 0, 2)
                student_stacked = student_stacked[loss_mask]
                teacher_stacked = torch.cat([h[..., :-1, :] for h in teacher_hidden_states], dim=0).permute(1, 0, 2)
                teacher_stacked = teacher_stacked[loss_mask]

                repr_align_loss = repr_align_loss_fn(student_stacked, teacher_stacked)

        # Logits / loss computation
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        loss = None
        logits = None
        teacher_loss = None

        if labels is not None:
            labels = labels.view(-1)
            if is_liger_kernel_available():
                if mask_ratio is not None:
                    loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
                    if not get_parallel_state().sp_enabled:
                        hidden_states = hidden_states[..., :-1, :].contiguous()
                    token_loss = loss_fct(
                        self.lm_head.weight,
                        hidden_states.view(-1, self.config.hidden_size),
                        labels,
                    )
                    loss_mask = labels != IGNORE_INDEX
                    denom = loss_mask.sum() + 1e-8
                    mdm_loss = (token_loss * loss_mask).sum() / denom
                    path_loss = (-token_loss).exp().detach() * token_loss
                    path_loss = (path_loss * loss_mask).sum() / denom

                    loss = mdm_loss + path_loss
                    loss_components["mdm"] = mdm_loss.detach()
                    loss_components["path"] = path_loss.detach()
                else:
                    loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="mean", ignore_index=IGNORE_INDEX)
                    if not get_parallel_state().sp_enabled:
                        hidden_states = hidden_states[..., :-1, :].contiguous()
                    hidden_states = hidden_states.view(-1, self.config.hidden_size)
                    loss = loss_fct(self.lm_head.weight, hidden_states, labels)
                    loss_components["ar"] = loss.detach()
            else:
                raise ValueError("liger kernel is not available for training.")

            if get_parallel_state().sp_enabled:
                num_valid_tokens = (labels != IGNORE_INDEX).sum()
                loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)

        else:
            logits = self.lm_head(hidden_states)

        if repr_align_loss is not None:
            loss = loss + repr_align_wt * repr_align_loss
            loss_components["repr_align"] = repr_align_loss.detach()
            if teacher_outputs is not None and teacher_outputs.loss is not None:
                teacher_loss = teacher_outputs.loss
                loss_components["teacher"] = teacher_loss.detach()

        if "mdm" not in loss_components and loss is not None and labels is not None:
            loss_components["mdm"] = loss.detach()

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        if loss_components:
            result.loss_components = {k: v.detach().float().item() for k, v in loss_components.items()}
        else:
            result.loss_components = {}

        return result


if is_liger_kernel_available():
    Qwen3_5RMSNorm = LigerFusedLinearCrossEntropyLoss  # placeholder — liger doesn't have RMSNorm replacement for this variant

ModelClass = Qwen3_5ForCausalLM

__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5Model", "Qwen3_5PreTrainedModel"]
