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

"""Qwen3.5/Qwen3.6 MoE model (Qwen3.6-35B-A3B).

Shares Qwen3_5RMSNorm, Qwen3_5RMSNormGated, Qwen3_5MLP, Qwen3_5GatedDeltaNet,
and Qwen3_5Attention with the dense qwen3_5 package. Adds MoE-specific classes
for expert routing, load-balancing aux loss, and shared expert.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings,
    can_return_tuple,
    replace_return_docstrings,
)

from veomni.models.transformers.qwen2.generation_utils import MDMGenerationMixin
from veomni.models.transformers.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    get_parallel_state,
    is_liger_kernel_available,
    reduce_sequence_parallel_loss,
    repr_align_loss_fn,
)
from veomni.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from ....data.constants import IGNORE_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import reduce_sequence_parallel_loss, slice_position_embedding
from ....ops.loss import causallm_loss_function
from ....utils import logging
from ...module_utils import GradientCheckpointingLayer


if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3.6-35B-A3B"
_CONFIG_FOR_DOC = "Qwen3_5MoeConfig"


# ---------------------------------------------------------------------------
# MoE components
# ---------------------------------------------------------------------------

class Qwen3_5MoeTopKRouter(nn.Module):
    """Top-k expert router with load-balancing aux loss."""

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=2 ** 0.5)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns topk_idx, topk_weights, aux_loss.

        Args:
            hidden_states: (B, S, H)

        Returns:
            topk_idx: (B, S, k) — token indices of selected experts
            topk_weights: (B, S, k) — normalized gating weights
            aux_loss: float — load balancing aux loss (or None)
        """
        B, S, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)  # (BS, H)

        logits = F.linear(hidden_states, self.weight)  # (BS, num_experts)

        # Select top-k experts
        k = self.num_experts_per_tok
        topk_weights, topk_idx = torch.topk(logits, k=k, dim=-1)  # (BS, k)

        # Normalize weights
        if self.norm_topk_prob and k > 1:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_idx = topk_idx.contiguous()

        # Load-balancing aux loss
        aux_loss = self._compute_aux_loss(logits, topk_idx, B * S)

        return topk_idx, topk_weights, aux_loss

    def _compute_aux_loss(self, logits, topk_idx, total_tokens):
        """Compute load balancing auxiliary loss.

        Encourages uniform expert utilization by penalizing
        P(token) * expert_probability product.
        """
        num_experts = self.num_experts
        k = self.num_experts_per_tok

        # Probability distribution over experts
        logits_dense = logits.float()
        prob = F.softmax(logits_dense, dim=-1)  # (BS, num_experts)

        # One-hot for top-k selected
        selected_one_hot = torch.zeros_like(prob)  # (BS, num_experts)
        # Set only top-k positions
        topk_idx_expanded = topk_idx.unsqueeze(-1)  # (BS, k, 1)
        selected_one_hot.scatter_(-1, topk_idx_expanded, 1.0 / k)  # (BS, num_experts)

        # Fraction of tokens routed to each expert
        expert_fraction = selected_one_hot.sum(dim=0) / total_tokens  # (num_experts,)

        # Probability of routing to each expert (average probability)
        prob_mean = prob.mean(dim=0)  # (num_experts,)

        aux_loss = prob_mean.dot(expert_fraction) * num_experts
        return aux_loss * self.router_aux_loss_coef


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with routed experts and gated shared expert."""

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.shared_expert_intermediate_size = config.shared_expert_intermediate_size

        self.router = Qwen3_5MoeTopKRouter(config)

        # Routed experts — each expert is a SwiGLU MLP
        self.experts = nn.ModuleList(
            [Qwen3_5MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )

        # Gated shared expert
        self.shared_expert = Qwen3_5MLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, self.shared_expert_intermediate_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through MoE block.

        Args:
            hidden_states: (B, S, H)

        Returns:
            output: (B, S, H) — expert output
            aux_loss: float — load balancing aux loss
        """
        B, S, H = hidden_states.shape
        orig_shape = hidden_states.shape

        topk_idx, topk_weights, aux_loss = self.router(hidden_states)

        # Flatten
        hidden_states_flat = hidden_states.view(-1, H)  # (BS, H)
        topk_idx_flat = topk_idx.view(-1)  # (BS * k,)
        topk_weights_flat = topk_weights.view(-1)  # (BS * k,)

        # Gather unique expert indices
        unique_experts = torch.unique(topk_idx_flat)

        # Accumulate expert outputs
        output = torch.zeros_like(hidden_states_flat)  # (BS, H)
        num_tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=hidden_states.device)

        for expert_idx in unique_experts:
            # Mask for tokens assigned to this expert
            mask = (topk_idx_flat == expert_idx)  # (BS * k,)
            num_tokens = mask.sum().item()
            if num_tokens == 0:
                continue

            num_tokens_per_expert[expert_idx] = num_tokens

            # Gather input tokens for this expert
            tokens_for_expert = hidden_states_flat[mask]  # (n, H)
            weights_for_expert = topk_weights_flat[mask]  # (n,)

            # Forward through expert
            expert_output = self.experts[expert_idx.item()](tokens_for_expert)  # (n, H)

            # Apply gating weights and accumulate
            expert_output = expert_output * weights_for_expert.unsqueeze(-1)
            output.scatter_add_(0, mask.unsqueeze(-1).expand_as(output), expert_output)

        # Reshape back
        output = output.view(*orig_shape)

        # Add gated shared expert: shared_expert(silu(gate(x)) * x)
        gate_output = torch.silu(self.shared_expert_gate(hidden_states))  # (B, S, shared_dim)
        shared_output = self.shared_expert(gate_output)  # (B, S, H)

        output = output + shared_output

        return output, aux_loss


# ---------------------------------------------------------------------------
# Decoder layer with MoE
# ---------------------------------------------------------------------------

class Qwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types) else "full_attention"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.self_attn = Qwen3_5GatedDeltaNet(config, layer_idx)

        if self.layer_type == "full_attention":
            self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3_5RMSNormGated(config.hidden_size, eps=config.rms_norm_eps)

        # MoE block replaces MLP
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, aux_loss = self.mlp(hidden_states)
        hidden_states = residual + moe_output

        outputs = (hidden_states, aux_loss)
        if output_attentions:
            outputs += (None,)

        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3_5MoePreTrainedModel(Qwen3_5PreTrainedModel):
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5MoeDecoderLayer"]


@add_start_docstrings("The bare Qwen3_5Moe Model.", "")
class Qwen3_5MoeModel(Qwen3_5MoePreTrainedModel):
    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None  # Will be created in forward

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
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
                past_key_values = new_cache

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create rotary embedding
        if self.rotary_emb is None:
            from veomni.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5RotaryEmbedding
            self.rotary_emb = Qwen3_5RotaryEmbedding(config=self.config)

        if attention_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids if position_ids.dim() > 1 else position_ids.squeeze(0))
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)

        all_hidden_states = () if output_hidden_states else None
        all_aux_losses = []

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
            if layer_outputs[1] is not None:
                all_aux_losses.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        ), all_aux_losses

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

class Qwen3_5MTPHead(nn.Module):
    """Qwen3.6-style NextN Multi-Token Prediction head.

    Lightweight: 1 decoder layer + RMSNorm + tied lm_head.
    Takes hidden_states from the main model, predicts next-N tokens.
    """

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.num_layers = config.mtp_num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, layer_idx=config.num_hidden_layers + i)
            for i in range(self.num_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        for layer in self.layers:
            layer_outputs = layer(hidden_states, **kwargs)
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


class Qwen3_5MoeForCausalLM(Qwen3_5MoePreTrainedModel, MDMGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_function = causallm_loss_function
        self.teacher_model = None
        self.align_layers: Optional[list] = None
        self.repr_align_sub_sample_ratio: float = 1.0
        self.repr_align_num_sample_layers: Optional[int] = None

        # MTP (Multi-Token Prediction, Qwen3.6-style NextN)
        self.mtp_head = None
        if getattr(config, "mtp_num_layers", 0) > 0:
            self.mtp_head = Qwen3_5MTPHead(config)

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
        Forward pass for the Qwen3_5MoeForCausalLM model.

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

        # Model forward — returns (output, aux_losses)
        model_outputs, aux_losses = self.model(
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

        hidden_states = model_outputs[0]
        total_aux_loss = sum(aux_losses) if aux_losses else None

        # Representation alignment loss
        repr_align_loss = None
        teacher_outputs = None
        if (self.teacher_model is not None and repr_align_wt is not None and repr_align_wt > 0
                and self.training and labels is not None and mask_ratio is not None):
            with torch.no_grad():
                teacher_model_out = self.teacher_model(
                    input_ids=casual_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    labels=casual_labels,
                    is_causal=True,
                )

            student_hidden_states = model_outputs.hidden_states
            teacher_hidden_states = teacher_model_out.hidden_states

            if self.align_layers is not None:
                student_hidden_states = tuple(student_hidden_states[i] for i in self.align_layers)
                teacher_hidden_states = tuple(teacher_hidden_states[i] for i in self.align_layers)

            loss_mask = (labels != IGNORE_INDEX)
            if loss_mask.any():
                student_stacked = torch.cat([h[..., :-1, :] for h in student_hidden_states], dim=0).permute(1, 0, 2)
                student_stacked = student_stacked[loss_mask]
                teacher_stacked = torch.cat([h[..., :-1, :] for h in teacher_hidden_states], dim=0).permute(1, 0, 2)
                teacher_stacked = teacher_stacked[loss_mask]

                # Layer subsampling: randomly pick k of L layers each step
                if self.repr_align_num_sample_layers is not None:
                    num_layers = student_stacked.size(1)
                    k = min(self.repr_align_num_sample_layers, num_layers)
                    layer_idx = torch.randperm(num_layers, device=student_stacked.device)[:k]
                    student_stacked = student_stacked[:, layer_idx, :]
                    teacher_stacked = teacher_stacked[:, layer_idx, :]

                # Token subsampling: randomly pick fraction of V tokens each step
                if self.repr_align_sub_sample_ratio < 1.0:
                    num_valid = student_stacked.size(0)
                    num_sample = max(1, int(num_valid * self.repr_align_sub_sample_ratio))
                    token_idx = torch.randperm(num_valid, device=student_stacked.device)[:num_sample]
                    student_stacked = student_stacked[token_idx]
                    teacher_stacked = teacher_stacked[token_idx]

                repr_align_loss = repr_align_loss_fn(student_stacked, teacher_stacked)

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

            # Add aux loss from MoE load balancing
            if total_aux_loss is not None:
                loss = loss + total_aux_loss
                loss_components["aux"] = total_aux_loss.detach()

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

        # MTP auxiliary loss
        mtp_loss_val = None
        if self.mtp_head is not None and self.training and loss is not None:
            mtp_logits = self.mtp_head(model_outputs[0], attention_mask=attention_mask, position_ids=position_ids)
            mtp_labels = input_ids[:, self.config.mtp_n_predict:].contiguous()
            mtp_labels = mtp_labels.view(-1)
            mtp_logits_flat = mtp_logits[:, :mtp_labels.shape[0] // mtp_labels.shape[0] if mtp_labels.shape[0] > 0 else 0, :]
            # Simple CE loss on MTP predictions
            mtp_logits_trimmed = mtp_logits[:, :mtp_labels.shape[0], :].contiguous().view(-1, self.vocab_size)
            if mtp_labels.shape[0] > 0 and mtp_logits_trimmed.shape[0] >= mtp_labels.shape[0]:
                mtp_logits_trimmed = mtp_logits_trimmed[:mtp_labels.shape[0]]
                mtp_loss_val = F.cross_entropy(mtp_logits_trimmed, mtp_labels, ignore_index=IGNORE_INDEX)
                loss = loss + self.config.mtp_loss_weight * mtp_loss_val
                loss_components["mtp"] = mtp_loss_val.detach()

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

        if loss_components:
            result.loss_components = {k: v.detach().float().item() for k, v in loss_components.items()}
        else:
            result.loss_components = {}

        return result


# The HF model snapshot for Qwen3.6-35B-A3B declares
# architectures=['Qwen3_5MoeForConditionalGeneration'] but for text-only
# Repr-Align training we want our bidirectional CausalLM.  This alias lets the
# veomni registry intercept the ConditionalGeneration arch name and route it to
# our text-only class, which simply ignores the vision weights in the checkpoint.
class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForCausalLM):
    pass


ModelClass = [Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration]

__all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
]
