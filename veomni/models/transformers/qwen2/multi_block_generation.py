from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.utils import logging

from .generation_utils import MDMGenerationConfig, sample_tokens


logger = logging.get_logger(__name__)


class MultiBlockDecoderConfig(MDMGenerationConfig):
    def __init__(self, **kwargs):
        self.block_size = kwargs.pop("block_size", 32)
        self.block_add_threshold = kwargs.pop("block_add_threshold", 0.5)
        self.decoded_token_threshold = kwargs.pop("decoded_token_threshold", 0.5)
        self.entropy_threshold = kwargs.pop("entropy_threshold", 0.9)
        self.early_stop = kwargs.pop("early_stop", False)
        self.use_kv_cache = kwargs.pop("use_kv_cache", False)
        super().__init__(**kwargs)


def create_block_causal_mask(
    prompt_length: int,
    max_length: int,
    block_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dtype = dtype or torch.bfloat16
    mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    mask[:, :, :prompt_length, :prompt_length] = 0
    remaining = max_length - prompt_length
    num_blocks = (remaining + block_size - 1) // block_size
    for b in range(num_blocks):
        b_start = prompt_length + b * block_size
        b_end = min(b_start + block_size, max_length)
        mask[:, :, b_start:b_end, :prompt_length] = 0
        for pb in range(b):
            pb_start = prompt_length + pb * block_size
            pb_end = min(pb_start + block_size, max_length)
            mask[:, :, b_start:b_end, pb_start:pb_end] = 0
        mask[:, :, b_start:b_end, b_start:b_end] = 0
    return mask


def extract_attention_mask(
    full_mask: torch.Tensor, start_pos: int, input_length: int, cache_length: int
) -> torch.Tensor:
    end_pos = start_pos + input_length
    total = cache_length + input_length
    extracted = torch.full(
        (1, 1, input_length, total), -torch.inf, device=full_mask.device, dtype=full_mask.dtype
    )
    extracted[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted


def handle_early_stop(x, block_states, eos_token_id, prompt_length, mask_token_id=None):
    if eos_token_id is None:
        return False, None
    gen_region = x[:, prompt_length:]
    eos_mask = gen_region == eos_token_id
    if not eos_mask.any():
        return False, None
    pos = torch.arange(gen_region.shape[1], device=x.device).unsqueeze(0)
    first_eos_rel = torch.where(eos_mask, pos, gen_region.shape[1]).amin(dim=1)
    first_eos_abs = prompt_length + first_eos_rel[0].item()
    x[:, first_eos_abs + 1 :] = eos_token_id
    for bid in sorted(block_states.keys()):
        if bid == 0:
            continue
        s, e = block_states[bid]["start"], block_states[bid]["end"]
        if s > first_eos_abs:
            block_states[bid]["mask_count"] = 0
            block_states[bid]["is_complete"] = True
        elif e > first_eos_abs and s <= first_eos_abs:
            if mask_token_id is not None:
                masks_before = (x[:, s : first_eos_abs + 1] == mask_token_id).sum().item()
                block_states[bid]["mask_count"] = masks_before
                if masks_before == 0:
                    block_states[bid]["is_complete"] = True
    return True, first_eos_abs


class MultiBlockDecoderMixin:
    @staticmethod
    def _make_block_state(start, end, is_complete=False):
        return {
            "start": start, "end": end,
            "mask_count": end - start, "total_masks": end - start,
            "is_complete": is_complete,
        }

    @torch.no_grad()
    def generate_multi_block(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MultiBlockDecoderConfig] = None,
        **kwargs,
    ) -> torch.Tensor:
        gen_config = self._mdm_prepare_generation_config(generation_config, **kwargs)
        if not isinstance(gen_config, MultiBlockDecoderConfig):
            gen_config = MultiBlockDecoderConfig(**gen_config.to_dict())
        for k, v in kwargs.items():
            if hasattr(gen_config, k):
                setattr(gen_config, k, v)

        input_ids, attention_mask = self._expand_inputs_for_generation(
            gen_config.num_return_sequences, inputs, kwargs.get("attention_mask")
        )

        if gen_config.mask_token_id is None:
            raise ValueError("mask_token_id must be set in generation config.")

        return self._sample_multi_block(input_ids, gen_config)

    def _init_block_states(self, prompt_length: int, max_new_tokens: int, block_size: int, use_kv_cache: bool = False):
        state0 = self._make_block_state(0, prompt_length, is_complete=True)
        if use_kv_cache:
            state0["completed_at_nfe"] = 0
            state0["is_cached"] = False
        states = {0: state0}
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        next_id = 1
        if next_id <= num_blocks:
            b_start = prompt_length + (next_id - 1) * block_size
            b_end = min(b_start + block_size, prompt_length + max_new_tokens)
            state = self._make_block_state(b_start, b_end, is_complete=True)
            if use_kv_cache:
                state["completed_at_nfe"] = None
                state["is_cached"] = False
            states[next_id] = state
            next_id += 1
        return states, next_id, num_blocks

    def _update_block_activation(self, block_states, decoded_token_threshold):
        for bid in sorted(block_states.keys()):
            if bid > 0 and not block_states[bid]["is_complete"]:
                prev = block_states[bid - 1]
                progress = 1 - prev["mask_count"] / max(prev["total_masks"], 1)
                if progress >= decoded_token_threshold:
                    block_states[bid]["is_complete"] = True

    def _add_new_block(
        self, block_states, next_block_id, num_blocks, prompt_length, max_new_tokens,
        block_size, block_add_threshold, has_eos, x, mask_token_id, use_kv_cache=False,
    ):
        if next_block_id > num_blocks or has_eos:
            return next_block_id
        last_bid = max(block_states.keys())
        if last_bid == 0:
            return next_block_id
        last = block_states[last_bid]
        progress = 1 - last["mask_count"] / max(last["total_masks"], 1)
        should_add = progress >= block_add_threshold or last["mask_count"] == 0
        if not should_add:
            return next_block_id
        b_start = prompt_length + (next_block_id - 1) * block_size
        b_end = min(b_start + block_size, prompt_length + max_new_tokens)
        if b_end <= b_start:
            return next_block_id
        prev = block_states[next_block_id - 1]
        prev_progress = 1 - prev["mask_count"] / max(prev["total_masks"], 1)
        state = self._make_block_state(b_start, b_end, prev_progress >= 0.5)
        state["mask_count"] = (x[:, b_start:b_end] == mask_token_id).sum().item()
        if use_kv_cache:
            state["completed_at_nfe"] = None
            state["is_cached"] = False
        block_states[next_block_id] = state
        return next_block_id + 1

    def _find_rightmost_active(self, block_states):
        rightmost = 0
        for bid, s in block_states.items():
            if s["is_complete"] or s["mask_count"] > 0:
                rightmost = bid
        return rightmost

    def _decode_entropy_threshold(
        self, logits, x, mask_index, active_end, block_states, entropy_threshold, temperature
    ):
        mask_idx = mask_index.clone()
        mask_idx[:, active_end:] = 0

        p = F.softmax(logits.to(torch.float64), dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)

        x0 = torch.argmax(logits, dim=-1)
        if temperature > 0:
            p_temp = F.softmax(logits / temperature, dim=-1)
            x0 = torch.multinomial(p_temp.view(-1, p_temp.shape[-1]), 1).view(x.shape)

        transfer = (entropy < entropy_threshold) & mask_idx

        first_activated = None
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                first_activated = bid
                break

        if first_activated is not None:
            s, e = block_states[first_activated]["start"], block_states[first_activated]["end"]
            block_t = transfer[:, s:e]
            if not block_t.any():
                b_mask = mask_idx[:, s:e]
                b_ent = entropy[:, s:e]
                b_ent = torch.where(b_mask, b_ent, torch.inf)
                best = b_ent[0].argmin()
                transfer[0, s + best] = True

        x[transfer] = x0[transfer]

        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["mask_count"] > 0:
                s, e = block_states[bid]["start"], block_states[bid]["end"]
                decoded = transfer[:, s:e].sum().item()
                if decoded > 0:
                    block_states[bid]["mask_count"] -= decoded

    def _sample_multi_block(self, input_ids, gen_config):
        prompt_length = input_ids.shape[1]
        max_new_tokens = gen_config.max_length - prompt_length
        x = F.pad(input_ids, (0, max_new_tokens), value=gen_config.mask_token_id)

        block_states, next_block_id, num_blocks = self._init_block_states(
            prompt_length, max_new_tokens, gen_config.block_size
        )

        attn_mask = create_block_causal_mask(
            prompt_length, gen_config.max_length, gen_config.block_size, device=x.device
        )

        eos_id = gen_config.eos_token_id if gen_config.early_stop else None
        nfe = 0

        while True:
            mask_idx = x == gen_config.mask_token_id
            total_masks = mask_idx[:, prompt_length:].sum()
            if total_masks == 0 and next_block_id > num_blocks:
                break

            nfe += 1

            if gen_config.early_stop and eos_id is not None:
                has_eos, _ = handle_early_stop(
                    x, block_states, eos_id, prompt_length, gen_config.mask_token_id
                )
                if has_eos:
                    mask_idx = x == gen_config.mask_token_id
                    total_masks = mask_idx[:, prompt_length:].sum()
                    if total_masks == 0:
                        break
                    while next_block_id <= num_blocks:
                        b_start = prompt_length + (next_block_id - 1) * gen_config.block_size
                        b_end = min(b_start + gen_config.block_size, prompt_length + max_new_tokens)
                        if b_start > has_eos:
                            block_states[next_block_id] = {
                                "start": b_start, "end": b_end,
                                "mask_count": 0, "total_masks": b_end - b_start,
                                "is_complete": True,
                            }
                            next_block_id += 1
                        else:
                            break

            self._update_block_activation(block_states, gen_config.decoded_token_threshold)

            next_block_id = self._add_new_block(
                block_states, next_block_id, num_blocks, prompt_length, max_new_tokens,
                gen_config.block_size, gen_config.block_add_threshold,
                False, x, gen_config.mask_token_id,
            )

            rightmost = self._find_rightmost_active(block_states)
            if rightmost == 0:
                break

            active_end = block_states[rightmost]["end"]
            outputs = self(x, attention_mask=attn_mask, is_causal=False)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            self._decode_entropy_threshold(
                logits, x, mask_idx, active_end, block_states,
                gen_config.entropy_threshold, gen_config.temperature,
            )

            if nfe > 10000:
                break

        return x


