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


from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ..distributed.parallel_state import get_parallel_state
from ..utils.seqlen_pos_transform_utils import len2culen, pos2culen
from .constants import IGNORE_INDEX


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


class CollatePipeline:
    def __init__(self, data_collators: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            data_collators: a list of data collators or a single data collator
        """

        if not isinstance(data_collators, (list, tuple)):
            data_collators = [data_collators]
        self.data_collators = data_collators

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch


@dataclass
class DataCollatorWithPadding(DataCollator):
    """
    Data collator with padding.
    """

    pad_token_id: int = 0

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids", "images_seq_mask"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch


@dataclass
class DataCollatorWithPacking(DataCollator):
    """
    Data collator with packing.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        seqlens = torch.tensor([len(feature["input_ids"]) for feature in features], dtype=torch.long)
        batch = {"cu_seqlens": len2culen(seqlens)}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features])
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        return batch


@dataclass
class DataCollatorWithPositionIDs(DataCollator):
    """
    Data collator with packing by position ids.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        if "labels" in batch:
            cu_seqlens = pos2culen(batch["position_ids"])
            batch["labels"][:, cu_seqlens[1:-1]] = IGNORE_INDEX

        return batch


@dataclass
class DataCollatorWithPositionIDsMasking(DataCollator):
    """
    Enhanced data collator with masking for MDM training with teacher model support.
    
    Generates:
    - input_ids: masked sequence for student model
    - casual_input_ids: original unmasked sequence for teacher model  
    - left_mask: positions to the left of masked tokens for representation alignment
    - mask_ratio: masking ratio for loss weighting
    """
    def __init__(self, mask_token_id: int):
        self.mask_token_id = mask_token_id

    def _random_masking(self, input_ids: "torch.Tensor") -> tuple:
        """
        Randomly mask input_ids and generate alignment masks.
        
        Returns:
            masked_input_ids: input_ids with masked tokens
            mask_ratio: ratio of masked tokens
            left_mask: mask indicating positions to the left of masked tokens
        """
        mask_ratio = torch.rand(1, device=input_ids.device).clamp(1/500, 1-1/500)
        mask_indices = torch.rand_like(input_ids.float()) < mask_ratio
        
        # Create left_mask: positions to the left of masked tokens
        left_mask = torch.zeros_like(input_ids, dtype=torch.float)
        for i in range(1, input_ids.size(-1)):
            left_mask[..., i-1] = mask_indices[..., i].float()
        
        # Apply masking
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_indices] = self.mask_token_id
        
        return masked_input_ids, mask_ratio.repeat(input_ids.size(0)), left_mask

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        
        # Process each input type
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                if input_name == "input_ids":
                    # Store original input_ids as casual_input_ids for teacher model
                    casual_input_ids = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
                    batch["casual_input_ids"] = casual_input_ids
                    
                    # Apply masking and generate alignment masks
                    masking_results = [self._random_masking(feature[input_name]) for feature in features]
                    batch[input_name] = torch.cat([result[0] for result in masking_results], dim=-1).unsqueeze(0)
                    batch["mask_ratio"] = torch.cat([result[1] for result in masking_results], dim=-1).unsqueeze(0)
                    batch["left_mask"] = torch.cat([result[2] for result in masking_results], dim=-1).unsqueeze(0)
                else:
                    batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        # Generate position_ids if not present
        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        # Set labels for loss computation (only masked positions matter)
        if "labels" in batch:
            batch["casual_labels"] = batch["labels"].clone()
            cu_seqlens = pos2culen(batch["position_ids"])
            batch["casual_labels"][:, cu_seqlens[1:-1]] = IGNORE_INDEX
            batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX
            
        return batch


class DataCollatorWithTrajectoryMasking(DataCollator):
    """Replace random masking with d3LLM-style trajectory-guided masking.

    For each sample, the unmasking pattern from a pre-computed teacher
    trajectory determines *which* tokens are masked at training time.
    Tokens unmasked early in the trajectory are predicted first; tokens
    unmasked late are predicted later.  This aligns training-time masking
    with inference-time decoding order.

    Adapted from d3LLM's ``forward_process_with_trajectory()``.

    Args:
        mask_token_id: ID of the [MASK] token.
        trajectory_dataset: ``TrajectoryDataset`` instance for lookup.
        current_mask_ratio: Target mask ratio for this training step
            (scheduled by curriculum, typically 0.0 → 0.8 over training).
        max_mask_ratio: Upper bound for jitter.
        current_block_size: Block size for block-wise loss.
        use_blockwise_loss: If True, only predict a random block per sample;
            otherwise mask the entire response region.
        use_complementary_loss: If True, also produce the inverse mask
            pattern for dParallel-style dual loss.
    """
    def __init__(
        self,
        mask_token_id: int,
        trajectory_dataset=None,
        current_mask_ratio: float = 0.5,
        max_mask_ratio: float = 0.8,
        current_block_size: int = 32,
        use_blockwise_loss: bool = False,
    ):
        self.mask_token_id = mask_token_id
        self.trajectory_dataset = trajectory_dataset
        self.current_mask_ratio = current_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.current_block_size = current_block_size
        self.use_blockwise_loss = use_blockwise_loss

    def _select_trajectory_step(self, sample_idx, mask_ratio, block_start, block_end):
        if self.trajectory_dataset is None:
            return None
        return self.trajectory_dataset.select_step(
            sample_idx, mask_ratio, self.mask_token_id, block_start, block_end
        )

    def _mask_with_trajectory(self, input_ids, sample_idx):
        b, l = input_ids.shape
        device = input_ids.device
        cur_mask_ratio = self.current_mask_ratio
        cur_mask_ratio = cur_mask_ratio + torch.rand(1, device=device).item() * (
            self.max_mask_ratio - cur_mask_ratio
        )

        masked = input_ids.clone()
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        for i in range(b):
            if self.use_blockwise_loss:
                max_blocks = l // self.current_block_size
                num_blocks = torch.randint(0, max_blocks + 1, (1,)).item()
                mask_start = num_blocks * self.current_block_size
                mask_end = min(
                    mask_start + self.current_block_size,
                    l,
                )
            else:
                mask_start = 0
                mask_end = l

            traj_step = self._select_trajectory_step(
                sample_idx[i].item() if sample_idx is not None else None,
                cur_mask_ratio,
                mask_start,
                mask_end,
            )

            seg_len = mask_end - mask_start
            if traj_step is not None:
                traj_tensor = torch.tensor(
                    traj_step, device=device, dtype=torch.long
                )
                seg_mask = traj_tensor[mask_start:mask_end] == self.mask_token_id
            else:
                p_mask = 0.999 * cur_mask_ratio + 0.001
                seg_mask = torch.rand(seg_len, device=device) < p_mask

            masked_indices[i, mask_start:mask_end] = seg_mask
            masked[i, mask_start:mask_end] = torch.where(
                seg_mask, self.mask_token_id, input_ids[i, mask_start:mask_end]
            )
            masked[i, mask_end:l] = self.mask_token_id

        return masked, masked_indices

    def __call__(self, features):
        batch = {}
        sample_indices = None

        for input_name in features[0].keys():
            if input_name == "input_ids":
                input_ids = torch.cat(
                    [f[input_name] for f in features], dim=-1
                ).unsqueeze(0)
                batch["casual_input_ids"] = input_ids.clone()

                masked_ids, mask_idx = self._mask_with_trajectory(
                    input_ids, sample_indices
                )
                batch[input_name] = masked_ids
                batch["mask_ratio"] = torch.full(
                    (input_ids.size(0),),
                    self.current_mask_ratio,
                    device=input_ids.device,
                )
                batch["left_mask"] = torch.zeros_like(
                    input_ids, dtype=torch.float, device=input_ids.device
                )
                for j in range(1, input_ids.size(-1)):
                    batch["left_mask"][..., j - 1] = mask_idx[..., j].float()

            elif input_name in ("attention_mask", "labels", "position_ids"):
                batch[input_name] = torch.cat(
                    [f[input_name] for f in features], dim=-1
                ).unsqueeze(0)
            elif input_name == "sample_idx":
                sample_indices = torch.tensor(
                    [f[input_name] for f in features], dtype=torch.long
                )
                batch[input_name] = sample_indices
            else:
                batch[input_name] = default_collate(
                    [f[input_name] for f in features]
                )

        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(f["input_ids"])) for f in features]
            ).unsqueeze(0)

        if "labels" in batch:
            batch["casual_labels"] = batch["labels"].clone()
            cu_seqlens = pos2culen(batch["position_ids"])
            batch["casual_labels"][:, cu_seqlens[1:-1]] = IGNORE_INDEX
            batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX

        return batch


@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used in dynamic batch dataloader at main process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        return features


@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader at worker process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        return features[0]


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        micro_batch_size = len(features) // self.num_micro_batch
        for i in range(len(features)):
            features[i] = features[i][0]  # 1-to-N inverse transform

        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            micro_batches.append(self.internal_data_collator(features[i : i + micro_batch_size]))

        return micro_batches


@dataclass
class TextSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.
    Args:
        rmpad: whether the samples is packing or not.
        rmpad_with_pos_ids: whether the samples is packing by position ids or not.
        pad_token_id: the id of the padding token.
    """

    rmpad: bool
    rmpad_with_pos_ids: bool
    pad_token_id: int = 0

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """
        Slices a tensor along the specified dimension for sequence parallelism.
        """
        seq_length = tensor.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return tensor.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_length: int = 0
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        input_ids = batch.pop("input_ids")
        labels = batch.pop("labels")[..., 1:].contiguous()  # shift labels
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        if self.rmpad_with_pos_ids:  # mask the last token of each sequence
            cu_seqlens = pos2culen(batch["position_ids"])
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX
        elif self.rmpad:
            labels = labels.view(-1)
            labels[batch["cu_seqlens"][1:-1] - 1] = IGNORE_INDEX
        else:
            if "position_ids" not in batch:  # we should calculate the position ids before chunking
                batch["position_ids"] = torch.arange(0, input_ids.size(-1)).unsqueeze(0)

        # sp padding
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)

        if self.rmpad_with_pos_ids:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
            )
        else:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=0, pad_length=pad_length
            )

        if self.rmpad:
            if pad_length > 0:
                batch["cu_seqlens"] = F.pad(
                    batch["cu_seqlens"], (0, 1), "constant", batch["cu_seqlens"][-1].item() + pad_length
                )
        else:
            batch["position_ids"] = self.sp_padding(batch["position_ids"], dim=-1, pad_value=0, pad_length=pad_length)

        # sp slice
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)

        return batch
