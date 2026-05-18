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

"""DeepSpeed engine initialization and config builder for Open-dLLM."""

import json
from typing import TYPE_CHECKING, Tuple

import torch

from ..utils import logging


if TYPE_CHECKING:
    from ..utils.arguments import TrainingArguments

logger = logging.get_logger(__name__)


def build_ds_config(train_args: "TrainingArguments") -> dict:
    """Translate TrainingArguments into a DeepSpeed JSON config.

    If ``train_args.ds_config_path`` is set, load that JSON verbatim
    and only patch ``train_batch_size`` / ``gradient_accumulation_steps``.
    Otherwise, build from the individual ``ds_*`` fields.
    """
    if train_args.ds_config_path:
        with open(train_args.ds_config_path) as f:
            config = json.load(f)
        config.setdefault(
            "train_batch_size",
            train_args.world_size
            * train_args.micro_batch_size
            * train_args.gradient_accumulation_steps,
        )
        config.setdefault("gradient_accumulation_steps", train_args.gradient_accumulation_steps)
        return config

    config = {
        "train_batch_size": train_args.world_size
        * train_args.micro_batch_size
        * train_args.gradient_accumulation_steps,
        "micro_batch_size_per_gpu": train_args.micro_batch_size,
        "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
        "gradient_clipping": train_args.max_grad_norm,
        "zero_optimization": {
            "stage": train_args.ds_zero_stage,
            "overlap_comm": train_args.ds_overlap_comm,
            "contiguous_gradients": train_args.ds_contiguous_gradients,
        },
        "bf16": {"enabled": train_args.enable_mixed_precision},
        "steps_per_print": 1,
        # Allow non-DS optimizers (e.g. AnyPrecisionAdamW) with ZeRO-Offload.
        # DeepSpeedCPUAdam is faster, but AnyPrecisionAdamW keeps optimizer states
        # in bf16 (half the RAM of fp32 Adam states).
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
    }

    zero = config["zero_optimization"]
    if train_args.ds_offload_optimizer:
        offload = {"device": train_args.ds_offload_optimizer}
        if train_args.ds_offload_optimizer == "nvme":
            offload["nvme_path"] = train_args.ds_nvme_path
        zero["offload_optimizer"] = offload

    if train_args.ds_offload_param:
        offload = {"device": train_args.ds_offload_param}
        if train_args.ds_offload_param == "nvme":
            offload["nvme_path"] = train_args.ds_nvme_path
        zero["offload_param"] = offload

    return config


def init_deepspeed_engine(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_args: "TrainingArguments",
    ds_config: dict,
) -> Tuple:
    """Initialize DeepSpeed engine.

    Returns ``(engine, ds_optimizer, ds_lr_scheduler)``.
    The engine wraps the model; access original via ``engine.module``.
    """
    import deepspeed

    engine, ds_optimizer, _, ds_lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=ds_config,
    )

    logger.info_rank0(
        f"DeepSpeed engine initialized. ZeRO stage={train_args.ds_zero_stage}, "
        f"offload_optimizer={train_args.ds_offload_optimizer}, "
        f"offload_param={train_args.ds_offload_param}"
    )

    return engine, ds_optimizer, ds_lr_scheduler


def load_hf_weights_zero3(model: "torch.nn.Module", weights_path: str) -> None:
    """Load HuggingFace safetensors into a ZeRO-3 partitioned model.

    Iterates shard files one at a time; only rank 0 reads each file.
    GatheredParameters gathers each param on rank 0, copies the weight, then
    re-partitions across all ranks. Peak RAM ≈ size of one shard file (~5-10 GB).
    """
    import json
    import os

    import deepspeed
    import torch.distributed as dist
    from safetensors.torch import load_file

    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    single_file = os.path.join(weights_path, "model.safetensors")

    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
    elif os.path.exists(single_file):
        weight_map = None
    else:
        logger.warning_rank0(f"No safetensors weights at {weights_path}; skipping weight load.")
        return

    param_dict = dict(model.named_parameters())

    if weight_map is None:
        shard_weights = load_file(single_file) if dist.get_rank() == 0 else {}
        for name, param in param_dict.items():
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                if dist.get_rank() == 0 and name in shard_weights:
                    param.data.copy_(shard_weights[name].to(dtype=param.dtype))
        logger.info_rank0(f"Loaded HF weights from {weights_path} into ZeRO-3 model.")
        return

    # Group params by shard file to load each file exactly once
    file_to_params: dict = {}
    for pname, fname in weight_map.items():
        file_to_params.setdefault(fname, []).append(pname)

    n_files = len(file_to_params)
    for i, (fname, pnames) in enumerate(file_to_params.items()):
        shard_weights = {}
        if dist.get_rank() == 0:
            shard_weights = load_file(os.path.join(weights_path, fname))
        for pname in pnames:
            if pname not in param_dict:
                continue
            param = param_dict[pname]
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                if dist.get_rank() == 0 and pname in shard_weights:
                    param.data.copy_(shard_weights[pname].to(dtype=param.dtype))
        del shard_weights
        if i % 5 == 0 or i == n_files - 1:
            logger.info_rank0(f"  ZeRO-3 weight load: shard {i + 1}/{n_files} ({fname})")

    logger.info_rank0(f"Loaded HF weights from {weights_path} into ZeRO-3 model.")


def patch_deepspeed_zero_init_for_meta_tensors() -> None:
    """Patch DeepSpeed Init._post_init_method to tolerate accelerate meta tensors.

    accelerate's init_empty_weights() creates params on meta device.  DeepSpeed's
    zero.Init()._post_init_method fires for each module and runs
    ``param.data = param.data.to(local_device)`` — which raises NotImplementedError
    for meta tensors.  This patch materialises them as empty CPU tensors on
    ``self.remote_device`` BEFORE the original method moves them to the accelerator
    for partitioning.  Idempotent: safe to call multiple times.
    """
    import torch
    from deepspeed.runtime.zero.partition_parameters import Init as _DSInit

    if getattr(_DSInit, "_meta_tensor_patch_applied", False):
        return

    _orig_post_init = _DSInit._post_init_method

    def _patched_post_init(self, module):
        # param.data = tensor raises "incompatible tensor type" when changing from
        # meta device to CPU.  Swap the Parameter object itself instead.
        for pname, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                module._parameters[pname] = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device=self.remote_device),
                    requires_grad=param.requires_grad,
                )
        _orig_post_init(self, module)

    _DSInit._post_init_method = _patched_post_init
    _DSInit._meta_tensor_patch_applied = True
    logger.info_rank0("Patched DeepSpeed Init._post_init_method to handle meta tensors.")
