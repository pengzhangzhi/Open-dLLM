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


from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .loader import BaseModelLoader, get_loader


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)


def build_foundation_model(
    config_path: str,
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "tropical"]] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    make_teacher: bool = False,
    anchor_cache_dir: Optional[str] = None,
    align_layers: Optional[str] = None,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}

    if moe_implementation is not None:
        config_kwargs["_moe_implementation"] = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")
        logger.info_rank0(f"config_kwargs: {config_kwargs}")
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")

    # "tropical" is not a registered HF attn_implementation; load with "eager" then post-patch.
    use_tropical = attn_implementation == "tropical"
    load_attn_impl = "eager" if use_tropical else attn_implementation

    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True, **config_kwargs)

    loader: Optional[BaseModelLoader] = get_loader(config)

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": load_attn_impl,
        "trust_remote_code": True,
    }

    _is_deepspeed = get_parallel_state().dp_mode == "deepspeed"
    if init_device == "meta" or (init_device == "cpu" and (_is_deepspeed or get_parallel_state().global_rank != 0)):
        # DeepSpeed: model is created inside zero.Init() context with empty CPU tensors;
        # zero.Init() partitions each param on-the-fly. Weights are loaded after
        # deepspeed.initialize() via load_hf_weights_zero3().
        empty_init = True
    else:
        empty_init = False
    if _is_deepspeed and weights_path is not None:
        logger.info_rank0("DeepSpeed mode: model created inside zero.Init() context; weights loaded post-init.")

    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
        make_teacher=make_teacher,
        anchor_cache_dir=anchor_cache_dir,
        align_layers=align_layers,
    )

    if use_tropical:
        model.config._attn_implementation = "tropical"
        logger.info_rank0("Patched model with tropical attention (τ={})".format(
            getattr(model.config, "tau", 0.1)
        ))

    return model
