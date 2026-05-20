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


"""Argument utils"""

import argparse
import json
import math
import os
import sys
import types
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from inspect import isclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints

import yaml

from . import logging


T = TypeVar("T")

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the model config. Defaults to `model_path`."},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the pre-trained model. If unspecified, use random init."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    encoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal encoder config and weights."},
    )
    decoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal decoder config and weights."},
    )
    input_encoder: Literal["encoder", "decoder"] = field(
        default="encoder",
        metadata={"help": "Use encoder to encode input images or use decoder.encoder to encode input images."},
    )
    output_encoder: Literal["encoder", "decoder"] = field(
        default="decoder",
        metadata={"help": "Use encoder to encode output images or use decoder.encoder to encode output images."},
    )
    encode_target: bool = field(
        default=False,
        metadata={"help": "Whether to encode target with decoder. Only supports stable diffusion as decoder."},
    )
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "tropical"]] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use. 'tropical' uses min-plus attention via LogSumExp identity."},
    )
    tau: float = field(
        default=0.1,
        metadata={"help": "Temperature for tropical attention (lower = sharper min-plus, 0.01 ≈ hard-min)."},
    )
    moe_implementation: Optional[Literal[None, "eager", "fused"]] = field(
        default=None,
        metadata={"help": "MoE implementation to use."},
    )
    basic_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Basic modules beyond model._no_split_modules to be sharded in FSDP."},
    )
    enable_nvfp4_qat: bool = field(
        default=False,
        metadata={"help": "Enable NVFP4 quantization-aware training (Blackwell 4-bit). Replaces nn.Linear with NVFP4FakeQuantizedLinear."},
    )
    ldlm: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "LDLM configuration (autoencoder, diffusion head, sampler)."},
    )

    def __post_init__(self):
        if self.config_path is None and self.model_path is None:
            raise ValueError("`config_path` must be specified when `model_path` is None.")

        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

        for encoder_type, encoder_args in self.encoders.items():
            if encoder_type not in ["image"]:
                raise ValueError(f"Unsupported encoder type: {encoder_type}. Should be one of {{image}}.")

            if encoder_args.get("config_path") is None and encoder_args.get("model_path") is None:
                raise ValueError("`config_path` and `model_path` cannot be both empty.")

            if encoder_args.get("config_path") is None:
                encoder_args["config_path"] = encoder_args["model_path"]

        for decoder_type, decoder_args in self.decoders.items():
            if decoder_type not in ["image"]:
                raise ValueError(f"Unsupported decoder type: {decoder_type}. Should be one of {{image}}.")

            if decoder_args.get("config_path") is None and decoder_args.get("model_path") is None:
                raise ValueError("`config_path` and `model_path` cannot be both empty.")

            if decoder_args.get("config_path") is None:
                decoder_args["config_path"] = decoder_args["model_path"]


@dataclass
class DataArguments:
    train_path: str = field(
        metadata={
            "help": "Local path/HDFS path/Magnus name of the training data. Use comma to separate multiple datasets."
        },
    )
    train_size: int = field(
        default=10_000_000,
        metadata={"help": "Number of tokens for training to compute training steps for dynamic batch dataloader."},
    )
    data_type: Literal["plaintext", "conversation"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    dataloader_type: Literal["native"] = field(
        default="native",
        metadata={"help": "Type of the dataloader."},
    )
    datasets_type: Literal["mapping", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    data_name: str = field(
        default=None,
        metadata={"help": "Dataset name for multimodal training."},
    )
    data_tag: Literal["default", "mmtag"] = field(
        default="default",
        metadata={"help": "Dataset tag for multimodal training."},
    )
    text_keys: str = field(
        default=None,
        metadata={"help": "Key to get text from the training data."},
    )
    image_keys: str = field(
        default="images",
        metadata={"help": "Key to get images from the training data."},
    )
    chat_template: str = field(
        default="default",
        metadata={"help": "Chat template to use."},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length in training."},
    )
    num_workers: int = field(
        default=2,
        metadata={"help": "Number of workers to load data."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    drop_last: bool = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch."},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for dataloader."},
    )

    def __post_init__(self):
        if self.text_keys is None:
            if self.data_type == "plaintext":
                self.text_keys = "content_split"
            elif self.data_type == "conversation":
                self.text_keys = "messages"
            else:
                raise ValueError(f"Unknown data type: {self.data_type}")


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "Path to save model checkpoints."},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "Maximum learning rate or defult learning rate, or init learning rate for warmup."},
    )
    lr_min: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate."},
    )
    lr_start: float = field(
        default=0.0,
        metadata={"help": "Learning rate for warmup start. Default to 0.0."},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "L2 regularization strength."},
    )
    optimizer: Literal["adamw", "anyprecision_adamw", "apollo", "galore", "scale", "persistent_sparse_adam"] = field(
        default="adamw",
        metadata={"help": "Optimizer. Default to adamw."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Clip value for gradient norm."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    global_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Global batch size. If None, use `micro_batch_size` * `data_parallel_size`."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Epochs to train."},
    )
    rmpad: bool = field(
        default=True,
        metadata={"help": "Enable padding-free training by using the cu_seqlens."},
    )
    rmpad_with_pos_ids: bool = field(
        default=False,
        metadata={"help": "Enable padding-free training by using the position_ids."},
    )
    enable_masking: bool = field(
        default=False,
        metadata={"help": "Enable masking during training."},
    )
    repr_align_wt: float = field(
        default=0.0,
        metadata={"help": "Weight for representation alignment loss (0 = disabled). Used for MDM training with teacher model."},
    )
    anchor_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory of precomputed teacher hidden states (produced by "
                "scripts/precompute_anchor.py). When set, the trainer skips the "
                "live-teacher deepcopy and loads anchors from disk per batch. "
                "Repr-Align is realignment, not distillation: the teacher's output "
                "for a given input is deterministic, so caching is strictly equivalent "
                "to running the live teacher every step."
            )
        },
    )
    align_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated layer indices to use in the repr_align cosine loss, "
                "e.g. '6,12,18,24'. Indices are 0=embedding output, 1..N=transformer "
                "blocks."
            )
        },
    )
    repr_align_sub_sample_ratio: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fraction of valid tokens to sub-sample for the repr_align cosine loss "
                "(1.0 = all tokens). Setting < 1.0 reduces gradient memory for the "
                "alignment branch proportionally. E.g. 0.25 = 4× memory cut."
            )
        },
    )
    repr_align_num_sample_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of layers to randomly sample per step for the repr_align cosine "
                "loss (None = all configured layers). E.g. with align_layers='7,14,21,28' "
                "(4 layers), setting 2 halves layer-axis gradient memory. Combine with "
                "repr_align_sub_sample_ratio for multiplicative savings."
            )
        },
    )
    # ------------------------------------------------------------------
    # Cola DLM (Continuous Latent Diffusion LM, arXiv:2605.06548)
    # auxiliary head on top of Repr-Align. Off when cola_wt == 0.
    # See veomni/models/cola_ldm/.
    # ------------------------------------------------------------------
    cola_wt: float = field(
        default=0.0,
        metadata={"help": "Weight for Cola DLM auxiliary loss (Text VAE + block-causal DiT). 0 = disabled."},
    )
    cola_num_global: int = field(
        default=16,
        metadata={"help": "Number of global semantic latents in the Cola Text VAE encoder."},
    )
    cola_num_local: int = field(
        default=64,
        metadata={"help": "Number of local detail latents in the Cola Text VAE encoder."},
    )
    cola_block_size: int = field(
        default=16,
        metadata={"help": "Local block size for the Cola block-causal DiT attention mask."},
    )
    cola_encoder_depth: int = field(
        default=2,
        metadata={"help": "Depth of each Perceiver in the Cola Text VAE encoder."},
    )
    cola_diffusion_depth: int = field(
        default=4,
        metadata={"help": "Depth of the Cola block-causal DiT."},
    )
    cola_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the Cola head (must divide hidden dim)."},
    )
    cola_source_layer: int = field(
        default=-3,
        metadata={"help": "Which LM hidden-state layer to feed into the Cola head (-1 = last)."},
    )
    cola_detach_student: bool = field(
        default=True,
        metadata={"help": "Detach LM hidden states before the Cola head so its gradient doesn't flow into the student."},
    )
    cola_log_hist_every: int = field(
        default=200,
        metadata={"help": "How often (in global steps) to log Cola latent histograms to wandb. Set 0 to disable."},
    )
    cola_prediction: str = field(
        default="v",
        metadata={"help": "Cola DiT prediction target: 'v' (Flow Matching velocity, paper default) or 'x0' (x0-prediction MSE)."},
    )
    cola_variant: str = field(
        default="block_causal",
        metadata={"help": "ColaDLM causal diffusion variant: 'block_causal' (Guide Labs), 'card' (soft-tail), 'fast_block' (Fast-dLLM v2)."},
    )
    cola_lambda_tail: float = field(
        default=0.6,
        metadata={"help": "CARD soft-tail aggressiveness (0.0–1.0). Only used when cola_variant='card'."},
    )
    # ------------------------------------------------------------------
    # MTP (Multi-Token Prediction, Qwen3.6-style NextN)
    # auxiliary head. Off when mtp_num_layers == 0.
    # ------------------------------------------------------------------
    mtp_num_layers: int = field(
        default=0,
        metadata={"help": "Number of MTP prediction layers (0 = disabled)."},
    )
    mtp_loss_weight: float = field(
        default=0.1,
        metadata={"help": "MTP auxiliary loss scaling factor."},
    )
    mtp_n_predict: int = field(
        default=1,
        metadata={"help": "Number of future tokens MTP head predicts."},
    )
    dyn_bsz: bool = field(
        default=True,
        metadata={"help": "Enable dynamic batch size for padding-free training."},
    )
    dyn_bsz_margin: int = field(
        default=0,
        metadata={"help": "Number of pad tokens in dynamic batch."},
    )
    dyn_bsz_buffer_size: int = field(
        default=200,
        metadata={"help": "Buffer size for dynamic batch size."},
    )
    bsz_warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of batch size warmup steps."},
    )
    bsz_warmup_init_mbtoken: int = field(
        default=200,
        metadata={"help": "Initial number of tokens in a batch in warmup phase."},
    )
    lr_warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of learning rate warmup steps."},
    )
    lr_decay_style: str = field(
        default="constant",
        metadata={"help": "Name of the learning rate scheduler."},
    )
    lr_decay_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of learning rate decay steps."},
    )
    use_doptim: bool = field(
        default=False,
        metadata={"help": "Use veScale's ZeRO optimizer."},
    )
    enable_mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training."},
    )
    enable_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing."},
    )
    enable_reentrant: bool = field(
        default=False,
        metadata={"help": "Use reentrant gradient checkpointing."},
    )
    enable_full_shard: bool = field(
        default=True,
        metadata={"help": "Enable fully shard for FSDP training (ZeRO-3)."},
    )
    enable_forward_prefetch: bool = field(
        default=True,
        metadata={"help": "Enable forward prefetch for FSDP1."},
    )
    enable_fsdp_offload: bool = field(
        default=False,
        metadata={"help": "Enable CPU offload for FSDP1."},
    )
    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Enable activation offload to CPU."},
    )
    activation_gpu_limit: float = field(
        default=0.0,
        metadata={
            "help": "When enabling activation offload, `activation_gpu_limit` GB activations are allowed to reserve on GPU."
        },
    )
    enable_manual_eager: bool = field(
        default=False,
        metadata={"help": "Enable veScale's manual eager."},
    )
    init_device: Literal["cpu", "cuda", "meta"] = field(
        default="cuda",
        metadata={
            "help": "Device to initialize model weights. 1. `cpu`: Init parameters on CPU in rank0 only. 2. `cuda`: Init parameters on GPU. 3. `meta`: Init parameters on meta."
        },
    )
    enable_full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full determinism."},
    )
    empty_cache_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between two empty cache operations."},
    )
    data_parallel_mode: Literal["ddp", "deepspeed", "fsdp1", "fsdp2", "fsdp2-vescale"] = field(
        default="ddp",
        metadata={"help": "Data parallel mode."},
    )
    ds_zero_stage: Literal[1, 2, 3] = field(
        default=3,
        metadata={"help": "DeepSpeed ZeRO optimization stage (1, 2, or 3). Only used when data_parallel_mode=deepspeed."},
    )
    ds_offload_optimizer: Optional[Literal["cpu", "nvme"]] = field(
        default=None,
        metadata={"help": "Offload optimizer state to 'cpu' or 'nvme'. None = no offload."},
    )
    ds_offload_param: Optional[Literal["cpu", "nvme"]] = field(
        default=None,
        metadata={"help": "Offload parameters to 'cpu' or 'nvme' (ZeRO-3 only). None = no offload."},
    )
    ds_nvme_path: str = field(
        default="",
        metadata={"help": "Directory path for NVMe offload. Required when offload to 'nvme'."},
    )
    ds_overlap_comm: bool = field(
        default=True,
        metadata={"help": "Overlap communication with computation in DeepSpeed ZeRO."},
    )
    ds_contiguous_gradients: bool = field(
        default=True,
        metadata={"help": "Use contiguous gradient buffers in DeepSpeed ZeRO."},
    )
    ds_config_path: str = field(
        default="",
        metadata={"help": "Path to a custom DeepSpeed JSON config. Overrides all ds_* fields."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size."},
    )
    ulysses_parallel_size: int = field(
        default=1,
        metadata={"help": "Ulysses sequence parallel size."},
    )
    context_parallel_size: int = field(
        default=1,
        metadata={"help": "Ring-attn context parallel size."},
    )
    ckpt_manager: Literal["bytecheckpoint", "dcp"] = field(
        default="bytecheckpoint",
        metadata={"help": "Checkpoint manager."},
    )
    load_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to bytecheckpoint checkpoint to resume from."},
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two checkpoint saves."},
    )
    save_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs between two checkpoint saves."},
    )
    save_time_interval_minutes: int = field(
        default=0,
        metadata={
            "help": (
                "Interval in minutes to save a rolling checkpoint used for auto-resume. "
                "Set to 0 to disable time-based checkpointing."
            )
        },
    )
    auto_resume: bool = field(
        default=True,
        metadata={
            "help": (
                "Automatically resume from the most recent checkpoint. When enabled,"
                " time-based checkpoints take precedence over step-based ones."
            )
        },
    )
    eval_every: int = field(
        default=0,
        metadata={"help": "Run HumanEval evaluation every N steps. Set to 0 to disable."},
    )
    save_hf_weights: bool = field(
        default=True,
        metadata={"help": "Save the huggingface format weights to the last checkpoint dir."},
    )
    save_total_limit: int = field(
        default=0,
        metadata={"help": "Maximum number of step-based checkpoints to keep. 0 = keep all. Oldest are deleted first."},
    )
    save_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Save DeepSpeed ZeRO optimizer state with each checkpoint. Set False to save only HF weights (~54GB for 27B) and skip the full ZeRO state (~211GB). Cannot resume training when False."},
    )
    freeze_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated layer patterns to freeze (e.g., 'attn,mlp'). Uses case-insensitive substring matching."},
    )
    quantize_frozen: bool = field(
        default=False,
        metadata={"help": "If true, apply torchao weight-only quantization to all Linear modules whose params are entirely frozen. Cuts resident weight VRAM ~2-4x. Requires `pip install torchao`. Applied after freeze_layers, before FSDP wrap."},
    )
    quantize_frozen_dtype: Literal["int8", "int4"] = field(
        default="int8",
        metadata={"help": "Precision for quantize_frozen. int8 ~halves weight memory with no measurable quality loss; int4 ~quarters it but may hurt convergence on small finetunes."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Use wandb to log experiment."},
    )
    wandb_project: str = field(
        default="VeOmni",
        metadata={"help": "Wandb project name."},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb experiment name."},
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "Wandb entity name."},
    )
    enable_profiling: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    profile_start_step: int = field(
        default=1,
        metadata={"help": "Start step for profiling."},
    )
    profile_end_step: int = field(
        default=2,
        metadata={"help": "End step for profiling."},
    )
    profile_trace_dir: str = field(
        default="./trace",
        metadata={"help": "Direction to export the profiling result."},
    )
    profile_record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the shapes of the input tensors."},
    )
    profile_profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to profile the memory usage."},
    )
    profile_with_stack: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the stack traces."},
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Max training steps per epoch. (for debug)"},
    )


    def __post_init__(self):
        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.global_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if self.context_parallel_size > 1 or self.ulysses_parallel_size > 1:
            if self.world_size % (self.context_parallel_size * self.ulysses_parallel_size) != 0:
                raise ValueError("World size should be a multiple of context_parallel_size * ulysses_parallel_size.")

            self.data_parallel_size = self.world_size // (self.context_parallel_size * self.ulysses_parallel_size)
        else:
            if self.world_size % (self.tensor_parallel_size * self.pipeline_parallel_size) != 0:
                raise ValueError("World size should be a multiple of tensor_parallel_size * pipeline_parallel_size.")

            self.data_parallel_size = self.world_size // (self.tensor_parallel_size * self.pipeline_parallel_size)

        if self.rmpad and self.rmpad_with_pos_ids:
            raise ValueError("`rmpad` and `rmpad_with_pos_ids` cannot be both True.")

        # init method check
        assert (
            self.expert_parallel_size == 1 or self.init_device != "cpu"
        ), "cpu init is not supported when enable ep. Please use `init_device = cuda` or `init_device = meta` instead."

        # calculate gradient accumulation steps
        if self.global_batch_size is None:
            self.global_batch_size = self.micro_batch_size * self.data_parallel_size
            self.gradient_accumulation_steps = 1
            logger.info_rank0("`global_batch_size` is None, disable gradient accumulation.")
        elif self.global_batch_size % (self.micro_batch_size * self.data_parallel_size) == 0:
            self.gradient_accumulation_steps = self.global_batch_size // (
                self.micro_batch_size * self.data_parallel_size
            )
            logger.info_rank0(f"Set gradient accumulation to {self.gradient_accumulation_steps}.")
        else:
            raise ValueError(
                f"`global_batch_size` should be a multiple of {self.micro_batch_size * self.data_parallel_size}."
            )

        if self.gradient_accumulation_steps > 1 and self.enable_fsdp_offload:
            raise ValueError("Gradient accumulation is not supported with FSDP offload.")

        # ── DeepSpeed validation ──
        if self.data_parallel_mode == "deepspeed":
            if self.ds_zero_stage not in (1, 2, 3):
                raise ValueError(f"ds_zero_stage must be 1, 2, or 3, got {self.ds_zero_stage}.")
            if self.ds_offload_param is not None and self.ds_zero_stage != 3:
                raise ValueError(
                    f"ds_offload_param={self.ds_offload_param!r} requires zero_stage=3, got {self.ds_zero_stage}."
                )
            if "nvme" in (str(self.ds_offload_optimizer or ""), str(self.ds_offload_param or "")):
                if not self.ds_nvme_path or not os.path.isdir(self.ds_nvme_path):
                    raise ValueError(
                        f"NVMe offload requires a valid ds_nvme_path directory, got: '{self.ds_nvme_path}'."
                    )
            # zero.Init() (and meta tensors) are only needed for ZeRO-3 parameter
            # partitioning. ZeRO-1/2 replicate weights like DDP — init on cuda directly.
            if self.ds_zero_stage == 3 and self.init_device not in ("cpu", "meta"):
                logger.info_rank0(
                    f"Forcing init_device='meta' for DeepSpeed ZeRO-3 (was '{self.init_device}')."
                )
                object.__setattr__(self, "init_device", "meta")

        self.dataloader_batch_size = self.global_batch_size // self.data_parallel_size  # = micro bsz * grad accu

        # merlin save paths
        self.save_checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        self.model_assets_dir = os.path.join(self.output_dir, "model_assets")

    def compute_train_steps(
        self, max_seq_len: Optional[int] = None, train_size: Optional[int] = None, dataset_length: Optional[int] = None
    ) -> None:
        """
        Computes the training steps per epoch according to the data length.
        """
        if self.rmpad or self.rmpad_with_pos_ids:
            assert max_seq_len is not None and train_size is not None, "max_seq_len and train_size are required."
            token_micro_bsz = self.micro_batch_size * max_seq_len
            train_size = int(train_size * (1 + self.bsz_warmup_ratio / 2))
            eff_token_rate = (token_micro_bsz - self.dyn_bsz_margin) / token_micro_bsz
            self._train_steps = math.ceil(train_size / (self.global_batch_size * max_seq_len * eff_token_rate))
        elif dataset_length is not None:
            self._train_steps = math.floor(dataset_length / self.dataloader_batch_size)  # assuming drop_last is true
        elif self.max_steps is not None:
            self._train_steps = self.max_steps
        else:
            raise ValueError("Please provide `dataset_length` or `max_steps`!")

    @property
    def train_steps(self) -> int:
        if self.max_steps is not None and self._train_steps >= self.max_steps:
            logger.warning_once(f"Set train_steps to {self.max_steps}. It should be for debug purpose only.")
            return self.max_steps

        if self._train_steps == -1:
            raise ValueError("Please run `compute_train_steps` first!")

        return self._train_steps


@dataclass
class InferArguments:
    model_path: str = field(
        metadata={"help": "Local path/HDFS path to the pre-trained model."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling in decoding."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature value of decoding."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "The top_p value of decoding."},
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens to generate."},
    )

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


def _string_to_bool(value: Union[bool, str]) -> bool:
    """
    Converts a string input to bool value.

    Taken from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Truthy value expected: got {value} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
    )


def _convert_str_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely checks that a passed value is a dictionary and converts any string values to their appropriate types.

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/training_args.py#L189
    """
    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):  # check for bool
                input_dict[key] = value.lower() == "true"
            elif value.isdigit():  # check for digit
                input_dict[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                input_dict[key] = float(value)

    return input_dict


def _make_choice_type_function(choices: List[Any]) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/hf_argparser.py#L48

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def parse_args(rootclass: T) -> T:
    """
    Parses the root argument class using the CLI inputs or yaml inputs.

    Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/hf_argparser.py#L266
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_to_subclass = {}
    dict_fields = set()
    for subclass in fields(rootclass):
        base = subclass.name
        base_to_subclass[base] = subclass.default_factory
        try:
            type_hints: Dict[str, type] = get_type_hints(subclass.default_factory)
        except Exception:
            raise RuntimeError(f"Type resolution failed for {subclass.default_factory}.")

        for attr in fields(subclass.default_factory):
            if not attr.init:
                continue

            attr_type = type_hints[attr.name]
            origin_type = getattr(attr_type, "__origin__", attr_type)
            if isinstance(attr_type, str):
                raise RuntimeError(f"Cannot resolve type {attr.type} of {attr.name}.")

            if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
                if len(attr_type.__args__) != 2 or type(None) not in attr_type.__args__:  # only allows Optional[X]
                    raise RuntimeError(f"Cannot resolve type {attr.type} of {attr.name}.")

                if bool not in attr_type.__args__:  # except for `Union[bool, NoneType]`
                    attr_type = (
                        attr_type.__args__[0] if isinstance(None, attr_type.__args__[1]) else attr_type.__args__[1]
                    )
                    origin_type = getattr(attr_type, "__origin__", attr_type)

            parser_kwargs = attr.metadata.copy()
            if origin_type is Literal or (isinstance(attr_type, type) and issubclass(attr_type, Enum)):
                if origin_type is Literal:
                    parser_kwargs["choices"] = attr_type.__args__
                else:
                    parser_kwargs["choices"] = [x.value for x in attr_type]

                parser_kwargs["type"] = _make_choice_type_function(parser_kwargs["choices"])

                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                else:
                    parser_kwargs["required"] = True

            elif attr_type is bool or attr_type == Optional[bool]:
                parser_kwargs["type"] = _string_to_bool
                if attr_type is bool or (attr.default is not None and attr.default is not MISSING):
                    parser_kwargs["default"] = False if attr.default is MISSING else attr.default
                    parser_kwargs["nargs"] = "?"
                    parser_kwargs["const"] = True

            elif isclass(origin_type) and issubclass(origin_type, list):
                parser_kwargs["type"] = attr_type.__args__[0]
                parser_kwargs["nargs"] = "+"
                if attr.default_factory is not MISSING:
                    parser_kwargs["default"] = attr.default_factory()
                elif attr.default is MISSING:
                    parser_kwargs["required"] = True

            elif isclass(origin_type) and issubclass(origin_type, dict):
                parser_kwargs["type"] = str  # parse dict inputs with json string
                dict_fields.add(f"{base}.{attr.name}")
                if attr.default_factory is not MISSING:
                    parser_kwargs["default"] = str(attr.default_factory())
                elif attr.default is MISSING:
                    parser_kwargs["required"] = True

            else:
                parser_kwargs["type"] = attr_type
                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                elif attr.default_factory is not MISSING:
                    parser_kwargs["default"] = attr.default_factory()
                else:
                    parser_kwargs["required"] = True

            parser.add_argument(f"--{base}.{attr.name}", **parser_kwargs)

    cmd_args = sys.argv[1:]
    cmd_args_string = "=".join(cmd_args)  # use `=` to mark the end of arg name
    input_data = {}
    if cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    elif cmd_args[0].endswith(".json"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = json.load(f)

    for base, arg_dict in input_data.items():
        for arg_name, arg_value in arg_dict.items():
            if f"--{base}.{arg_name}=" not in cmd_args_string:  # lower priority
                cmd_args.append(f"--{base}.{arg_name}")
                cmd_args.append(arg_value if isinstance(arg_value, str) else json.dumps(arg_value))

    args, remaining_args = parser.parse_known_args(cmd_args)
    if remaining_args:
        raise ValueError(f"Some specified arguments are not used by the ArgumentParser: {remaining_args}")

    parse_result = defaultdict(dict)
    for key, value in vars(args).items():
        if key in dict_fields:
            if isinstance(value, str) and value.startswith("{"):
                value = _convert_str_dict(json.loads(value))
            else:
                raise ValueError(f"Expect a json string for dict argument, but got {value}")

        base, name = key.split(".", maxsplit=1)
        parse_result[base][name] = value

    data_classes = {}
    for base, subclass_type in base_to_subclass.items():
        data_classes[base] = subclass_type(**parse_result.get(base, {}))

    return rootclass(**data_classes)


def save_args(args: T, output_path: str) -> None:
    """
    Saves arguments to a json file.

    Args:
        args (dataclass): Arguments.
        output_path (str): Output path.
    """

    local_dir = output_path

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "veomni_cli.yaml")
    with open(local_path, "w") as f:
        f.write(yaml.safe_dump(asdict(args), default_flow_style=False))
