import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.data.constants import IGNORE_INDEX
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.ops.replay_buffer import ReprAlignReplayBuffer
from veomni.ops.trajectory_dataset import TrajectoryDataset
from veomni.data.data_collator import DataCollatorWithTrajectoryMasking
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


def extract_humaneval_scores(output_path: str) -> Dict[str, float]:
    """
    Extract pass@1 and pass@10 scores from HumanEval evaluation results.
    
    Finds the most recent results_*.json file in the output directory, 
    handling timestamped filenames from lm-evaluation-harness.
    
    Args:
        output_path: Path to the evaluation output directory
        
    Returns:
        Dictionary with pass@1 and pass@10 scores, or empty dict if not found
    """
    # Find all results files with timestamps (e.g., results_2025-10-04T07-03-56.294312.json)
    results_pattern = os.path.join(output_path, "*", "results_*.json")
    results_files = glob.glob(results_pattern)
    print(f"results_files: {results_files}")
    if not results_files:
        logger.warning(f"No results files found matching pattern: {results_pattern}")
        return {}

    # Get the latest file based on timestamp in filename
    def extract_timestamp(filepath):
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)', filepath)
        return match.group(1) if match else ""

    results_file = max(results_files, key=extract_timestamp)
    logger.info(f"Using results file: {results_file}")

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        scores = {}

        # HumanEval results are typically under 'results' -> 'humaneval'
        if 'results' in results and 'humaneval' in results['results']:
            humaneval_results = results['results']['humaneval']

            # Extract pass@k metrics
            for key, value in humaneval_results.items():
                if 'pass@' in key.lower():
                    scores[key] = value

            logger.info(f"Extracted HumanEval scores: {scores}")
        else:
            logger.warning(f"HumanEval results not found in expected format. Keys: {results.keys()}")

        return scores

    except Exception as e:
        logger.error(f"Error extracting HumanEval scores: {e}")
        return {}


def freeze_layers_by_patterns(model, patterns_str, logger):
    """Freeze model parameters matching specified patterns."""
    if not patterns_str:
        return

    patterns = [p.strip().lower() for p in patterns_str.split(',')]
    total_params = 0
    frozen_params = 0
    frozen_param_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        name_lower = name.lower()

        if any(pattern in name_lower for pattern in patterns):
            param.requires_grad_(False)
            frozen_params += param.numel()
            frozen_param_names.append(name)

    # Log results
    if frozen_params > 0:
        percentage = (frozen_params / total_params) * 100
        logger.info_rank0(f"Frozen {frozen_params:,} / {total_params:,} parameters ({percentage:.2f}%)")
        logger.info_rank0(f"Frozen parameters matching patterns {patterns}:")
        for name in frozen_param_names:
            logger.info_rank0(f"  - {name}")
    else:
        logger.warning_rank0(f"No parameters matched patterns: {patterns}")


def quantize_frozen_linears(model, dtype: str, logger):
    """Apply torchao weight-only quantization to fully-frozen nn.Linear modules.

    Targets one-layer / frozen-base training where most of the model is
    read-only. Quantizing those weights cuts resident VRAM ~2x (int8) or ~4x
    (int4) at the cost of needing torchao installed and the standard
    weight-only-int* matmul kernels at forward time. Must run AFTER
    freeze_layers_by_patterns and BEFORE FSDP wrap.
    """
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    except ImportError as e:
        raise ImportError(
            "quantize_frozen=true requires torchao. Install with `pip install torchao`."
        ) from e

    config = Int8WeightOnlyConfig() if dtype == "int8" else Int4WeightOnlyConfig()

    def is_fully_frozen_linear(module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        # recurse=False: only this module's own params. Linear has weight (+optional bias) only.
        return all(not p.requires_grad for p in module.parameters(recurse=False))

    n_quantized = sum(1 for m, n in [(m, n) for n, m in model.named_modules()] if is_fully_frozen_linear(m, n))
    quantize_(model, config, filter_fn=is_fully_frozen_linear)
    logger.info_rank0(f"Quantized {n_quantized} frozen Linear modules with torchao {dtype} weight-only.")


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def _prune_old_checkpoints(checkpoint_dir: str, keep: int) -> None:
    """Delete step-based checkpoints older than the most recent `keep` ones."""
    pattern = re.compile(r"global_step_(\d+)$")
    entries = []
    try:
        for name in os.listdir(checkpoint_dir):
            m = pattern.match(name)
            if m:
                entries.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))
    except FileNotFoundError:
        return
    entries.sort()
    for _, path in entries[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")

    # Initialize process group with extended timeout to handle long evaluation periods
    # Default timeout is 10 minutes, but evaluation can take 15-30 minutes
    if os.getenv("WORLD_SIZE"):
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
    else:
        # Fallback for single-device/standalone runs
        dist.init_process_group(backend="gloo", init_method="file:///tmp/rank0", world_size=1, rank=0)
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    time_checkpoint_dir = os.path.join(args.train.output_dir, "last_checkpoint")
    time_checkpoint_dir_exists = args.train.save_time_interval_minutes > 0
    if time_checkpoint_dir_exists and args.train.global_rank == 0:
        os.makedirs(time_checkpoint_dir, exist_ok=True)
    if time_checkpoint_dir_exists and dist.get_world_size() > 1:
        dist.barrier()

    latest_checkpoint_path = None
    if args.train.auto_resume:
        if time_checkpoint_dir_exists:
            latest_checkpoint_path = helper.find_latest_time_checkpoint(time_checkpoint_dir)

        if latest_checkpoint_path is None:
            latest_checkpoint_path = helper.find_latest_step_checkpoint(args.train.save_checkpoint_path)

    if args.train.load_checkpoint_path:
        latest_checkpoint_path = args.train.load_checkpoint_path

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    print(f'tokenizer.mask_token_id: {tokenizer.mask_token_id}')
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        eval_dataloader = None
        if args.data.eval_size > 0 and args.data.datasets_type == "mapping":
            from veomni.data.data_collator import DataCollatorWithPadding
            full_dataset = train_dataset
            n_total = len(full_dataset)
            n_eval = min(args.data.eval_size, max(1, n_total // 10))
            n_train = n_total - n_eval
            if n_train <= 0:
                logger.warning_rank0(f"eval_size={args.data.eval_size} >= dataset size={n_total}, skipping eval split")
            else:
                train_dataset = torch.utils.data.Subset(full_dataset, range(n_train))
                eval_subset = torch.utils.data.Subset(full_dataset, range(n_train, n_total))
                eval_collate = DataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id or 0)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_subset,
                    batch_size=args.train.micro_batch_size,
                    shuffle=False,
                    num_workers=min(args.data.num_workers, 2),
                    pin_memory=True,
                    collate_fn=eval_collate,
                )
                logger.info_rank0(f"Eval split: {n_eval} examples held out from {n_total} total")

        trajectory_dataset = None
        trajectory_collator = None
        _traj_path = getattr(args.train, "trajectory_data_path", None)
        if _traj_path and getattr(args.train, "repr_align_wt", 0) > 0:
            trajectory_dataset = TrajectoryDataset(_traj_path)
            logger.info_rank0(
                f"Trajectory dataset loaded: {len(trajectory_dataset)} samples from {_traj_path}"
            )
            trajectory_collator = DataCollatorWithTrajectoryMasking(
                mask_token_id=tokenizer.mask_token_id,
                trajectory_dataset=trajectory_dataset,
                current_mask_ratio=getattr(args.train, "trajectory_min_mask_ratio", 0.0),
                max_mask_ratio=getattr(args.train, "trajectory_max_mask_ratio", 0.8),
                current_block_size=32,
                use_blockwise_loss=getattr(args.train, "trajectory_use_blockwise", False),
            )
            # When using trajectory collator, override to avoid random masking collator
            _enable_masking = True
            _custom_collate = trajectory_collator
        else:
            _enable_masking = args.train.enable_masking
            _custom_collate = None

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            enable_masking=_enable_masking,
            mask_token_id=tokenizer.mask_token_id,
            collate_fn=_custom_collate,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    eval_dataloader = None

    def run_eval(model, eval_dataloader, tokenizer, args, max_batches=20):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= max_batches:
                    break
                batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch, use_cache=False, repr_align_wt=0.0)
                loss = outputs.loss
                if torch.isfinite(loss):
                    n_tokens = batch.get("attention_mask", batch.get("input_ids")).sum().item()
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
        model.train()
        if total_tokens == 0:
            return None, None
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        return avg_loss, perplexity

    logger.info_rank0("Prepare model")
    # Enter ZeRO-3 Init context BEFORE model creation so DeepSpeed partitions each
    # param on-the-fly during __init__.  Peak RAM per rank = 1/N of total params
    # instead of the full model on every rank.  The context is exited after all
    # model params have been registered (before build_parallelize_model).
    # patch_deepspeed_zero_init_for_meta_tensors() is applied first to handle the
    # meta tensors created by accelerate's init_empty_weights() inside the loader.
    _ds_zero_ctx = None
    if args.train.data_parallel_mode == "deepspeed" and args.train.ds_zero_stage == 3 and args.train.init_device in ("meta", "cpu"):
        import deepspeed as _ds_import

        from veomni.distributed.deepspeed_init import (
            build_ds_config as _build_ds_config_early,
        )
        from veomni.distributed.deepspeed_init import (
            patch_deepspeed_zero_init_for_meta_tensors as _patch_ds_meta,
        )
        _patch_ds_meta()
        _ds_remote_device = "nvme" if args.train.ds_offload_param == "nvme" else "cpu"
        _ds_zero_ctx = _ds_import.zero.Init(
            config_dict_or_path=_build_ds_config_early(args.train),
            remote_device=_ds_remote_device,
        )
        _ds_zero_ctx.__enter__()

    time.sleep(args.train.global_rank * 2)
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        make_teacher=args.train.repr_align_wt > 0,
        config_kwargs={"tau": getattr(args.model, "tau", 0.1)} if args.model.attn_implementation == "tropical" else None,
        anchor_cache_dir=getattr(args.train, "anchor_cache_dir", None),
        align_layers=getattr(args.train, "align_layers", None),
        repr_align_sub_sample_ratio=getattr(args.train, "repr_align_sub_sample_ratio", 1.0),
        repr_align_num_sample_layers=getattr(args.train, "repr_align_num_sample_layers", None),
        enable_nvfp4_qat=getattr(args.model, "enable_nvfp4_qat", False),
        enable_qlorafy=getattr(args.model, "enable_qlorafy", False),
        qlorafy_config=getattr(args.model, "qlorafy_config", None),
    )

    model_config = model.config
    # lm_head_module = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    # if lm_head_module is not None:
    #     for param in lm_head_module.parameters():
    #         param.requires_grad_(False)
    #     logger.info_rank0("Frozen LM head parameters.")

    # Freeze layers based on patterns
    freeze_layers_by_patterns(model, args.train.freeze_layers, logger)

    # Optional: shrink frozen weights via torchao weight-only quantization.
    # Must run after freeze_layers_by_patterns (we filter on requires_grad) and
    # before build_parallelize_model (so FSDP sees the quantized tensors).
    if args.train.quantize_frozen:
        quantize_frozen_linears(model, args.train.quantize_frozen_dtype, logger)

    helper.print_device_mem_info("VRAM usage after building model")

    # ------------------------------------------------------------------
    # Optional: wrap with Cola DLM (Text VAE + block-causal DiT) aux head.
    # Off when cola_wt == 0. See veomni/models/cola_ldm/.
    # ------------------------------------------------------------------
    if args.train.cola_wt > 0:
        from veomni.models.cola_ldm import ColaReprAlignWrapper, build_cola_head

        cfg = model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            base_dim = cfg.text_config.hidden_size
        else:
            base_dim = cfg.hidden_size

        cola_variant = getattr(args.train, "cola_variant", "block_causal")
        cola_head = build_cola_head(
            dim=base_dim,
            variant=cola_variant,
            num_global=args.train.cola_num_global,
            num_local=args.train.cola_num_local,
            block_size=args.train.cola_block_size,
            encoder_depth=args.train.cola_encoder_depth,
            diffusion_depth=args.train.cola_diffusion_depth,
            heads=args.train.cola_heads,
            prediction_type=args.train.cola_prediction,
            lambda_tail=getattr(args.train, "cola_lambda_tail", 0.6),
        )
        n_cola_params = sum(p.numel() for p in cola_head.parameters())
        logger.info_rank0(
            f"Cola head: dim={base_dim} variant={cola_variant} "
            f"global={args.train.cola_num_global} "
            f"local={args.train.cola_num_local} block={args.train.cola_block_size} "
            f"enc_depth={args.train.cola_encoder_depth} diff_depth={args.train.cola_diffusion_depth} "
            f"pred={args.train.cola_prediction} "
            f"params={n_cola_params/1e6:.1f}M  detach_student={args.train.cola_detach_student}"
        )

        # Move cola_head to same device as the base model before wrapping.
        # build_cola_head() creates on CPU; the LM may already be on CUDA.
        _lm_device = next(model.parameters()).device
        cola_head = cola_head.to(_lm_device)
        model = ColaReprAlignWrapper(
            lm=model,
            cola_head=cola_head,
            cola_wt=args.train.cola_wt,
            cola_source_layer=args.train.cola_source_layer,
            cola_detach_student=args.train.cola_detach_student,
        )

    # Exit zero.Init() context: all params now ZeRO-3 partitioned, no more new params expected.
    if _ds_zero_ctx is not None:
        _ds_zero_ctx.__exit__(None, None, None)
        _ds_zero_ctx = None

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=list(model._no_split_modules) + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )


    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        if optimizer_pre_hook is not None:
            optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    # ── DeepSpeed engine init (must happen after optimizer + lr_scheduler) ──
    ds_engine = None
    if args.train.data_parallel_mode == "deepspeed":
        from veomni.distributed.deepspeed_init import build_ds_config, init_deepspeed_engine

        ds_config = build_ds_config(args.train)
        logger.info_rank0(f"DeepSpeed config: {json.dumps(ds_config, indent=2)}")
        ds_engine, optimizer, lr_scheduler = init_deepspeed_engine(
            model, optimizer, lr_scheduler, args.train, ds_config
        )
        model = ds_engine

        # Load actual HF weights into the ZeRO-3 partitioned model one shard at a time.
        # Only needed for ZeRO-3 (meta init); ZeRO-1/2 loads weights normally before DS init.
        if args.train.ds_zero_stage == 3 and args.train.init_device in ("meta", "cpu"):
            from veomni.distributed.deepspeed_init import load_hf_weights_zero3
            logger.info_rank0(f"Loading HF weights into ZeRO-3 model from {args.model.model_path}")
            load_hf_weights_zero3(ds_engine.module, args.model.model_path)

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                tags=["train"],
                resume="allow" if latest_checkpoint_path else None,
                entity=args.train.wandb_entity,
                id=args.train.wandb_name if latest_checkpoint_path else None,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        # save model_assets before training
        model_assets = [model_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    time_checkpoint_timer = None
    if time_checkpoint_dir_exists:
        time_checkpoint_timer = helper.PeriodicTimer(args.train.save_time_interval_minutes * 60)
        time_checkpoint_timer.reset()

    if latest_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(latest_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {latest_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    consecutive_nan_steps = 0
    nan_abort_threshold = 3

    replay_buffer: Optional[ReprAlignReplayBuffer] = None
    _replay_capacity = getattr(args.train, "replay_buffer_capacity", 0)
    if _replay_capacity > 0 and getattr(args.train, "repr_align_wt", 0) > 0:
        replay_buffer = ReprAlignReplayBuffer(capacity=_replay_capacity)
        logger.info_rank0(
            f"ReprAlign replay buffer initialised (capacity={_replay_capacity}, "
            f"prob={args.train.replay_prob}, warmup={args.train.replay_warmup_steps})"
        )

    # d3LLM curriculum schedule — parse once before training loop
    _traj_prog_blocks = getattr(args.train, "trajectory_progressive_block_sizes", None)
    if _traj_prog_blocks is not None:
        _traj_prog_blocks = [int(x) for x in _traj_prog_blocks.split(",")]
    _traj_min_mask = getattr(args.train, "trajectory_min_mask_ratio", 0.0)
    _traj_max_mask = getattr(args.train, "trajectory_max_mask_ratio", 0.8)
    _traj_entropy_wt = getattr(args.train, "trajectory_entropy_weight", 0.0)

    if trajectory_collator is not None:
        logger.info_rank0(
            f"d3LLM trajectory distillation active: "
            f"mask_ratio [{_traj_min_mask} → {_traj_max_mask}], "
            f"progressive_blocks={_traj_prog_blocks or 'full-seq'}, "
            f"entropy_wt={_traj_entropy_wt}"
        )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            # Update trajectory collator curriculum schedule
            if trajectory_collator is not None:
                progress = min(global_step / max(args.train.train_steps, 1), 1.0)
                trajectory_collator.current_mask_ratio = _traj_min_mask + progress * (_traj_max_mask - _traj_min_mask)
                if _traj_prog_blocks:
                    ep = min(epoch, len(_traj_prog_blocks) - 1)
                    trajectory_collator.current_block_size = _traj_prog_blocks[ep]

            step_loss_components: Dict[str, float] = {}

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()

            # NaN param check: skip for quantized models (NF4 params are uint8, never NaN)
            if global_step > 1 and args.train.local_rank == 0 and not getattr(args.model, "enable_qlorafy", False):
                nan_params = [
                    n for n, p in list(model.named_parameters())[:20]
                    if p.data.is_floating_point() and not torch.isfinite(p.data).all()
                ]
                if nan_params:
                    logger.warning_rank0(f"[step {global_step}] NaN/Inf in params before forward: {nan_params[:5]}")

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    outputs = model(**micro_batch, use_cache=False, repr_align_wt=args.train.repr_align_wt)
                    loss_tensor: "torch.Tensor" = outputs.loss.mean() / len(micro_batches)
                    loss_components = getattr(outputs, "loss_components", {})
                    for name, value in loss_components.items():
                        step_loss_components[name] = step_loss_components.get(name, 0.0) + value / len(micro_batches)

                    # Repr-Align replay buffer: push current batch, sample past batch,
                    # re-run student alignment on old data to prevent forgetting.
                    if replay_buffer is not None:
                        replay_buffer.push(micro_batch)
                        _replay_prob = getattr(args.train, "replay_prob", 0.3)
                        _replay_warmup = getattr(args.train, "replay_warmup_steps", 50)
                        _replay_weight = getattr(args.train, "replay_weight", 0.1)
                        if (
                            len(replay_buffer) >= _replay_warmup
                            and torch.rand(1, device=loss_tensor.device).item() < _replay_prob
                        ):
                            _replay_batch = replay_buffer.sample(loss_tensor.device)
                            if _replay_batch is not None:
                                _replay_outputs = model(
                                    **_replay_batch, use_cache=False, repr_align_wt=args.train.repr_align_wt
                                )
                                _replay_align = getattr(_replay_outputs, "loss_components", {}).get("repr_align", None)
                                if _replay_align is not None and torch.isfinite(torch.tensor(_replay_align)):
                                    loss_tensor = loss_tensor + _replay_weight * _replay_align / len(micro_batches)
                                    step_loss_components["replay"] = (
                                        step_loss_components.get("replay", 0.0)
                                        + _replay_align / len(micro_batches)
                                    )

                    # d3LLM-style entropy regularization on correctly-predicted masked tokens
                    if trajectory_collator is not None and _traj_entropy_wt > 0:
                        _logits = getattr(outputs, "logits", None)
                        _labels = micro_batch.get("labels", None)
                        if _logits is not None and _labels is not None:
                            from torch.nn import functional as _F
                            _logits_s = _logits[:, :-1].float()
                            _labels_s = _labels[:, 1:]
                            _L = min(_logits_s.shape[1], _labels_s.shape[1])
                            _logits_s = _logits_s[:, :_L]
                            _labels_s = _labels_s[:, :_L]
                            _mask = _labels_s != IGNORE_INDEX
                            if _mask.any():
                                _temp = getattr(args.train, "trajectory_temperature", 0.5)
                                _mlogits = _logits_s[_mask]
                                _mlabels = _labels_s[_mask]
                                _probs = _F.softmax(_mlogits / _temp, dim=-1)
                                _H = -(_probs * torch.log(_probs + 1e-12)).sum(dim=-1)
                                _pred = _mlogits.argmax(dim=-1)
                                _correct = _pred == _mlabels
                                if _correct.any():
                                    _ent_loss = (_H * _correct).sum() / _correct.sum().clamp_min(1)
                                    loss_tensor = loss_tensor + _traj_entropy_wt * _ent_loss / len(micro_batches)
                                    step_loss_components["trajectory_entropy"] = (
                                        step_loss_components.get("trajectory_entropy", 0.0)
                                        + _ent_loss.item() / len(micro_batches)
                                    )

                    step_has_nan = not torch.isfinite(loss_tensor)
                    if step_has_nan and args.train.local_rank == 0:
                        comp_str = ", ".join(f"{k}={v:.4f}" for k, v in loss_components.items())
                        logger.warning_rank0(
                            f"[step {global_step}] NaN loss detected. raw_loss={outputs.loss.mean():.4f} components=[{comp_str}]"
                        )

                if not step_has_nan:
                    with model_bwd_context:
                        if ds_engine is not None:
                            ds_engine.backward(loss_tensor)
                        else:
                            loss_tensor.backward()
                else:
                    logger.warning_rank0(f"[step {global_step}] Skipping backward for NaN loss")

                if not step_has_nan and global_step <= 3 and args.train.local_rank == 0:
                    nan_grads = [
                        n for n, p in model.named_parameters()
                        if p.grad is not None and not torch.isfinite(p.grad).all()
                    ]
                    if nan_grads:
                        logger.warning_rank0(
                            f"[step {global_step}] NaN grads in {len(nan_grads)} params: {nan_grads[:5]}"
                        )
                    else:
                        logger.info_rank0(f"[step {global_step}] All gradients finite after backward")

                if step_has_nan:
                    total_loss += 0.0
                else:
                    total_loss += loss_tensor.item()
                del micro_batch

            step_had_nan = math.isnan(total_loss) or math.isinf(total_loss)
            if step_had_nan:
                if ds_engine is None:
                    optimizer.zero_grad()
                consecutive_nan_steps += 1
                logger.warning_rank0(
                    f"[step {global_step}] NaN/Inf loss ({consecutive_nan_steps}/{nan_abort_threshold}): "
                    f"skipping optimizer step"
                )
                if consecutive_nan_steps >= nan_abort_threshold:
                    nan_params = [
                        n for n, p in list(model.named_parameters())[:50]
                        if p.data.is_floating_point() and not torch.isfinite(p.data).all()
                    ]
                    logger.warning_rank0(
                        f"ABORT: {nan_abort_threshold} consecutive NaN steps. "
                        f"NaN params (first 5): {nan_params[:5]}"
                    )
                    raise RuntimeError(
                        f"Training aborted: {nan_abort_threshold} consecutive NaN/Inf loss steps "
                        f"at global_step={global_step}."
                    )
                data_loader_tqdm.update()
                continue

            consecutive_nan_steps = 0

            qlora_lora_gnorm = 0.0
            qlora_lora_pnorm = 0.0
            if args.model.enable_qlorafy:
                for n, p in model.named_parameters():
                    if "lora_" in n:
                        qlora_lora_pnorm += float(p.data.float().norm(2).item()) ** 2
                        if p.grad is not None:
                            qlora_lora_gnorm += float(p.grad.data.float().norm(2).item()) ** 2

            if ds_engine is not None:
                # DeepSpeed handles gradient clipping + opt step + zero_grad internally
                ds_engine.step()
                grad_norm = ds_engine.get_global_grad_norm()
                if grad_norm is None:  # ZeRO-2 + CPU optimizer may not compute norm
                    grad_norm = 0.0
            elif args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            reduce_group = dist.group.WORLD if ds_engine is not None else get_parallel_state().fsdp_group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=reduce_group)
            if step_loss_components:
                names = sorted(step_loss_components.keys())
                values = tuple(step_loss_components[name] for name in names)
                reduced_values = all_reduce(values, group=reduce_group)
                if not isinstance(reduced_values, (tuple, list)):
                    reduced_values = (reduced_values,)
                step_loss_components = {name: value for name, value in zip(names, reduced_values)}

            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            if torch.cuda.is_available():
                train_metrics["system/vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                train_metrics["system/vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            for name, value in step_loss_components.items():
                train_metrics[f"losses/{name}"] = value

            component_parts = [
                f"{name}:{step_loss_components[name]:.2f}"
                for name in sorted(step_loss_components.keys())
            ]
            postfix_components = ", " + ", ".join(component_parts) if component_parts else ""
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}{postfix_components}"
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    # ----------------------------------------------------------
                    # Cola DLM extras (only when active). Scalars already
                    # live in step_loss_components → train_metrics; here
                    # we add (a) a separate grad-norm of the Cola head and
                    # (b) periodic histograms of z_global / z_local.
                    # ----------------------------------------------------------
                    if args.train.cola_wt > 0:
                        # Cola head grad norm (rank-local; informative even
                        # without all-reduce since FSDP shards uniformly).
                        try:
                            cola_root = model
                            for attr in ("module", "lm"):
                                if hasattr(cola_root, "cola_head"):
                                    break
                                cola_root = getattr(cola_root, attr, cola_root)
                            cola_module = getattr(cola_root, "cola_head", None)
                            if cola_module is not None:
                                cg = 0.0
                                for p in cola_module.parameters():
                                    if p.grad is not None:
                                        cg += float(p.grad.data.float().norm(2).item()) ** 2
                                train_metrics["cola/grad_norm"] = cg ** 0.5
                        except Exception:
                            pass

                        # Periodic histograms (cheap-ish — z_global/local are tiny)
                        hist_every = max(int(args.train.cola_log_hist_every), 0)
                        extras = getattr(outputs, "cola_extras", None)
                        if hist_every > 0 and extras is not None and global_step % hist_every == 0:
                            zg = extras["z_global"].flatten().numpy()
                            zl = extras["z_local"].flatten().numpy()
                            train_metrics["cola_hist/z_global"] = wandb.Histogram(zg)
                            train_metrics["cola_hist/z_local"] = wandb.Histogram(zl)

                        # Variant-specific metrics
                        if extras is not None:
                            if "tail_start" in extras:
                                train_metrics["cola/card_tail_start"] = extras["tail_start"]
                                train_metrics["cola/card_tail_ratio"] = extras["tail_ratio"]
                            if "complementary_mask_ratio" in extras:
                                train_metrics["cola/fast_block_mask_ratio"] = extras["complementary_mask_ratio"]

                    if args.model.enable_qlorafy:
                        lora_pnorm = qlora_lora_pnorm ** 0.5
                        lora_gnorm = qlora_lora_gnorm ** 0.5
                        train_metrics["qlora/param_norm"] = lora_pnorm
                        train_metrics["qlora/grad_norm"] = lora_gnorm
                        train_metrics["qlora/grad_to_param_ratio"] = lora_gnorm / max(lora_pnorm, 1e-8)

                    if args.model.enable_qlorafy and global_step % 100 == 0:
                        try:
                            model.eval()
                            gen_prompts = ["The meaning of life is", "def fibonacci(n):"]
                            gen_samples = []
                            for gp in gen_prompts:
                                enc = tokenizer(gp, return_tensors="pt")
                                pids = enc.input_ids.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    # AR generation
                                    ar_out = model.generate(pids, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=200)
                                    ar_text = tokenizer.decode(ar_out[0][pids.shape[1]:], skip_special_tokens=True)

                                    # Diffusion generation (masked, bidirectional)
                                    from veomni.models.transformers.qwen2.generation_utils import mdm_generate
                                    if tokenizer.mask_token_id is not None:
                                        diff_ids = mdm_generate(model, pids, mask_token_id=tokenizer.mask_token_id, max_new_tokens=64, steps=16, temperature=0.7)
                                        diff_text = tokenizer.decode(diff_ids[0][pids.shape[1]:], skip_special_tokens=True)
                                    else:
                                        diff_text = "(no mask token)"
                                gen_samples.append(
                                    f"<b>Prompt:</b> {gp}<br>"
                                    f"<b>AR:</b> {ar_text}<br>"
                                    f"<b>Diffusion (16-step):</b> {diff_text}"
                                )
                            train_metrics["generation/sample"] = wandb.Html("<hr>".join(gen_samples))
                            model.train()
                        except Exception as e:
                            logger.warning_rank0(f"[step {global_step}] Generation probe failed: {e}")

                    wandb.log(train_metrics, step=global_step)

                if eval_dataloader is not None and args.train.eval_every > 0 and global_step % args.train.eval_every == 0 and global_step > 0:
                    try:
                        eval_loss, eval_ppl = run_eval(model, eval_dataloader, tokenizer, args)
                        if eval_loss is not None and args.train.global_rank == 0:
                            logger.info_rank0(f"[step {global_step}] eval loss={eval_loss:.4f} ppl={eval_ppl:.2f}")
                            if args.train.use_wandb:
                                wandb.log({"eval/loss": eval_loss, "eval/perplexity": eval_ppl}, step=global_step)
                    except Exception as e:
                        logger.warning_rank0(f"[step {global_step}] Eval failed: {e}")

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
            save_step = args.train.save_steps and global_step % args.train.save_steps == 0
            eval_step = args.train.eval_every > 0 and global_step % args.train.eval_every == 0
            # Check time-based save trigger - synchronize decision across all ranks to prevent deadlock
            save_time = False
            if time_checkpoint_dir_exists and time_checkpoint_timer is not None:
                # Only rank 0 checks the timer
                if args.train.global_rank == 0:
                    save_time = time_checkpoint_timer.should_trigger()
                # Broadcast rank 0's decision to all ranks
                save_time_tensor = torch.tensor([int(save_time)], dtype=torch.int32, device='cuda')
                dist.broadcast(save_time_tensor, src=0)
                save_time = bool(save_time_tensor.item())

            if save_step or eval_step:
                helper.empty_cache()
                is_qlora = getattr(args.model, 'enable_qlorafy', False)
                skip_dcp = is_qlora or not args.train.save_optimizer_state

                if save_step:
                    save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                elif eval_step:
                    save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "eval")
                else:
                    raise ValueError("Invalid save or eval step")

                if not skip_dcp:
                    os.makedirs(save_checkpoint_path, exist_ok=True)
                    state = {
                        "model": model,
                        "optimizer": optimizer,
                        "extra_state": {
                            "global_step": global_step,
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "train_dataloader": train_dataloader.state_dict(),
                            "environ_meter": environ_meter.state_dict(),
                            "torch_rng_state": torch.get_rng_state(),
                        },
                    }
                    Checkpointer.save(save_checkpoint_path, state)
                    logger.info_rank0(f"Checkpoint saved to {save_checkpoint_path}")
                else:
                    os.makedirs(save_checkpoint_path, exist_ok=True)
                    logger.info_rank0("Skipping DCP checkpoint (QLoRA/Params4bit or save_optimizer_state=False); adapter weights only.")

                if args.train.global_rank == 0 and args.train.save_total_limit > 0:
                    _prune_old_checkpoints(args.train.save_checkpoint_path, args.train.save_total_limit)

                # Barrier after checkpoint save, before evaluation
                # This ensures all ranks have completed checkpoint before rank 0 starts eval
                dist.barrier()

                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    if is_qlora:
                        from peft import PeftModel
                        if isinstance(model, PeftModel):
                            model.save_pretrained(hf_weights_path)
                            logger.info_rank0(f"PEFT adapter saved at {hf_weights_path}")
                        else:
                            adapter_state = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
                            save_model_weights(hf_weights_path, adapter_state, model_assets=model_assets)
                            logger.info_rank0(f"Trainable weights saved at {hf_weights_path}")
                    else:
                        logger.info(f"Converting checkpoint from {save_checkpoint_path} to HF format")
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

                    # Run HumanEval evaluation
                    eval_output_path = os.path.join(save_checkpoint_path, "humaneval")
                    logger.info(f"Starting HumanEval evaluation for global_step {global_step}")

                    # Extract a clean model name for directory creation (avoid path sanitization)
                    # Use just the checkpoint directory name instead of full path

                    cmd = [
                            "python", "eval/eval_completion/eval_single.py",
                            "--model", "custom_coder",
                            "--model_args",
                            f"pretrained={hf_weights_path},"
                            "max_new_tokens=128,"
                            "steps=128,"
                            "add_bos_token=true,"
                            "temperature=0.8,"
                            "top_p=0.95,"
                            "alg=p2",
                            "--tasks", "humaneval",
                            "--num_fewshot", "0",
                            "--batch_size", "10",
                            "--output_path", eval_output_path,
                            "--log_samples",
                            "--confirm_run_unsafe_code",
                    ]
                    env = dict(os.environ)
                    env.update({"HF_ALLOW_CODE_EVAL": "1"})
                    try:
                        result = subprocess.run(
                            cmd,
                            env=env,
                            stdout=sys.stdout,
                            stderr=sys.stderr
                        )
                    except Exception as e:
                        logger.error(f"HumanEval evaluation failed: {e}")

                    eval_scores = extract_humaneval_scores(eval_output_path)

                    if eval_scores and args.train.use_wandb:
                        wandb_metrics = {
                            f"eval/humaneval/{k}": v
                            for k, v in eval_scores.items()
                        }
                        wandb.log(wandb_metrics, step=global_step)
                        logger.info(f"Logged HumanEval scores to wandb: {wandb_metrics}")

                # Note: No barrier here! Other ranks continue immediately while rank 0 evaluates.
                # This prevents timeout when evaluation takes longer than NCCL timeout.
                logger.info_rank0(f"Checkpoint saved at {save_checkpoint_path} successfully!")

            if save_time and args.train.save_optimizer_state:
                helper.empty_cache()
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                helper.save_time_checkpoint(Checkpointer, time_checkpoint_dir, state)
                dist.barrier()
                logger.info_rank0("Time-based checkpoint refreshed at last_checkpoint/")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            if args.train.save_optimizer_state:
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            if args.train.global_rank == 0 and args.train.save_total_limit > 0:
                _prune_old_checkpoints(args.train.save_checkpoint_path, args.train.save_total_limit)
            # save model in huggingface's format
            if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
                hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                model_state_dict = ckpt_to_state_dict(
                    save_checkpoint_path=save_checkpoint_path,
                    output_dir=args.train.output_dir,
                    ckpt_manager=args.train.ckpt_manager,
                )
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
