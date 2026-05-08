import glob
import json
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
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
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




@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    
    # Initialize process group with extended timeout to handle long evaluation periods
    # Default timeout is 10 minutes, but evaluation can take 15-30 minutes
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
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
            enable_masking=args.train.enable_masking,
            mask_token_id=tokenizer.mask_token_id,
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

    logger.info_rank0("Prepare model")
    time.sleep(args.train.global_rank * 2)
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        make_teacher=args.train.repr_align_wt > 0,
    )
    
    model_config = model.config
    # lm_head_module = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    # if lm_head_module is not None:
    #     for param in lm_head_module.parameters():
    #         param.requires_grad_(False)
    #     logger.info_rank0("Frozen LM head parameters.")
    
    # Freeze layers based on patterns
    freeze_layers_by_patterns(model, args.train.freeze_layers, logger)
    helper.print_device_mem_info("VRAM usage after building model")


    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
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

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                tags=["train"],
                resume="allow",
                entity=args.train.wandb_entity,
                id=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
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

                with model_bwd_context:
                    loss_tensor.backward()

                total_loss += loss_tensor.item()
                del micro_batch

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            if step_loss_components:
                names = sorted(step_loss_components.keys())
                values = tuple(step_loss_components[name] for name in names)
                reduced_values = all_reduce(values, group=get_parallel_state().fsdp_group)
                step_loss_components = {name: value for name, value in zip(names, reduced_values)}
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
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
                    wandb.log(train_metrics, step=global_step)

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
                if save_step:
                    save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                elif eval_step:
                    save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"eval")
                else:
                    raise ValueError("Invalid save or eval step")
                print(f"save_checkpoint_path: {save_checkpoint_path}")
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
                
                # Barrier after checkpoint save, before evaluation
                # This ensures all ranks have completed checkpoint before rank 0 starts eval
                dist.barrier()
                
                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
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
            
            if save_time:
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
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
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
