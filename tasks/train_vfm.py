import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.models import build_tokenizer, save_model_assets, save_model_weights
from veomni.models.vfm import VariationalFlowMap
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


@dataclass
class VFMArguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def build_vfm_model(args, tokenizer):
    """Build VFM model from a pretrained bidirectional LLM + noise adapter."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = args.model.model_path
    logger.info_rank0(f"Loading base model from {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))
    hidden_size = config.hidden_size
    vocab_size = base_model.get_input_embeddings().weight.shape[0]

    vfm_cfg = getattr(args.model, "vfm", {}) or {}
    vfm = VariationalFlowMap(
        model=base_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        adapter_layers=vfm_cfg.get("adapter_layers", 4),
        adapter_heads=vfm_cfg.get("adapter_heads", 8),
        tau=vfm_cfg.get("tau", 1.0),
        sigma=vfm_cfg.get("sigma", 1.0),
        alpha=vfm_cfg.get("alpha", 0.5),
        max_seq_len=vfm_cfg.get("max_seq_len", args.data.max_seq_len),
    )

    if vfm_cfg.get("freeze_base", True):
        for name, param in vfm.flow_map.model.named_parameters():
            param.requires_grad = False
        logger.info_rank0("Froze base LLM parameters")

    trainable = sum(p.numel() for p in vfm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vfm.parameters())
    logger.info_rank0(f"VFM params: {trainable:,} trainable / {total:,} total")

    return vfm, config


def vfm_forward(vfm, input_ids, attention_mask, mask_token_id):
    """Run VFM forward pass and return loss dict."""
    out = vfm(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=mask_token_id)
    return out


def run_eval(vfm, train_dataset, mask_token_id, device, tokenizer, max_seq_len, n_samples=20):
    """Evaluate VFM on a few samples from the training data."""
    vfm.eval()
    total_loss = 0.0
    total_data = 0.0
    total_obs = 0.0
    total_kl = 0.0
    n = 0
    with torch.no_grad():
        for i in range(min(n_samples, len(train_dataset))):
            sample = train_dataset[i]
            if not isinstance(sample, dict) or "input_ids" not in sample:
                continue
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample.get("attention_mask", torch.ones_like(sample["input_ids"]))
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.unsqueeze(0).to(device)
            else:
                attention_mask = torch.ones(1, input_ids.shape[1], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = vfm(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=mask_token_id)
            total_loss += out["loss"].item()
            total_data += out["data_loss"].item()
            total_obs += out["obs_loss"].item()
            total_kl += out["kl_loss"].item()
            n += 1
    vfm.train()
    if n == 0:
        return {}
    return {
        "eval/loss": total_loss / n,
        "eval/data_loss": total_data / n,
        "eval/obs_loss": total_obs / n,
        "eval/kl_loss": total_kl / n,
        "eval/perplexity": math.exp(min(total_loss / n, 20)),
    }


def main():
    args = parse_args(VFMArguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # ------------------------------------------------------------------
    # 1. Tokenizer + Data
    # ------------------------------------------------------------------
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    mask_token_id = tokenizer.mask_token_id
    logger.info_rank0(f"mask_token_id: {mask_token_id}")

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
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}")

    if args.data.datasets_type == "mapping":
        train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
    else:
        train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)

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
        enable_masking=True,
        mask_token_id=mask_token_id,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    eval_dataset_ref = train_dataset

    # ------------------------------------------------------------------
    # 2. Build VFM Model
    # ------------------------------------------------------------------
    logger.info_rank0("Building VFM model")
    vfm, model_config = build_vfm_model(args, tokenizer)

    device = torch.device(f"cuda:{args.train.local_rank}")
    vfm = vfm.to(device)

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info_rank0(f"VRAM after model load: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    # ------------------------------------------------------------------
    # 3. Optimizer + LR Scheduler
    # ------------------------------------------------------------------
    trainable_params = [p for p in vfm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
    )

    from veomni.optim import build_lr_scheduler
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        train_steps=args.train.train_steps,
        lr_min=args.train.lr_min,
    )

    helper.print_device_mem_info("VRAM after optimizer")

    # ------------------------------------------------------------------
    # 4. Wandb
    # ------------------------------------------------------------------
    if args.train.global_rank == 0 and args.train.use_wandb:
        resume_id = None
        if args.train.auto_resume:
            latest = helper.find_latest_step_checkpoint(args.train.save_checkpoint_path)
            if latest:
                resume_id = args.train.wandb_name
        wandb.init(
            project=args.train.wandb_project,
            name=args.train.wandb_name,
            tags=["vfm"],
            resume="allow" if resume_id else None,
            entity=args.train.wandb_entity,
            id=args.train.wandb_name,
            config={**vars(args.model), **vars(args.data), **vars(args.train)},
        )

    model_assets = [model_config, tokenizer]
    save_model_assets(args.train.model_assets_dir, model_assets)

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    start_epoch, start_step, global_step = 0, 0, 0
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )
    vfm.train()

    logger.info_rank0(
        f"Start VFM training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )

    consecutive_nan_steps = 0
    nan_abort_threshold = 3

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
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()

            step_had_nan = False

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                ids = micro_batch["input_ids"]
                if global_step % 10 == 0 or ids.max() > 151936:
                    logger.info_rank0(
                        f"[step {global_step}] input_ids: shape={ids.shape}, max={ids.max().item()}, "
                        f"min={ids.min().item()}, n_mask={(ids == mask_token_id).sum().item()}/{ids.numel()}"
                    )

                with model_fwd_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fwd_out = vfm_forward(
                        vfm=vfm,
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch.get("attention_mask"),
                        mask_token_id=mask_token_id,
                    )

                    loss_tensor = fwd_out["loss"].mean() / len(micro_batches)

                    step_has_nan = not torch.isfinite(loss_tensor)
                    if step_has_nan:
                        logger.warning_rank0(f"[step {global_step}] NaN loss in VFM forward")
                        step_had_nan = True
                        continue

                    step_loss_components["vfm/data_loss"] = (
                        step_loss_components.get("vfm/data_loss", 0.0)
                        + fwd_out["data_loss"].item() / len(micro_batches)
                    )
                    step_loss_components["vfm/obs_loss"] = (
                        step_loss_components.get("vfm/obs_loss", 0.0)
                        + fwd_out["obs_loss"].item() / len(micro_batches)
                    )
                    step_loss_components["vfm/kl_loss"] = (
                        step_loss_components.get("vfm/kl_loss", 0.0)
                        + fwd_out["kl_loss"].item() / len(micro_batches)
                    )

                with model_bwd_context:
                    loss_tensor.backward()

                total_loss += loss_tensor.item()
                del micro_batch

            if step_had_nan:
                optimizer.zero_grad()
                consecutive_nan_steps += 1
                logger.warning_rank0(
                    f"[step {global_step}] NaN loss ({consecutive_nan_steps}/{nan_abort_threshold}): skipping optimizer step"
                )
                if consecutive_nan_steps >= nan_abort_threshold:
                    raise RuntimeError(
                        f"Training aborted: {nan_abort_threshold} consecutive NaN steps at step {global_step}"
                    )
                data_loader_tqdm.update()
                continue

            consecutive_nan_steps = 0

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.train.max_grad_norm, foreach=True
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce(
                (total_loss, grad_norm), group=get_parallel_state().fsdp_group
            )

            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            train_metrics.update({
                "training/loss": total_loss,
                "training/grad_norm": grad_norm,
                "training/lr": lr,
            })
            for name, value in step_loss_components.items():
                train_metrics[f"losses/{name}"] = value

            if torch.cuda.is_available():
                train_metrics["system/vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                train_metrics["system/vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

            component_parts = [
                f"{name}:{step_loss_components[name]:.2f}"
                for name in sorted(step_loss_components.keys())
            ]
            postfix = ", " + ", ".join(component_parts) if component_parts else ""
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}{postfix}"
            )
            data_loader_tqdm.update()

            # Wandb logging
            if args.train.global_rank == 0 and args.train.use_wandb:
                log_dict = {**train_metrics}
                log_dict.update(step_loss_components)

                # Adapter stats
                if global_step % 50 == 0:
                    z_gen = fwd_out.get("z_gen")
                    if z_gen is not None:
                        log_dict["vfm_stats/z_gen_norm"] = z_gen.norm(dim=-1).mean().item()
                        log_dict["vfm_stats/z_gen_std"] = z_gen.std(dim=-1).mean().item()

                # Generation sample
                if global_step % 100 == 0:
                    try:
                        vfm.eval()
                        prompt = "The meaning of life is"
                        enc = tokenizer(prompt, return_tensors="pt")
                        pids = enc.input_ids.to(device)
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            gen_ids = vfm.generate(
                                prompt_ids=pids,
                                mask_token_id=mask_token_id,
                                max_new_tokens=32,
                                num_steps=1,
                                temperature=1.0,
                            )
                        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        log_dict["vfm/sample_1step"] = wandb.Html(
                            f"<b>Prompt:</b> {prompt}<br><b>Generated:</b> {gen_text}",
                            inject=False,
                        )
                        vfm.train()
                    except Exception as e:
                        logger.warning(f"VFM generation probe failed: {e}")
                        vfm.train()

                wandb.log(log_dict, step=global_step)

            # Eval
            if args.train.eval_every > 0 and global_step % args.train.eval_every == 0:
                eval_metrics = run_eval(
                    vfm, eval_dataset_ref, mask_token_id, device, tokenizer,
                    args.data.max_seq_len, n_samples=args.data.eval_size,
                )
                if eval_metrics and args.train.global_rank == 0 and args.train.use_wandb:
                    wandb.log(eval_metrics, step=global_step)
                logger.info_rank0(
                    f"[step {global_step}] Eval: loss={eval_metrics.get('eval/loss', 0):.4f}, "
                    f"ppl={eval_metrics.get('eval/perplexity', 0):.2f}"
                )

            # Save checkpoint
            if args.train.save_steps > 0 and global_step % args.train.save_steps == 0:
                ckpt_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": vfm,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(ckpt_path, state)
                logger.info_rank0(f"Checkpoint saved to {ckpt_path}")
                dist.barrier()

                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    hf_path = os.path.join(ckpt_path, "hf_ckpt")
                    state_dict = {
                        n: p.data for n, p in vfm.named_parameters() if p.requires_grad
                    }
                    save_model_weights(hf_path, state_dict, model_assets=model_assets)

        data_loader_tqdm.close()
        start_step = 0

    # Final save
    if args.train.global_rank == 0 and args.train.save_hf_weights:
        hf_path = os.path.join(args.train.save_checkpoint_path, "hf_ckpt_final")
        state_dict = {n: p.data for n, p in vfm.named_parameters() if p.requires_grad}
        save_model_weights(hf_path, state_dict, model_assets=model_assets)
        logger.info_rank0(f"Final weights saved at {hf_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
