"""Periodic generation probe for QLoRA Repr-Align training.

Runs on a separate GPU (RTX PRO 4000) while training runs on GPU 0.
Loads the latest LoRA checkpoint, generates text, logs to wandb.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/generation_probe.py \
        --base_model /home/johndpope/ds_offload/models/Qwen3.6-27B \
        --checkpoint_dir /home/johndpope/ds_offload/checkpoints/qlorafy-27b-train \
        --wandb_project open-dllm \
        --wandb_name qlorafy-27b-probe \
        --interval 300

If no checkpoint exists yet, runs the base NF4 model without LoRA.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


PROMPTS = [
    "Write a quick sort algorithm in Python.",
    "The meaning of life is",
    "In the year 2050, artificial intelligence will",
    "def fibonacci(n):",
    "The key difference between diffusion models and autoregressive models is",
]


def load_model(base_model_path: str, lora_path: str | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    hf_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    if getattr(hf_config, "language_model_only", False) is False:
        hf_config.language_model_only = True

    total_vram = torch.cuda.get_device_properties(0).total_mem if torch.cuda.is_available() else 0
    max_mem_gb = min(int(total_vram / 1e9) - 4, 28)
    max_memory = {0: f"{max_mem_gb}GiB", "cpu": "32GiB"} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=hf_config,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()

    if lora_path is not None and os.path.isdir(lora_path):
        print(f"[probe] Loading LoRA adapter from {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64, steps: int = 64) -> str:
    from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    gen_cfg = MDMGenerationConfig(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=0.5,
        top_k=200,
        alg="p2",
        alg_temp=0.5,
        num_return_sequences=1,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

    prompt_len = input_ids.shape[1]
    generated = tokenizer.decode(outputs.sequences[0][prompt_len:], skip_special_tokens=True)
    return generated


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    step_dirs = sorted(cp_dir.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not step_dirs:
        return None
    latest = step_dirs[-1]
    hf_ckpt = latest / "hf_ckpt"
    if hf_ckpt.exists():
        return str(hf_ckpt)
    return str(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--wandb_project", default="open-dllm")
    parser.add_argument("--wandb_name", default="qlorafy-27b-probe")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between probes")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    last_ckpt = None
    model = None

    while True:
        ckpt = find_latest_checkpoint(args.checkpoint_dir)
        needs_reload = (ckpt != last_ckpt) or (model is None)

        if needs_reload:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()

            print(f"[probe] Loading model (checkpoint={ckpt})")
            model = load_model(args.base_model, lora_path=ckpt)
            last_ckpt = ckpt

        samples = {}
        for prompt in PROMPTS:
            try:
                generated = generate(model, tokenizer, prompt, args.max_new_tokens, args.steps)
                samples[prompt] = generated
                print(f"[probe] Prompt: {prompt[:50]}...")
                print(f"[probe] Generated: {generated[:100]}...")
            except Exception as e:
                samples[prompt] = f"ERROR: {e}"
                print(f"[probe] Error for '{prompt[:30]}...': {e}")

        step_str = os.path.basename(last_ckpt) if last_ckpt else "base"
        wandb.log(
            {
                "samples_table": wandb.Table(
                    columns=["prompt", "generated", "checkpoint"],
                    data=[[p, g, step_str] for p, g in samples.items()],
                )
            }
        )

        if args.once:
            break

        print(f"[probe] Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
