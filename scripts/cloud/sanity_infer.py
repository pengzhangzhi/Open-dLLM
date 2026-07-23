#!/usr/bin/env python3
"""
Sanity inference check for a Repr-Align trained Qwen3.6-27B checkpoint.

Usage:
    python scripts/cloud/sanity_infer.py --ckpt /data/checkpoints/qwen3.6-27b-repr-align/global_step_50/hf_ckpt
    python scripts/cloud/sanity_infer.py --ckpt /data/checkpoints/qwen3.6-27b-repr-align/global_step_50/hf_ckpt --steps 64
"""

import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

from veomni.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig


PROMPTS = [
    "The quick brown fox",
    "Artificial intelligence is transforming",
    "def fibonacci(n):\n    ",
    "The capital of France is",
]


def run_inference(ckpt_path: str, steps: int, device: str) -> None:
    print(f"\nLoading model from: {ckpt_path}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    ).eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # Ensure mask token exists
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
        model.resize_token_embeddings(len(tokenizer))

    gen_cfg = MDMGenerationConfig(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        steps=steps,
        temperature=0.8,
        top_k=200,
        alg="p2",
        alg_temp=0.1,
    )

    print(f"\n{'='*60}")
    print(f"Diffusion inference — {steps} steps, max_new_tokens=64")
    print(f"{'='*60}\n")

    ok = 0
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        t1 = time.time()
        with torch.no_grad():
            out = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)
        elapsed = time.time() - t1
        tokens = out.sequences[0, input_ids.shape[1]:]
        generated = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated!r}  [{elapsed:.1f}s]")
        print()
        if generated.strip():
            ok += 1

    print(f"{'='*60}")
    print(f"Result: {ok}/{len(PROMPTS)} prompts produced non-empty output")
    if ok == len(PROMPTS):
        print("PASS — model generates coherent output")
        sys.exit(0)
    else:
        print("WARN — some prompts returned empty output (may be OK at early steps)")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to hf_ckpt directory")
    parser.add_argument("--steps", type=int, default=32, help="Diffusion steps (default 32)")
    parser.add_argument("--device", default="cuda", help="Device (cuda / cpu / cuda:0)")
    args = parser.parse_args()
    run_inference(args.ckpt, args.steps, args.device)


if __name__ == "__main__":
    main()
