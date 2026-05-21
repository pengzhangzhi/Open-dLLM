"""Precompute frozen-teacher hidden states for Repr-Align training.

Loads a Qwen3 (or compatible) HF checkpoint, iterates a plaintext JSONL dataset
using the SAME tokenization the trainer's `process_pretrain_example` uses, then
runs a forward pass per chunk and dumps the selected layers' hiddens to
safetensors files sharded by SHA-256 of the chunk's input_ids.

The trainer's pipeline (see veomni/data/data_transform.py:process_pretrain_example):
    tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_token_id]
    chunks = split_into_chunks(tokens, max_seq_len)
Each chunk becomes one training example (no padding). Multiple chunks are then
packed into one rmpad row by the collator. `CachedTeacher` splits the packed
row using position_ids before looking up per-chunk hidden states, so the
precompute must produce exactly one cache file per chunk, keyed by the hash of
the chunk's unpadded input_ids tensor.

Example:
    python scripts/precompute_anchor.py \\
        --model_path Qwen/Qwen3-1.7B \\
        --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \\
        --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \\
        --layers 7,14,21,28 \\
        --max_seq_len 2048 \\
        --max_examples 1000

# Example: First batch of layers at 260k
.venv/bin/python scripts/precompute_anchor.py \
  --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
  --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data_smoke_1000.jsonl \
  --output_dir /home/johndpope/ds_offload/anchors/qwen3.6-27b-260k \
  --layers all \
  --layer_batch 1 \
  --max_seq_len 260000 \
  --max_examples 1000 \
  --seed 42 \
  --device_map auto \
  --max_memory '{"0": "38GiB", "1": "28GiB", "cpu": "80GiB"}'

"""
import os 
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterator, List

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def hash_chunk(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()[:16]


def cache_path(output_dir: Path, h: str) -> Path:
    return output_dir / h[:2] / f"{h}.safetensors"


def iter_jsonl_texts(path: str, text_key: str, max_examples: int | None) -> Iterator[str]:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            try:
                yield json.loads(line)[text_key]
            except (json.JSONDecodeError, KeyError):
                continue


def chunk_text(tokenizer, text: str, max_seq_len: int) -> list[list[int]]:
    """Reproduce trainer's process_pretrain_example chunking exactly."""
    tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    return [tokens[i : i + max_seq_len] for i in range(0, len(tokens), max_seq_len)]


def hash_chunk(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()[:16]


def cache_path(output_dir: Path, h: str) -> Path:
    return output_dir / h[:2] / f"{h}.safetensors"


def iter_jsonl_texts(path: str, text_key: str, max_examples: int | None) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            try:
                yield json.loads(line)[text_key]
            except (json.JSONDecodeError, KeyError):
                continue

def hash_chunk(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()[:16]


def cache_path(output_dir: Path, h: str) -> Path:
    return output_dir / h[:2] / f"{h}.safetensors"


def iter_jsonl_texts(path: str, text_key: str, max_examples: int | None) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            try:
                yield json.loads(line)[text_key]
            except Exception:
                continue


def chunk_text(tokenizer, text: str, max_seq_len: int) -> List[List[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    return [tokens[i : i + max_seq_len] for i in range(0, len(tokens), max_seq_len)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--layer_batch", type=int, default=1)
    ap.add_argument("--start_layer", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_memory", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] max_seq_len={args.max_seq_len} | Processing layer(s): {args.start_layer}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Model
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
    }
    if args.max_memory:
        raw = json.loads(args.max_memory)
        model_kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in raw.items()}

    print("[precompute] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_layers = model.config.num_hidden_layers

    # One layer per run
    layer_idx = args.start_layer % (num_layers + 1)
    layers = [layer_idx]
    print(f"[precompute] → Computing **LAYER {layer_idx}** only")

    # Manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "model_path": args.model_path,
            "max_seq_len": args.max_seq_len,
            "layers": layers,
            "dtype": args.dtype,
            "seed": args.seed,
        }, f, indent=2)

    device = next(model.parameters()).device

    # ====================== PROCESSING ======================
    skipped = written = chunks_seen = 0
    t0 = time.time()

    with open(args.data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_examples and i >= args.max_examples:
                break
            try:
                text = json.loads(line)["text"]
            except:
                continue

            # Tokenize + chunk exactly like the trainer
            tokens = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
            for j in range(0, len(tokens), args.max_seq_len):
                chunk_tokens = tokens[j : j + args.max_seq_len]
                chunks_seen += 1

                ids_cpu = torch.tensor(chunk_tokens, dtype=torch.long)
                h = hash_chunk(ids_cpu)
                p = cache_path(output_dir, h)

                if p.exists() and not args.force:
                    skipped += 1
                    continue

                # Forward pass
                input_ids = ids_cpu.unsqueeze(0).to(device)
                attn_mask = torch.ones_like(input_ids)

                with torch.inference_mode():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                hs = out.hidden_states
                shard = {
                    f"hidden_layer_{layer_idx}": hs[layer_idx][0].to(dtype).cpu().contiguous()
                }
                shard["input_ids"] = ids_cpu.contiguous()

                p.parent.mkdir(parents=True, exist_ok=True)
                save_file(shard, str(p))
                written += 1

                if (written + skipped) % 20 == 0:
                    rate = (written + skipped) / (time.time() - t0)
                    print(f"[Layer {layer_idx}] {written+skipped:5d} chunks | written={written} | {rate:.1f} chunk/s")

    print(f"\n✅ Finished Layer {layer_idx} | Written: {written} | Skipped: {skipped} | Chunks seen: {chunks_seen}")


if __name__ == "__main__":
    main()