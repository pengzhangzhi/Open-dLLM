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
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterator

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model path or local dir")
    ap.add_argument("--data_path", required=True, help="JSONL with one text per line")
    ap.add_argument("--output_dir", required=True, help="Where to write cache shards")
    ap.add_argument("--layers", required=True, help="Comma-separated layer indices, e.g. 7,14,21,28")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default=None, help="HF device_map, e.g. 'auto' for big teachers")
    args = ap.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",")})
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] output_dir={output_dir}  layers={layers}  max_seq_len={args.max_seq_len}")

    print(f"[precompute] loading tokenizer + model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if not args.device_map:
        model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    bad = [i for i in layers if not (0 <= i <= num_layers)]
    if bad:
        raise ValueError(
            f"layers {bad} out of range [0, {num_layers}] "
            f"(model has {num_layers} transformer blocks + 1 embedding output)"
        )

    manifest = {
        "model_path": args.model_path,
        "tokenizer_name_or_path": getattr(tok, "name_or_path", args.model_path),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layers": layers,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "schema": "v2-chunked",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[precompute] manifest: {manifest}")

    device = args.device if not args.device_map else model.device

    skipped = written = chunks_seen = 0
    t0 = time.time()

    for txt in iter_jsonl_texts(args.data_path, args.text_key, args.max_examples):
        for chunk in chunk_text(tok, txt, args.max_seq_len):
            chunks_seen += 1
            ids_cpu = torch.tensor(chunk, dtype=torch.long)
            h = hash_chunk(ids_cpu)
            p = cache_path(output_dir, h)
            if p.exists():
                skipped += 1
                continue

            input_ids = ids_cpu.unsqueeze(0).to(device)
            attn = torch.ones_like(input_ids)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = out.hidden_states  # tuple of (num_layers+1) [1, S, D]

            shard = {
                f"hidden_layer_{li}": hs[li][0].to(dtype).cpu().contiguous() for li in layers
            }
            shard["input_ids"] = ids_cpu.contiguous()
            p.parent.mkdir(parents=True, exist_ok=True)
            save_file(shard, str(p))
            written += 1

            done = written + skipped
            if done and done % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"[precompute] {done} chunks ({written} written, {skipped} skipped, "
                    f"{chunks_seen} seen) {rate:.1f} chunk/s"
                )

    elapsed = time.time() - t0
    print(
        f"[precompute] done. {written} written, {skipped} skipped, "
        f"{chunks_seen} chunks seen in {elapsed:.0f}s"
    )
    print(f"[precompute] cache root: {output_dir}")


if __name__ == "__main__":
    main()
