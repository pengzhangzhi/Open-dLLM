"""Precompute frozen-teacher hidden states for Repr-Align training.

Loads a Qwen3 (or compatible) HF checkpoint, iterates a plaintext JSONL dataset,
runs forward with `output_hidden_states=True`, and dumps the selected layers'
hiddens to safetensors files sharded by SHA-256 of input_ids.

The student trainer (`train_torch.py` with `train.anchor_cache_dir` set) then
reads these files via `veomni.models.cached_teacher.CachedTeacher` — no live
teacher needed at train time.

Example:
    python scripts/precompute_anchor.py \\
        --model_path Qwen/Qwen3-1.7B \\
        --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \\
        --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \\
        --layers 6,12,18,24 \\
        --max_seq_len 2048 \\
        --batch_size 4 \\
        --max_examples 1000
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def hash_input_ids(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()[:16]


def cache_path(output_dir: Path, h: str) -> Path:
    return output_dir / h[:2] / f"{h}.safetensors"


def iter_jsonl(path: str, text_key: str, max_examples: int | None):
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            try:
                yield json.loads(line)[text_key]
            except (json.JSONDecodeError, KeyError):
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model path or local dir")
    ap.add_argument("--data_path", required=True, help="JSONL with one text per line")
    ap.add_argument("--output_dir", required=True, help="Where to write cache shards")
    ap.add_argument("--layers", required=True, help="Comma-separated layer indices, e.g. 6,12,18,24")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default=None, help="HF device_map, e.g. 'auto' for big teachers")
    args = ap.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",")})
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest captures the contract this cache satisfies; the trainer checks
    # tokenizer / hidden_size / layers match before using a cache.
    manifest_path = output_dir / "manifest.json"
    print(f"[precompute] output_dir={output_dir}  layers={layers}")

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
        raise ValueError(f"layers {bad} out of range [0, {num_layers}] (model has {num_layers} transformer blocks + 1 embedding output)")

    manifest = {
        "model_path": args.model_path,
        "tokenizer_name_or_path": getattr(tok, "name_or_path", args.model_path),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layers": layers,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "schema": "v1",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[precompute] manifest: {manifest}")

    skipped = written = 0
    t0 = time.time()
    batch_texts: list[str] = []

    def flush(batch_texts: list[str]) -> tuple[int, int]:
        s = w = 0
        if not batch_texts:
            return 0, 0
        enc = tok(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(args.device if not args.device_map else model.device)
        attn = enc.attention_mask.to(input_ids.device)

        # Skip any rows whose cache file already exists.
        keep = []
        keep_hashes = []
        for b in range(input_ids.size(0)):
            h = hash_input_ids(input_ids[b])
            if cache_path(output_dir, h).exists():
                s += 1
            else:
                keep.append(b)
                keep_hashes.append(h)
        if not keep:
            return s, 0

        ids = input_ids[keep]
        am = attn[keep]
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of (num_layers+1) [B, S, D]

        for i, b in enumerate(range(ids.size(0))):
            h = keep_hashes[i]
            p = cache_path(output_dir, h)
            p.parent.mkdir(parents=True, exist_ok=True)
            shard = {f"hidden_layer_{li}": hs[li][b].to(dtype).cpu().contiguous() for li in layers}
            shard["input_ids"] = ids[b].cpu().contiguous()
            shard["attention_mask"] = am[b].cpu().contiguous()
            save_file(shard, str(p))
            w += 1
        return s, w

    for txt in iter_jsonl(args.data_path, args.text_key, args.max_examples):
        batch_texts.append(txt)
        if len(batch_texts) >= args.batch_size:
            s, w = flush(batch_texts)
            skipped += s
            written += w
            batch_texts = []
            done = skipped + written
            if done and done % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[precompute] {done} examples  ({written} written, {skipped} skipped)  {rate:.1f} ex/s")

    s, w = flush(batch_texts)
    skipped += s
    written += w
    elapsed = time.time() - t0
    print(f"[precompute] done. {written} written, {skipped} skipped in {elapsed:.0f}s")
    print(f"[precompute] cache root: {output_dir}")


if __name__ == "__main__":
    main()
