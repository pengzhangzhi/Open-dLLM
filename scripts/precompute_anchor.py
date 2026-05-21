"""Precompute frozen-teacher hidden states for Repr-Align training.

Loads a model, iterates a JSONL dataset, runs ONE forward pass per chunk
using forward hooks to capture ALL specified layer outputs simultaneously,
and dumps to safetensors files keyed by SHA-256 hash of input_ids.

Each hook fires during the forward pass and immediately moves the layer's
output to CPU, so peak GPU memory holds at most one layer's hidden state
regardless of how many layers are captured.

Usage:
    python scripts/precompute_anchor.py \\
        --model_path /path/to/model \\
        --data_path /path/to/data.jsonl \\
        --output_dir /path/to/anchors \\
        --layers all \\
        --max_seq_len 4096 \\
        --max_examples 1000

Output: {output_dir}/{hash[:2]}/{hash}.safetensors  (one per chunk)
         {output_dir}/manifest.json                   (for CachedTeacher)
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def hash_chunk(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()[:16]


def cache_path(output_dir: Path, h: str) -> Path:
    return output_dir / h[:2] / f"{h}.safetensors"


def parse_layer_spec(s: str, num_hidden: int) -> list[int]:
    if s == "all":
        return list(range(num_hidden + 1))
    return sorted(set(int(x) for x in s.split(",") if x.strip()))


class HiddenCapture:
    """Captures specific layer hidden states via forward hooks.

    Each hook fires during the forward pass and immediately moves the
    layer's output to CPU, so only one layer's hidden state is ever
    resident on GPU at a time.
    """
    def __init__(self):
        self.hiddens: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            self.hiddens[layer_idx] = h.detach().cpu()
        return hook

    def register(self, model, layer_indices: list[int]):
        self.hiddens.clear()
        self._handles.clear()
        for idx in layer_indices:
            if idx == 0:
                continue
            handle = model.model.layers[idx - 1].register_forward_hook(
                self._make_hook(idx)
            )
            self._handles.append(handle)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layers", default="all",
                    help="Comma-separated layer indices or 'all' (default: all)")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_memory", default=None,
                    help='e.g. \'{"0": "28GiB", "cpu": "80GiB"}\'')
    ap.add_argument("--quantize", default=None, choices=["8bit", "4bit", None],
                    help="Quantize model with bitsandbytes (8bit or 4bit)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing cache files")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] max_seq_len={args.max_seq_len}  layers={args.layers}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    if args.quantize:
        from transformers import BitsAndBytesConfig
        bnb_kwargs = dict(
            load_in_4bit=args.quantize == "4bit",
            load_in_8bit=args.quantize == "8bit",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
    if args.max_memory:
        raw = json.loads(args.max_memory)
        model_kwargs["max_memory"] = {
            int(k) if str(k).lstrip("-").isdigit() else k: v
            for k, v in raw.items()
        }

    print("[precompute] Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if hasattr(model, "lm_head"):
        model.lm_head = None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = parse_layer_spec(args.layers, num_layers)

    n = len(layer_indices)
    preview = layer_indices[:4]
    if n > 4:
        preview.append(layer_indices[-1])
    print(f"[precompute] Capturing {n} layers: {preview}")

    device = next(iter(model.parameters())).device

    has_embed = 0 in layer_indices
    non_embed = [l for l in layer_indices if l > 0]

    capture = HiddenCapture()
    capture.register(model, non_embed)

    skipped = written = chunks_seen = 0
    t0 = time.time()

    with open(args.data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_examples is not None and i >= args.max_examples:
                break
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue

            tokens = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]

            for j in range(0, len(tokens), args.max_seq_len):
                chunk = tokens[j:j + args.max_seq_len]
                chunks_seen += 1

                ids_cpu = torch.tensor(chunk, dtype=torch.long)
                h = hash_chunk(ids_cpu)
                p = cache_path(output_dir, h)

                if p.exists() and not args.force:
                    skipped += 1
                    continue

                input_ids = ids_cpu.unsqueeze(0).to(device)
                attn_mask = torch.ones_like(input_ids)

                with torch.inference_mode():
                    capture.hiddens.clear()
                    _ = model.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
                    if has_embed and hasattr(model.model, "embed_tokens"):
                        capture.hiddens[0] = (
                            model.model.embed_tokens(input_ids).detach().cpu()
                        )

                shard = {}
                for li in layer_indices:
                    t = capture.hiddens.get(li)
                    if t is not None:
                        shard[f"hidden_layer_{li}"] = t[0].to(dtype).cpu().contiguous()
                shard["input_ids"] = ids_cpu.contiguous()

                p.parent.mkdir(parents=True, exist_ok=True)
                save_file(shard, str(p))
                written += 1

                if (written + skipped) % 50 == 0:
                    rate = (written + skipped) / (time.time() - t0)
                    print(f"[{written+skipped:5d}]  written={written}  "
                          f"{rate:.1f} chunk/s")

    capture.remove()

    manifest = {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layers": layer_indices,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"  manifest -> {manifest_path}")

    elapsed = time.time() - t0
    print(f"\nDone \u2014 Written: {written}  Skipped: {skipped}  "
          f"Total chunks: {chunks_seen}  "
          f"{elapsed:.1f}s  {(written+skipped)/elapsed:.1f} chunk/s")


if __name__ == "__main__":
    main()
