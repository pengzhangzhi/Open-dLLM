"""Quick scan: how many data lines are fully covered by the existing anchor cache?"""
import hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer

CACHE_DIR = Path("/home/johndpope/ds_offload/anchors/qwen3.6-27b")
DATA_PATH = "/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl"
MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
MAX_SEQ_LEN = 2048
SCAN_LINES = 10000

def hash_chunk(ids):
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16]

def chunk_exists(h):
    return (CACHE_DIR / h[:2] / f"{h}.safetensors").exists()

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

covered = 0
total_examples = 0
total_chunks = 0
missed_examples = 0

with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if i >= SCAN_LINES:
            break
        text = json.loads(line)["text"]
        tokens = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
        chunks = [tokens[j:j+MAX_SEQ_LEN] for j in range(0, len(tokens), MAX_SEQ_LEN)]
        
        all_cached = True
        for chunk in chunks:
            import torch
            ids = torch.tensor(chunk, dtype=torch.long)
            h = hash_chunk(ids)
            total_chunks += 1
            if not chunk_exists(h):
                all_cached = False
        
        total_examples += 1
        if all_cached:
            covered += 1
        else:
            missed_examples += 1
        
        if i % 1000 == 0 and i > 0:
            print(f"[{i}] covered={covered} missed={missed_examples} chunks={total_chunks}", flush=True)

print(f"\n=== Results ===")
print(f"Scanned:     {total_examples} examples, {total_chunks} chunks")
print(f"Fully cached: {covered}")
print(f"Has misses:  {missed_examples}")
