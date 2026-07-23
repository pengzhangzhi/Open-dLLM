#!/bin/bash
set -euo pipefail

# Smoke test: verify precomputed anchors work with CachedTeacher forward pass.
# Must be run after at least a few anchor files exist in the cache dir.

CACHE_DIR=/run/media/johndpope/12TB/open_dllm/anchors/qwen3.6-27b-160k
MODEL_PATH=/home/johndpope/ds_offload/models/Qwen3.6-27B
DATA_PATH=/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl

# Pick 3 random lines from the data and test them
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import json, random, sys
from pathlib import Path

# Create manifest if it doesn't exist yet
cache_dir = Path('$CACHE_DIR')
manifest_path = cache_dir / 'manifest.json'
if not manifest_path.exists():
    manifest = {
        'num_hidden_layers': 64,
        'hidden_size': 5120,
        'layers': [16, 32, 48, 64],
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f'[smoke] Created manifest at {manifest_path}')
else:
    manifest = json.loads(open(manifest_path).read())
    print(f'[smoke] Loaded existing manifest: {manifest[\"layers\"]}')

sys.path.insert(0, '.')

import torch
from transformers import AutoTokenizer
from veomni.models.cached_teacher import CachedTeacher

tok = AutoTokenizer.from_pretrained('$MODEL_PATH', trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

teacher = CachedTeacher(
    cache_dir=str(cache_dir),
    num_hidden_layers=64,
    hidden_size=5120,
)

# Read the data file and pick lines that match existing cache
data_path = '$DATA_PATH'
with open(data_path) as f:
    lines = [json.loads(l)['text'] for l in f]

# Test the first few texts (they should be cached since dump starts at beginning)
n_tested = 0
for i, text in enumerate(lines):
    if n_tested >= 3:
        break

    tokens = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]

    # Check if this chunk is cached
    import hashlib
    ids_cpu = torch.tensor(tokens, dtype=torch.long)
    h = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()[:16]
    p = cache_dir / h[:2] / f'{h}.safetensors'
    if not p.exists():
        continue

    input_ids = ids_cpu.unsqueeze(0)
    position_ids = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)

    out = teacher(input_ids=input_ids, position_ids=position_ids)
    hs = out.hidden_states

    print(f'\\n[smoke] Example {i}: {len(tokens):5d} tokens')
    for li in manifest['layers']:
        t = hs[li]
        print(f'  Layer {li:2d}: {list(t.shape)}  '
              f'mean={t.mean().item():.4f}  std={t.std().item():.4f}  '
              f'device={t.device}')
    n_tested += 1

if n_tested == 0:
    print('\\n[smoke] No cached files found yet. Run dump-anchors.sh first.')
    sys.exit(1)

print(f'\\n[smoke] PASSED — {n_tested} examples verified')
" 2>&1