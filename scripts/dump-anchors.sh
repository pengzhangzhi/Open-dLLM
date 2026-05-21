#!/bin/bash
set -euo pipefail

# Precompute anchor hidden states for Repr-Align training (4 layers, 160k ctx).
# Output to 12TB drive. Run in background: bash scripts/dump-anchors.sh &

OUTDIR=/run/media/johndpope/12TB/open_dllm/anchors/qwen3.6-27b-160k
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/precompute_anchor.py \
  --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
  --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
  --output_dir "$OUTDIR" \
  --layers "16,32,48,64" \
  --max_seq_len 160000 \
  --seed 42 \
  --max_memory '{"0": "30GiB", "cpu": "80GiB"}'
