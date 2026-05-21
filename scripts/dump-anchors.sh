#!/bin/bash
set -euo pipefail

# Precompute anchor hidden states for Repr-Align training (4 layers, 160k ctx).
# Uses 8-bit quantization on RTX 5090 to fit the 27B model in ~24 GB.
# Single GPU — no cross-GPU issues, no CPU offload NaN.

OUTDIR=/run/media/johndpope/12TB/open_dllm/anchors/qwen3.6-27b-160k
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/precompute_anchor.py \
  --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
  --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
  --output_dir "$OUTDIR" \
  --layers "16,32,48,64" \
  --max_seq_len 160000 \
  --seed 42 \
  --force \
  --max_examples 100 \
  --quantize 4bit
