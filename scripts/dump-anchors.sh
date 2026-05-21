#!/bin/bash
set -euo pipefail

# Precompute anchor hidden states for Repr-Align training.
#
# CRITICAL — these MUST match the training config you will run, or every step
# is an anchor cache miss (CachedTeacher hashes chunks by token ids):
#   * MAX_SEQ_LEN  == config data.max_seq_len
#   * OUTDIR       == config train.anchor_cache_dir
#   * MAX_EXAMPLES >= config data.train_size
#   * tokenization is fixed in precompute_anchor.py (add_special_tokens=False + eos)
#
# 4-bit NF4 quantization fits the 27B teacher in ~19 GB on the RTX 5090.
# CUDA_DEVICE_ORDER=FASTEST_FIRST pins device 0 to the 5090 (32 GB), not the
# RTX PRO 4000 (24 GB), regardless of PCI ordering.

MODEL=/home/johndpope/ds_offload/models/Qwen3.6-27B
DATA=/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl

# ---- target (edit to match your training config) ----
OUTDIR=/home/johndpope/ds_offload/anchors/qwen3.6-27b-all64
LAYERS=all          # "all" for full-depth repr-align, or e.g. "16,32,48,64"
MAX_SEQ_LEN=1024    # MUST equal data.max_seq_len
MAX_EXAMPLES=1000   # MUST be >= data.train_size

mkdir -p "$OUTDIR"

# Omit --force to extend an existing cache (skips chunks already dumped);
# add --force to rebuild from scratch.
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=FASTEST_FIRST \
  .venv/bin/python scripts/precompute_anchor.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUTDIR" \
  --layers "$LAYERS" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_examples "$MAX_EXAMPLES" \
  --seed 42 \
  --quantize 4bit
