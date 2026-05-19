#!/bin/bash
# =============================================================================
# Push the latest (or specified) HF checkpoint to S3 and run sanity inference
# =============================================================================
# Usage:
#   bash scripts/cloud/push_ckpt_s3.sh                        # latest checkpoint
#   bash scripts/cloud/push_ckpt_s3.sh global_step_50         # specific step
#   SKIP_INFER=1 bash scripts/cloud/push_ckpt_s3.sh           # skip inference check
# =============================================================================

set -e

CKPT_BASE="${CKPT_BASE:-/data/checkpoints/qwen3.6-27b-repr-align}"
S3_BUCKET="${S3_BUCKET:-s3://qwen3-6}"
S3_PREFIX="${S3_PREFIX:-checkpoints/qwen3.6-27b-repr-align}"
REPO_DIR="${REPO_DIR:-/workspace/Open-dLLM}"
INFER_STEPS="${INFER_STEPS:-32}"

# Resolve which checkpoint to push
if [ -n "$1" ]; then
    STEP_DIR="$1"
else
    # Find the highest-numbered global_step_* directory
    STEP_DIR=$(ls -d "$CKPT_BASE"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 | xargs basename)
fi

if [ -z "$STEP_DIR" ]; then
    echo "ERROR: No checkpoint found in $CKPT_BASE"
    exit 1
fi

HF_CKPT="$CKPT_BASE/$STEP_DIR/hf_ckpt"

if [ ! -d "$HF_CKPT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CKPT"
    echo "Available checkpoints:"
    ls "$CKPT_BASE" 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "=============================================="
echo "  Checkpoint → S3"
echo "=============================================="
echo "  Source:  $HF_CKPT"
echo "  S3 dest: $S3_BUCKET/$S3_PREFIX/$STEP_DIR/hf_ckpt/"
echo "  Size:    $(du -sh $HF_CKPT | cut -f1)"
echo "=============================================="

# Run inference sanity check before pushing
if [ "${SKIP_INFER:-0}" != "1" ]; then
    echo ""
    echo "[1/2] Running sanity inference check..."
    export PATH="$HOME/.local/bin:$PATH"
    cd "$REPO_DIR"
    .venv/bin/python scripts/cloud/sanity_infer.py \
        --ckpt "$HF_CKPT" \
        --steps "$INFER_STEPS"
    echo ""
fi

# Push to S3
echo "[2/2] Syncing to S3..."
aws s3 sync "$HF_CKPT/" "$S3_BUCKET/$S3_PREFIX/$STEP_DIR/hf_ckpt/" \
    --no-progress \
    --storage-class STANDARD_IA

echo ""
echo "Done! Checkpoint available at:"
echo "  $S3_BUCKET/$S3_PREFIX/$STEP_DIR/hf_ckpt/"
echo ""
echo "To pull on another machine:"
echo "  aws s3 sync $S3_BUCKET/$S3_PREFIX/$STEP_DIR/hf_ckpt/ ./hf_ckpt/"
