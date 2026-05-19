#!/bin/bash
# =============================================================================
# 27B Repr-Align Training on Vast.ai
# =============================================================================
# Run this INSIDE the Vast.ai instance after launch_vast.sh has provisioned it.
#
# Usage:
#   bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh
#   ANCHORS_ONLY=1 bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh  # precompute only
# =============================================================================

set -e

# Config
REPO_DIR="/workspace/Open-dLLM"
MODEL_DIR="/data/models/Qwen3.6-27B"
DATA_FILE="/data/training/data_smoke_1000.jsonl"
ANCHOR_DIR="/data/anchors/qwen3.6-27b"
CHECKPOINT_DIR="/data/checkpoints/qwen3.6-27b-repr-align"
CONFIG="configs/pretrain/qwen3_6_27b_full_repr_align_ds.yaml"
LAYERS="16,32,48,64"
MAX_EXAMPLES=1000
MAX_SEQ_LEN=2048

export PATH="$HOME/.local/bin:$PATH"
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_DIR"

echo "=============================================="
echo "  Open-dLLM 27B Repr-Align Training"
echo "=============================================="
echo "  Model:     $MODEL_DIR"
echo "  Data:      $DATA_FILE"
echo "  Anchors:   $ANCHOR_DIR"
echo "  Layers:    $LAYERS"
echo "  GPU count: $(nvidia-smi -L | wc -l)"
echo "  RAM:       $(free -g | grep Mem | awk '{print $2}')GB"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: Verify model weights
# =============================================================================
echo "[1/4] Verifying model weights..."
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download with:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', local_dir='$MODEL_DIR')\""
    exit 1
fi
echo "  OK — $(ls $MODEL_DIR/*.safetensors 2>/dev/null | wc -l) shards found"
echo ""

# =============================================================================
# Step 2: Download training data if not present
# =============================================================================
echo "[2/4] Checking training data..."
if [ ! -f "$DATA_FILE" ]; then
    echo "  No training data found. Generating synthetic smoke data..."
    bash "$REPO_DIR/scripts/cloud/prepare_data.sh" /data/training
fi
echo "  OK — $(wc -l < $DATA_FILE) examples"
echo ""

# =============================================================================
# Step 3: Precompute anchor cache
# =============================================================================
echo "[3/4] Precomputing anchor cache..."
ANCHER_COUNT=$(find "$ANCHOR_DIR" -name "*.safetensors" 2>/dev/null | wc -l)

if [ "$ANCHER_COUNT" -gt 0 ]; then
    echo "  OK — $ANCHER_COUNT anchor shards already cached"
else
    echo "  Computing anchors (this takes ~20 min for 1000 examples)..."
    python scripts/precompute_anchor.py \
        --model_path "$MODEL_DIR" \
        --data_path "$DATA_FILE" \
        --output_dir "$ANCHOR_DIR" \
        --layers "$LAYERS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --max_examples "$MAX_EXAMPLES" \
        --device_map auto
    echo "  Done — $(find $ANCHOR_DIR -name '*.safetensors' | wc -l) anchor shards"
fi
echo ""

if [ "${ANCHORS_ONLY:-0}" = "1" ]; then
    echo "ANCHORS_ONLY=1 — skipping training."
    exit 0
fi

# =============================================================================
# Step 4: Generate cloud-specific config and launch training
# =============================================================================
echo "[4/4] Launching training..."

NUM_GPUS=$(nvidia-smi -L | wc -l)
TOTAL_RAM_GB=$(free -g | grep Mem | awk '{print $2}')

# Build config adapted to this machine
python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

# Point to local paths
cfg['model']['model_path'] = '$MODEL_DIR'
cfg['data']['train_path'] = '$DATA_FILE'
cfg['train']['anchor_cache_dir'] = '$ANCHOR_DIR'
cfg['train']['output_dir'] = '$CHECKPOINT_DIR'
cfg['train']['ds_nvme_path'] = '/data/ds_offload'

# Adjust parallelism to available hardware
cfg['train']['global_batch_size'] = $NUM_GPUS
cfg['train']['ulysses_parallel_size'] = 1

# If RAM is generous, keep params on CPU; otherwise use NVMe
ram = $TOTAL_RAM_GB
if ram >= 180:
    cfg['train']['ds_offload_param'] = 'cpu'
    cfg['train']['ds_offload_optimizer'] = 'cpu'
    print(f'  RAM={ram}GB: using CPU offload for params+optimizer')
elif ram >= 120:
    cfg['train']['ds_offload_param'] = 'cpu'
    cfg['train']['ds_offload_optimizer'] = 'nvme'
    print(f'  RAM={ram}GB: CPU params + NVMe optimizer')
else:
    cfg['train']['ds_offload_param'] = 'nvme'
    cfg['train']['ds_offload_optimizer'] = 'nvme'
    print(f'  RAM={ram}GB: NVMe offload for params+optimizer')

with open('/tmp/train_cloud.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('  Config written to /tmp/train_cloud.yaml')
"

echo ""
echo "Launching with $NUM_GPUS GPUs..."
echo ""

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    .venv/bin/torchrun --nproc_per_node="$NUM_GPUS" \
    tasks/train_torch.py /tmp/train_cloud.yaml

echo ""
echo "=============================================="
echo "  Training complete!"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Wandb: https://wandb.ai"
echo "=============================================="
