#!/bin/bash
# =============================================================================
# Launch Vast.ai instance for Open-dLLM 27B Repr-Align Training
# =============================================================================
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
#
# Usage:
#   bash scripts/cloud/launch_vast.sh
#   bash scripts/cloud/launch_vast.sh --gpu A100 --region europe
# =============================================================================

set -e

# Configurable repo URL — override via env var for forks
REPO_URL="${OPEN_DLLM_REPO:-https://github.com/johndpope/Open-dLLM.git}"

# Defaults
GPU_FILTER=""
REGION_FILTER=""
NUM_GPUS=2
MIN_RAM_GB=180
MIN_VRAM_GB=24
DISK_GB=300
IMAGE="nvidia/cuda:13.0.0-devel-ubuntu22.04"
LABEL="open-dllm-27b-repr-align"
WANDB_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_FILTER="$2"; shift 2 ;;
        --region) REGION_FILTER="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --ram) MIN_RAM_GB="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Open-dLLM Vast.ai Instance Launcher"
echo "=============================================="
echo "  Target: ${NUM_GPUS}x GPU, ${MIN_RAM_GB}GB+ RAM, ${MIN_VRAM_GB}GB+ VRAM"
echo "  Disk: ${DISK_GB}GB, Label: ${LABEL}"
echo "=============================================="
echo ""

# Search for offers
QUERY="ram_gb>=${MIN_RAM_GB} num_gpus>=${NUM_GPUS} gpu_ram>=${MIN_VRAM_GB} disk_space>=${DISK_GB} reliability>=0.95 rentable=true"

if [ -n "$GPU_FILTER" ]; then
    QUERY="gpu_name=${GPU_FILTER} ${QUERY}"
fi

echo "Searching: $QUERY"
echo ""

OFFERS_JSON=$(curl -s -H "Authorization: Bearer $(vastai api-key 2>/dev/null || echo $VAST_API_KEY)" \
    "https://cloud.vast.ai/api/v0/bundles/" 2>/dev/null)

if [ -z "$OFFERS_JSON" ]; then
    echo "ERROR: Failed to query Vast.ai API. Check your API key."
    exit 1
fi

# Parse and rank offers
RESULTS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offers = data.get('offers', [])
results = []
for o in offers:
    ram = o.get('cpu_ram', 0) / 1024
    gpus = o.get('num_gpus', 0)
    gpu_name = o.get('gpu_name', 'unknown')
    gpu_vram = o.get('gpu_ram', 0) / 1024
    total_vram = gpu_vram * gpus
    price = o.get('dph_total', 999)
    reli = o.get('reliability', 0) or 0
    geoloc = o.get('geolocation', '')
    offer_id = o.get('id', '')
    disk = o.get('disk_space', 0)
    host_id = o.get('host_id', '')

    if ram < $MIN_RAM_GB or total_vram < $MIN_VRAM_GB or gpus < $NUM_GPUS:
        continue

    gpu_match = '${GPU_FILTER}'.lower()
    if gpu_match and gpu_match not in gpu_name.lower():
        continue

    region_match = '${REGION_FILTER}'.lower()
    if region_match and region_match not in geoloc.lower():
        continue

    results.append({
        'id': offer_id,
        'host_id': host_id,
        'gpu': gpu_name,
        'gpus': gpus,
        'ram_gb': round(ram, 1),
        'total_vram': round(total_vram, 1),
        'price_hr': round(price, 3),
        'reliability': round(reli, 2),
        'location': geoloc,
        'disk_gb': round(disk, 0),
    })

results.sort(key=lambda x: x['price_hr'])
for r in results[:10]:
    print(json.dumps(r))
" 2>/dev/null)

if [ -z "$RESULTS" ]; then
    echo "No suitable offers found. Try relaxing constraints:"
    echo "  --ram 128   (tighter RAM, relies on NVMe offload)"
    echo "  --gpu ''    (any GPU type)"
    echo "  --gpus 1    (single GPU)"
    exit 1
fi

echo "Top 10 cheapest offers:"
echo ""
printf "%-5s %-20s %4s %6s %6s %8s %s\n" "Rank" "GPU" "GPUs" "RAM" "VRAM" "Price/hr" "Location"
echo "--------------------------------------------------------------------------------"

OFFER_IDS=()
i=0
while IFS= read -r line; do
    id=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    gpu=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['gpu'])")
    gpus=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['gpus'])")
    ram=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['ram_gb'])")
    vram=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_vram'])")
    price=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['price_hr'])")
    loc=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['location'])")

    OFFER_IDS+=("$id")
    printf "%-5d %-20s %4d %5.0fG %5.0fG \$%7.3f %s\n" "$i" "$gpu" "$gpus" "$ram" "$vram" "$price" "$loc"
    i=$((i+1))
done <<< "$RESULTS"

echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run — not creating instance."
    echo "To create: bash $0 (without --dry-run)"
    exit 0
fi

# Pick cheapest by default
CHOSEN=${OFFER_IDS[0]}
CHOSEN_PRICE=$(echo "$RESULTS" | head -1 | python3 -c "import json,sys; print(json.load(sys.stdin)['price_hr'])")

echo "Creating instance from offer $CHOSEN (\$${CHOSEN_PRICE}/hr)..."

# Build env vars
ENV_ARGS="-e DS_SKIP_CUDA_CHECK=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "$WANDB_KEY" ] && ENV_ARGS="$ENV_ARGS -e WANDB_API_KEY=$WANDB_KEY"
[ -n "$HF_TOKEN" ] && ENV_ARGS="$ENV_ARGS -e HF_TOKEN=$HF_TOKEN"
[ -n "$AWS_ACCESS_KEY_ID" ] && ENV_ARGS="$ENV_ARGS -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}"

# Onstart: install deps, clone repo, download model + anchors, generate data
ONSTART=$(cat <<ONSTART_EOF
bash -c '
set -e
export HF_TOKEN="${HF_TOKEN}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persist to bashrc
cat >> /root/.bashrc <<ENVEOF
export HF_TOKEN="${HF_TOKEN}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVEOF

apt-get update -qq && apt-get install -y -qq git git-lfs python3-venv python3-pip curl awscli
git lfs install

# Clone repo
cd /workspace
git clone "${REPO_URL}"
cd Open-dLLM

# Install uv + deps
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
uv sync --extra dev --extra deepspeed

# Download model weights (~54GB)
echo "=== Downloading Qwen3.6-27B... ==="
mkdir -p /data/models
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(\"Qwen/Qwen3.6-27B\", local_dir=\"/data/models/Qwen3.6-27B\", token=os.environ.get(\"HF_TOKEN\"))
print(\"Model downloaded\")
"

# Pull anchor latents from S3
echo "=== Pulling anchors from S3... ==="
mkdir -p /data/anchors/qwen3.6-27b
aws s3 sync s3://qwen3-6/anchors/qwen3.6-27b/ /data/anchors/qwen3.6-27b/ --no-progress

# Generate smoke data
echo "=== Generating training data... ==="
bash /workspace/Open-dLLM/scripts/cloud/prepare_data.sh /data/training

echo "=== Instance ready. Run: bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh ==="
'
ONSTART_EOF
)

vastai create instance "$CHOSEN" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --label "$LABEL" \
    --onstart-cmd "$ONSTART" \
    --env "-e DS_SKIP_CUDA_CHECK=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENV_ARGS" \
    --raw

echo ""
echo "Instance created! Check status:"
echo "  vastai show instances"
echo ""
echo "SSH in:"
echo "  vastai ssh-url <instance_id>"
echo ""
echo "Then run training:"
echo "  bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh"
