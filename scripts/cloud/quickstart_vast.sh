#!/bin/bash
# =============================================================================
# One-shot: Find cheapest Vast.ai machine, create instance, and start training
# =============================================================================
# This script does everything:
#   1. Finds cheapest machine with enough RAM + VRAM
#   2. Creates the instance
#   3. SSHs in and runs the full pipeline
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   export VAST_API_KEY=your_key  (or set in ~/.vast_api_key)
#
# Usage:
#   bash scripts/cloud/quickstart_vast.sh
#   WANDB_API_KEY=xxx HF_TOKEN=xxx bash scripts/cloud/quickstart_vast.sh
# =============================================================================

set -e

# Configurable repo URL — override via env var for forks
REPO_URL="${OPEN_DLLM_REPO:-https://github.com/johndpope/Open-dLLM.git}"

VAST_API_KEY="${VAST_API_KEY:-$(cat ~/.vast_api_key 2>/dev/null || vastai api-key 2>/dev/null)}"
if [ -z "$VAST_API_KEY" ]; then
    echo "ERROR: Set VAST_API_KEY or run: vastai set api-key YOUR_KEY"
    exit 1
fi

# MAX_BID: hard $/hr cap. Instance is INTERRUPTED (not silently upgraded) if host raises above this.
# Instance 37044404 was charged $2.773/hr because --price was missing; Vast.ai auto-upgraded to on-demand.
MAX_BID="${VAST_MAX_BID:-1.0}"

echo "=============================================="
echo "  Open-dLLM Vast.ai Quickstart"
echo "=============================================="

# Step 1: Find cheapest offer
echo "[1/3] Finding cheapest 2×A100 (or better) with 180GB+ RAM..."

BEST=$(curl -s -H "Authorization: Bearer $VAST_API_KEY" \
    "https://cloud.vast.ai/api/v0/bundles/" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offers = data.get('offers', [])
results = []
for o in offers:
    ram = o.get('cpu_ram', 0) / 1024
    gpus = o.get('num_gpus', 0)
    gpu_name = o.get('gpu_name', '')
    gpu_vram = o.get('gpu_ram', 0) / 1024
    total_vram = gpu_vram * gpus
    price = o.get('dph_total', 999)
    reli = o.get('reliability', 0) or 0
    geoloc = o.get('geolocation', '')

    if ram < 180 or total_vram < 40 or gpus < 1 or price > 5.0 or reli < 0.90:
        continue
    # Skip V100 (no bf16)
    if 'V100' in gpu_name.upper():
        continue

    results.append({
        'id': o.get('id'),
        'gpu': gpu_name,
        'gpus': gpus,
        'ram_gb': round(ram, 1),
        'total_vram': round(total_vram, 1),
        'price_hr': round(price, 3),
        'location': geoloc,
        'disk_space': o.get('disk_space', 0),
    })

results.sort(key=lambda x: x['price_hr'])
if not results:
    print('NONE')
    sys.exit(0)

best = results[0]
print(f'{best[\"id\"]}|{best[\"gpu\"]}|{best[\"gpus\"]}|{best[\"ram_gb\"]}|{best[\"total_vram\"]}|{best[\"price_hr\"]}|{best[\"location\"]}|{best[\"disk_space\"]}')
for r in results[:5]:
    print(f'  {r[\"gpu\"]} x{r[\"gpus\"]} ({r[\"total_vram\"]}GB VRAM, {r[\"ram_gb\"]}GB RAM) @ \${r[\"price_hr\"]}/hr — {r[\"location\"]}', file=sys.stderr)
")

if [ "$BEST" = "NONE" ] || [ -z "$BEST" ]; then
    echo "No suitable offers found. Try again later or relax constraints."
    exit 1
fi

OFFER_ID=$(echo "$BEST" | cut -d'|' -f1)
GPU_NAME=$(echo "$BEST" | cut -d'|' -f2)
GPU_COUNT=$(echo "$BEST" | cut -d'|' -f3)
RAM_GB=$(echo "$BEST" | cut -d'|' -f4)
VRAM_GB=$(echo "$BEST" | cut -d'|' -f5)
PRICE=$(echo "$BEST" | cut -d'|' -f6)
LOCATION=$(echo "$BEST" | cut -d'|' -f7)

echo "  Selected: ${GPU_NAME} x${GPU_COUNT} (${VRAM_GB}GB VRAM, ${RAM_GB}GB RAM)"
echo "  Offer price : \$${PRICE}/hr — ${LOCATION}"
echo "  Max bid cap : \$${MAX_BID}/hr  (set VAST_MAX_BID=x.xx to override)"
echo ""

if python3 -c "exit(0 if float('${PRICE}') <= float('${MAX_BID}') else 1)" 2>/dev/null; then
    echo "  Offer is within max bid. Proceeding."
else
    echo "  WARNING: cheapest offer (\$${PRICE}/hr) exceeds max bid (\$${MAX_BID}/hr)."
    echo "  Set VAST_MAX_BID=<higher_value> or wait for cheaper offers. Exiting."
    exit 1
fi
echo ""

# Step 2: Create instance
echo "[2/3] Creating instance (offer $OFFER_ID, capped at \$${MAX_BID}/hr)..."

ENV_STRING="-e DS_SKIP_CUDA_CHECK=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "$WANDB_API_KEY" ] && ENV_STRING="$ENV_STRING -e WANDB_API_KEY=$WANDB_API_KEY"
[ -n "$HF_TOKEN" ] && ENV_STRING="$ENV_STRING -e HF_TOKEN=$HF_TOKEN"
[ -n "$AWS_ACCESS_KEY_ID" ] && ENV_STRING="$ENV_STRING -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}"

ONSTART_SCRIPT=$(cat <<INNEREOF
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== Open-dLLM Cloud Setup ==="

# Persist env vars to bashrc so interactive shells have them
cat >> /root/.bashrc <<ENVEOF
export HF_TOKEN="\${HF_TOKEN}"
export AWS_ACCESS_KEY_ID="\${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="\${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="\${AWS_DEFAULT_REGION:-us-east-1}"
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVEOF

# Install system deps
apt-get update -qq
apt-get install -y -qq git git-lfs python3-venv python3-pip curl awscli htop nvtop tmux
git lfs install

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"

# Clone repo
cd /workspace
git clone "${REPO_URL}"
cd Open-dLLM

# Install deps
uv sync --extra dev --extra deepspeed

# Download model weights
echo "=== Downloading Qwen3.6-27B (~54GB)... ==="
mkdir -p /data/models
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('Qwen/Qwen3.6-27B', local_dir='/data/models/Qwen3.6-27B', token=os.environ.get('HF_TOKEN'))
print('Model downloaded')
"

# Pull anchor latents from S3
echo "=== Pulling anchors from S3 (~27GB)... ==="
mkdir -p /data/anchors/qwen3.6-27b
aws s3 sync s3://qwen3-6/anchors/qwen3.6-27b/ /data/anchors/qwen3.6-27b/ --no-progress

# Generate smoke data
echo "=== Generating training data... ==="
bash /workspace/Open-dLLM/scripts/cloud/prepare_data.sh /data/training

echo ""
echo "=== Setup complete! ==="
echo "Run: bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh"
INNEREOF
)

INSTANCE_JSON=$(vastai create instance "$OFFER_ID" \
    --image "nvidia/cuda:13.0.0-devel-ubuntu22.04" \
    --disk 300 \
    --ssh \
    --direct \
    --label "open-dllm-27b" \
    --price "$MAX_BID" \
    --onstart-cmd "$ONSTART_SCRIPT" \
    --env "$ENV_STRING" \
    --raw)

echo "$INSTANCE_JSON"
INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")

echo ""
echo "[3/3] Instance created: ID=$INSTANCE_ID"
echo "  Verifying actual billing rate (give it a moment)..."
sleep 5
vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    actual = d.get('dph_total', 'unknown')
    status = d.get('actual_status', 'unknown')
    print(f'  Actual rate : \${actual}/hr  (status: {status})')
    if isinstance(actual, (int,float)) and actual > float('${MAX_BID}') * 1.05:
        print(f'  WARNING: actual rate exceeds max bid! Destroy immediately:')
        print(f'    vastai destroy instance ${INSTANCE_ID}')
    else:
        print(f'  OK: rate within expected range.')
except Exception as e:
    print(f'  Could not verify: {e}  — check: vastai show instances')
" 2>/dev/null || echo "  (run: vastai show instances  to verify rate)"

echo ""
echo "  Billing dashboard : https://cloud.vast.ai/billing/"
echo "  SSH URL           : vastai ssh-url $INSTANCE_ID"
echo "  Model download    : ~54GB from HuggingFace + ~27GB anchors from S3 (~\$2.74 one-time)"
echo "  Estimated GPU time for 10 steps: ~0.5hr @ \$${PRICE}/hr = \$$(python3 -c "print(f'{float(\"${PRICE}\")*0.5:.2f}')")"
