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

VAST_API_KEY="${VAST_API_KEY:-$(cat ~/.vast_api_key 2>/dev/null || vastai api-key 2>/dev/null)}"
if [ -z "$VAST_API_KEY" ]; then
    echo "ERROR: Set VAST_API_KEY or run: vastai set api-key YOUR_KEY"
    exit 1
fi

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
echo "  Price: \$${PRICE}/hr — ${LOCATION}"
echo ""

# Step 2: Create instance
echo "[2/3] Creating instance (offer $OFFER_ID)..."

ENV_STRING="-e DS_SKIP_CUDA_CHECK=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "$WANDB_API_KEY" ] && ENV_STRING="$ENV_STRING -e WANDB_API_KEY=$WANDB_API_KEY"
[ -n "$HF_TOKEN" ] && ENV_STRING="$ENV_STRING -e HF_TOKEN=$HF_TOKEN"

ONSTART_SCRIPT=$(cat <<'INNEREOF'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== Open-dLLM Cloud Setup ==="

# Install system deps
apt-get update -qq
apt-get install -y -qq git git-lfs python3-venv python3-pip curl htop nvtop tmux
git lfs install

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
cd /workspace
git clone https://github.com/scrya-com/Open-dLLM.git
cd Open-dLLM

# Install deps
uv sync --extra dev

# Pre-build DeepSpeed async_io
DS_SKIP_CUDA_CHECK=1 .venv/bin/python -c "import deepspeed.ops.op_builder as b; b.AsyncIOBuilder().load(); print('async_io OK')"

# Download model weights
echo "Downloading Qwen3.6-27B (~54GB)..."
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B', local_dir='/data/models/Qwen3.6-27B')
print('Model downloaded')
"

echo ""
echo "=== Setup complete! ==="
echo "Run: bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh"
INNEREOF
)

vastai create instance "$OFFER_ID" \
    --image "nvidia/cuda:13.0.0-devel-ubuntu22.04" \
    --disk 300 \
    --ssh \
    --direct \
    --label "open-dllm-27b" \
    --onstart-cmd "$ONSTART_SCRIPT" \
    --env "$ENV_STRING" \
    --raw

echo ""
echo "[3/3] Waiting for instance to provision..."
echo "  Model download (~54GB) will take 10-30 min depending on bandwidth."
echo ""
echo "  Check status:  vastai show instances"
echo "  Get SSH URL:   vastai ssh-url <instance_id>"
echo ""
echo "  Once SSH'd in, run:"
echo "    bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh"
echo ""
echo "  Estimated cost for 10 steps: \$(echo "$PRICE * 0.5" | bc)"
