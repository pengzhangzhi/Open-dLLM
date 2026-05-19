# Cloud Training Guide

## Overview

27B Repr-Align training doesn't fit on local hardware (96GB RAM; needs ~170GB peak). This guide covers renting GPU instances on Vast.ai with enough RAM to run training.

## Quick Start

```bash
# 1. Install Vast.ai CLI
pip install vastai
vastai set api-key YOUR_KEY

# 2. Launch (finds cheapest machine, creates instance, installs everything)
WANDB_API_KEY=xxx bash scripts/cloud/quickstart_vast.sh

# 3. SSH in and train
bash /workspace/Open-dLLM/scripts/cloud/train_27b_vast.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `quickstart_vast.sh` | One-shot: find cheapest machine, create instance, install deps, download model |
| `launch_vast.sh` | Search/filter offers by GPU type, RAM, region; create instance |
| `train_27b_vast.sh` | Run inside instance: verify model, precompute anchors, launch training |
| `prepare_data.sh` | Generate synthetic smoke data if no real dataset available |

## API Keys & Credentials

You need three API keys before starting. Set them as environment variables:

```bash
# 1. Vast.ai — for renting GPU instances
#    Get your key at: https://cloud.vast.ai/account/settings
pip install vastai
vastai set api-key YOUR_VAST_KEY

# 2. HuggingFace — for downloading model weights (Qwen3.6-27B is gated)
#    Get your token at: https://huggingface.co/settings/tokens
#    You may need to request access at: https://huggingface.co/Qwen/Qwen3.6-27B
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx

# 3. Weights & Biases — for training logging
#    Get your key at: https://wandb.ai/authorize
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

# 4. AWS S3 — for sharing anchor latents between instances
#    (see S3 Setup below)
```

## S3 Setup

Anchor latents (27GB) are stored in S3 so cloud instances can download them instead of recomputing (~20 min each time). A 7-day auto-delete lifecycle rule keeps costs minimal.

### Create the S3 bucket

1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com/s3/home)
2. Click **Create bucket**
3. Name: `qwen3-6` (or your preferred name)
4. Region: `us-east-1`
5. Leave defaults, click **Create bucket**

### Create IAM credentials

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/home#/users)
2. Click **Add users** → name it `vastai-s3`
3. Select **Access key - Programmatic access**
4. Attach policy → **Create policy**:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
               "Resource": [
                   "arn:aws:s3:::qwen3-6",
                   "arn:aws:s3:::qwen3-6/*"
               ]
           }
       ]
   }
   ```
5. Save the **Access key ID** and **Secret access key** (you only see the secret once!)

### Set the 7-day auto-delete lifecycle

```bash
export AWS_ACCESS_KEY_ID=YOUR_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY

aws s3api put-bucket-lifecycle-configuration \
    --bucket qwen3-6 \
    --region us-east-1 \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "AutoDeleteAfter7Days",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": 7}
        }]
    }'
```

### Upload anchor latents

```bash
# From your local machine (after precomputing anchors)
aws s3 sync /home/johndpope/ds_offload/anchors/qwen3.6-27b/ \
    s3://qwen3-6/anchors/qwen3.6-27b/ \
    --region us-east-1 --no-progress
```

### Download on cloud instance

```bash
# Inside the Vast.ai instance (skips ~20 min anchor precompute)
aws s3 sync s3://qwen3-6/anchors/qwen3.6-27b/ /data/anchors/qwen3.6-27b/ \
    --region us-east-1
```

### Bucket layout

```
s3://qwen3-6/
  ├── anchors/
  │   └── qwen3.6-27b/          # 27GB, 1085 safetensors files
  │       ├── 00/ 01/ ... ff/    # 252 shard dirs (2-char hash prefix)
  │       └── *.safetensors      # cached teacher hidden states (auto-delete after 7 days)
  └── training/                  # training data (TODO)
```

### Cost estimate

S3 Standard storage: ~$0.023/GB/month. 27GB of anchors = **~$0.62/month** (but auto-deletes after 7 days, so ~$0.14 per upload cycle).

## Hardware Requirements

### Minimum for 27B Repr-Align

| Component | Requirement | Why |
|-----------|-------------|-----|
| RAM | **180 GB+** | ZeRO-3 init peak ~170GB before offload |
| VRAM | 24 GB+ (total) | Active layer forward/backward |
| Disk | 200 GB+ NVMe | Model (54GB) + anchors (27GB) + DS swap |
| CUDA | 12.x+ | PyTorch 2.12+ requirement |

### Budget Machine Options (Vast.ai, May 2026)

| Machine | RAM | VRAM | Price/hr | Notes |
|---------|-----|------|----------|-------|
| 2× A100 PCIE | 189 GB | 80 GB | $0.27 | **Best value** |
| 1× RTX PRO 6000 WS | 249 GB | 96 GB | $0.45 | Plenty of RAM |
| 2× RTX 5090 | 189 GB | 64 GB | $0.67 | Good if available |
| 1× H100 SXM | 189 GB | 80 GB | $1.47 | Fastest, pricier |

> Avoid Tesla V100 — no bf16 support (requires fp32, doubles memory).

### Local vs Cloud Cost Comparison

| Approach | Upfront | Per-run (10 steps) | Notes |
|----------|---------|-------------------|-------|
| RAM upgrade (96→192GB) | ~$2,000 AUD | $0 | 30 min/10 steps, reusable |
| 2× A100 on Vast.ai | $0 | ~$0.14 | 5 min/10 steps |
| 8× H100 (Lambda) | $0 | ~$25-50 | Overkill for 10 steps |

## Training Flow on Cloud Instance

```
1. quickstart_vast.sh
   ├── Query Vast.ai API for cheapest 180GB+ RAM machine
   ├── Create instance with CUDA 13.0 image
   └── On-start: clone repo, install deps, download Qwen3.6-27B (~54GB)

2. train_27b_vast.sh (SSH into instance)
   ├── Verify model weights (15 safetensors shards)
   ├── Download/generate training data
   ├── Precompute anchor cache (4 layers, ~20 min for 1000 examples)
   │   OR download from S3: aws s3 sync s3://qwen3-6/anchors/qwen3.6-27b/ /data/anchors/
   ├── Auto-tune offload strategy based on available RAM:
   │   ├── 180GB+ RAM → CPU param + CPU optimizer offload
   │   ├── 120-180GB → CPU param + NVMe optimizer offload
   │   └── <120GB → NVMe param + NVMe optimizer offload
   └── Launch: torchrun tasks/train_torch.py /tmp/train_cloud.yaml
```

## DeepSpeed NVMe Gotchas

These are documented for anyone debugging cloud instances:

- **`DS_SKIP_CUDA_CHECK=1`**: Required when system CUDA != PyTorch CUDA. Without it, async_io won't compile.
- **Pre-build async_io**: Run once before training:
  ```bash
  DS_SKIP_CUDA_CHECK=1 python -c "import deepspeed.ops.op_builder as b; b.AsyncIOBuilder().load()"
  ```
- **`buffer_size`**: Must exceed largest combined partition (set to 2B elements in code).
- **`torch.empty("nvme")`**: Patched in `veomni/distributed/deepspeed_init.py` to allocate on `"cpu"` — PyTorch doesn't recognize nvme as a device.

## Monitoring

```bash
# Check instance status
vastai show instances

# Check training logs
tmux attach  # if running in tmux
tail -f /data/checkpoints/qwen3.6-27b-repr-align/logs/train.log

# Check S3 bucket
aws s3 ls s3://qwen3-6/ --recursive --region us-east-1 | head -20

# Wandb dashboard
# https://wandb.ai/<your-team>/open-dllm-27b
```

## Teardown

```bash
# Stop instance (keeps disk, stops billing for GPU)
vastai stop instance <id>

# Delete instance (destroys disk)
vastai destroy instance <id>

# S3 auto-deletes after 7 days (lifecycle rule)
```
