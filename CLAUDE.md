# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-dLLM is a diffusion-based large language model framework for training, evaluation, and inference. It converts autoregressive LMs (Qwen, LLaMA, DeepSeek) into discrete diffusion LMs, offering ~4x speedup. The core package is `veomni/`.

## Commands

```bash
# Setup (Python 3.11+)
pip install -e ".[dev]"
# or with uv (preferred for local dev, pins CUDA 12.9 for Blackwell GPUs)
uv sync --extra dev

# Linting & formatting (ruff, line-length=119, target py311)
make style          # auto-fix
make quality        # check only

# Tests
make test           # pytest tests/
pytest tests/path/test_file.py  # single file

# Pre-commit
make commit         # pre-commit run --all-files
```

## Architecture

### Training Entry Points (`tasks/`)
YAML-driven via `veomni/utils/arguments.py` (three dataclass groups: `ModelArguments`, `DataArguments`, `TrainingArguments`):

- **`tasks/train_torch.py`** — standard MDM training. Also handles **Repr-Align** (bidirectional student + frozen causal teacher) when `train.repr_align_wt > 0` and/or `train.enable_masking=true`. Supports FSDP1, DDP, and **DeepSpeed** (`data_parallel_mode: deepspeed`).
- **`tasks/train_ldlm.py`** — **LDLM** training (Perceiver encoder/decoder + DiT head on top of a frozen AR encoder). Manages multi-GPU placement internally via `device_map="auto"` — frozen encoder on GPU 0, trainable components on GPU 1. Always launch with `--nproc_per_node=1`.
- **`tasks/benchmark_ldlm.py`** / **`benchmark_ldlm_35b.py`** — throughput benchmarks for the 27B / 35B-A3B LDLM (encoder deleted, inference-only).
- **`tasks/infer.py`**, **`sample.py`** — generation entry points using `model.diffusion_generate()`.

Configs:
- **Pretraining**: `configs/pretrain/` — plaintext datasets, FSDP1/DDP. Includes `qwen3_6_27b_ldlm.yaml`, `qwen3_6_35b_a3b_ldlm.yaml`, `qwen3_6_35b_a3b_repr_align.yaml`.
- **SFT**: `configs/sft/` — conversation data, DeepSeek MoE support.
- **Multimodal**: `configs/multimodal/` — vision-language, omni-modal, representation alignment.

### Model Implementations (`veomni/models/transformers/`)
Each model family is a subpackage with its own `modeling_*.py` and optional `generation_utils.py`:
- **qwen2** — base autoregressive (Qwen2-0.5B/7B/32B/72B)
- **qwen2_vl** / **qwen2_5vl** — vision-language variants
- **qwen3** — Qwen3 (newer generation)
- **qwen3_5** — Qwen3.5/3.6 architecture with hybrid linear/full attention (Gated DeltaNet)
- **qwen3_5_moe** — Qwen3.5/3.6 MoE variant (256 experts, shared expert, expert parallelism)
- **llama** — LLaMA3-8B/72B
- **deepseek_v3** — MoE models with routed experts

New models are registered in `veomni/models/transformers/__init__.py`. Architecture JSON configs live in `configs/model_configs/{family}/`.

### Seed Omni (`veomni/models/seed_omni/`)
Multi-modal foundation model combining encoders (e.g., Qwen2-VL vision) with decoders (e.g., MOVQGAN). Built via `build_omni_model()`.

### Distributed Training (`veomni/distributed/`)
Controlled by `data_parallel_mode` in `TrainingArguments`:

- **`ddp`**: standard distributed data parallel
- **`fsdp1`**: full-shard data parallel via PyTorch FSDP (default for large models)
- **`deepspeed`**: ZeRO-1/2/3 + CPU/NVMe offload. Relevant YAML fields:
  ```yaml
  train:
    data_parallel_mode: deepspeed
    ds_zero_stage: 3
    ds_offload_param: cpu      # null | cpu | nvme  (zero3 only)
    ds_offload_optimizer: cpu  # null | cpu | nvme
    ds_nvme_path: /run/media/johndpope/12TB/open_dllm/ds_offload
  ```
  Launch via `torchrun` (not `deepspeed` CLI). `enable_full_shard` and `enable_fsdp_offload` are ignored under DeepSpeed.
- **Sequence parallel (Ulysses)**: `veomni/distributed/sequence_parallel/` — splits long sequences across GPUs
- **MoE**: `veomni/distributed/moe/` — expert parallelism, fused MoE kernels
- **Parallel plan**: `parallel_plan.py` / `vescale_plan.py` define sharding strategies

### Data (`veomni/data/`)
Supports both plaintext and conversation formats. Key: `build_mapping_dataset()` (map-style), `build_iterative_dataset()` (iterable/streaming). Dynamic batching via `dynamic_batching.py`.

### Loss Functions (`veomni/ops/loss.py`)
Cross-entropy losses with fused kernel support: `seed_kernels` > `liger-kernel` > vanilla fallback.

### Checkpointing (`veomni/checkpoint/`)
Primary manager is `bytecheckpoint` with DCP (Distributed Checkpoint) format. `scripts/mereg_dcp_to_hf.py` converts to HF format.

## Evaluation

- **Code completion**: `eval/eval_completion/` — uses lm-evaluation-harness (HumanEval, MBPP)
- **Code infilling**: `eval/eval_infill/` — uses torchrun with DDP
- Both use `accelerate launch` or `torchrun` with custom diffusion generation

## Key Patterns

- Models are loaded via `veomni/models/auto.py`: `build_foundation_model(config_path, weights_path, ...)` which dispatches to per-family loaders in `veomni/models/loader.py`
- Diffusion generation uses `model.diffusion_generate()` with `MDMGenerationConfig` (mask tokens, steps, algorithm selection like `p2`)
- All model classes use `trust_remote_code=True`
- Config files reference HDFS paths for ByteDance internal clusters; local development uses HF model paths

### Three diffusion paths
The repo supports three ways of producing a diffusion LM (don't confuse them):

1. **Repr-Align** (`train_torch.py` with `repr_align_wt > 0`) — flips the AR model's attention mask to bidirectional and adds a cosine-sim alignment loss against a frozen causal teacher's hidden states. **No new parameters** — reuses the existing model weights. 3-4× faster convergence. Built into `modeling_qwen2.py`, `modeling_qwen3.py`, `modeling_qwen3_5_moe.py`. The teacher is a **frozen anchor**, not a live distillation source — precompute its hidden states once via `scripts/precompute_anchor.py` and cache to disk.

2. **LDLM** (`train_ldlm.py`) — trains a new Perceiver encoder/decoder + DiT head (1.39B–6.75B params) on top of a **frozen** AR encoder. Latent-space diffusion. Implementation in `veomni/models/ldlm/` (`LDLMAutoencoder`).

3. **Cola DLM** (opt-in auxiliary head on Repr-Align, `cola_enabled: true`) — adds a hierarchical Text VAE encoder (Perceiver → `z_global`, `z_local`) + block-causal DiT denoiser on top of Repr-Align. Documented in `docs/cola_ldm.md`. The LDLM stack is untouched. Configure `cola_prediction: "v"` (Flow Matching, default) or `"x0"` (cosine schedule).

If the user says "train a diffusion model" without specifying, ask which path they want. Repr-Align is the default recommendation for converting an existing AR model.

### Repr-Align anchor precomputation
Before training with `repr_align_wt > 0`, precompute teacher hidden states once:

```bash
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \
    --layers 7,14,21,28 \
    --max_seq_len 2048 \
    --max_examples 1000   # omit for full dataset
```

Cache contract: one `.safetensors` file per sequence chunk, keyed by SHA-256 of `input_ids`, stored in a 2-char prefix subdirectory. The trainer's `CachedTeacher` (in `veomni/models/cached_teacher.py`) splits packed rmpad rows via `position_ids` before lookup. Cache 4–8 selected layers (not all 40) to stay under 7 TB for a 35B model.

## Local Data & Models

Training data and pre-initialized model checkpoints live on an external 12TB drive:

- **Training data** (FineWeb 100K sample): `/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl` (~300MB, 100K plaintext examples)
- **35B-A3B LDLM untrained checkpoint**: `/run/media/johndpope/12TB/open_dllm/ldlm_model/ldlm_35b_a3b_untrained.pt` (~5.5GB, state dict with keys: `latent_encoder`, `latent_decoder`, `token_decoder`, `lm_head`, `diffusion_head`, `config`)
- **27B LDLM untrained checkpoint**: `/run/media/johndpope/12TB/open_dllm/ldlm_model/ldlm_untrained.pt` (~27GB)
- **Training checkpoints output**: `/run/media/johndpope/12TB/open_dllm/checkpoints/35b_a3b_ldlm/`

The 35B-A3B config (`configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml`) points to these paths. Launch with:
```bash
torchrun --nproc_per_node=1 tasks/train_ldlm.py configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

**Multi-GPU for LDLM**: Always use `--nproc_per_node=1`. The script places the frozen encoder on GPU 0 via `device_map="auto"` and trainable components (Perceiver, diffusion head) on GPU 1. Do NOT use `--nproc_per_node=2`.

## Local Training Hardware

See **`docs/local_training.md`** for the full inventory and upgrade path analysis, the 35B-A3B Repr-Align memory budget, the split-compute architecture, and the rent-vs-buy decision tree.

Key facts:
- **HP Z6 G4** (`johndpope@192.168.1.101`): Xeon Silver 4108 (Skylake-SP, no PMEM), 48 GB DDR4 mixed, RTX 3090 + Quadro P2000. 6 DIMM slots (1-DPC → Memory Mode Optane impossible regardless of CPU).
- **MSI box**: i5-13600KF, RTX 5090 (32 GB) + RTX PRO 4000 (24 GB), 96 GB DDR5. CUDA 12.9 required for Blackwell (RTX 5090) — handled by `[tool.uv]` index in `pyproject.toml`.
- Repr-Align teacher is a frozen anchor → precompute hidden states **once**, cache to the 12 TB drive, reuse forever. Do not build live RPC teacher infra.
- 35B-A3B student state is ~580 GB; no on-hand machine fits this without CPU offload. Default to **renting 8× H100** ($300–500 per epoch) unless a sustained-local-iteration case is made. DeepSpeed ZeRO-3 + NVMe offload is the local fallback path (see `docs/prd_deepspeed_integration.md`).
- Split-compute strategy: anchor precompute on MSI → student training on HP Z6 (or rented cluster).

## Cloud Training (Vast.ai)

See **`docs/cloud_training.md`** for the full Vast.ai setup guide (instance provisioning, S3 sync, launch scripts).

### Active instance (may change on restart)
- **Hardware**: 2× RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97.9 GB VRAM each, SM 12.0
- **CUDA**: 13.0, PyTorch 2.12.0+cu130
- **SSH** (port changes per instance): `ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<IP>`
- **Workspace**: `/workspace/Open-dLLM`
- **Python venv**: `/workspace/Open-dLLM/.venv/bin/python3` (no pip — use `/root/.local/bin/uv pip install`)

### On-instance paths
```
/data/models/Qwen3.6-27B/          # model weights
/data/anchors/qwen3.6-27b/         # precomputed Repr-Align anchor cache (1085 files, 27 GB)
/data/training/data_smoke_1000.jsonl
/data/checkpoints/qwen3.6-27b-repr-align/
/data/ds_offload/                  # DeepSpeed NVMe offload scratch
```

### Cloud training config
`configs/pretrain/cloud_27b.yaml` — 27B Repr-Align on 2× RTX PRO 6000 Blackwell.

Launch command:
```bash
cd /workspace/Open-dLLM
nohup .venv/bin/torchrun --nproc_per_node=2 tasks/train_torch.py configs/pretrain/cloud_27b.yaml \
    > /tmp/train.log 2>&1 &
echo $! > /tmp/train.pid
```

Monitor: `tail -f /tmp/train.log`
Push checkpoint to S3: `bash scripts/cloud/push_ckpt_s3.sh`

### Critical gotchas for Qwen3.6-27B (qwen3_5 architecture)

**Gated DeltaNet NaN backward pass** — Qwen3.6-27B uses `model_type: qwen3_5`, which has 75% Gated DeltaNet linear attention layers (every 4th layer is full attention). Without `flash-linear-attention` + `causal-conv1d`, training falls back to a torch sequential implementation that produces NaN gradients from step 2 onward. Symptoms: step 1 trains fine (loss ~9.3, large grad_norm), step 2+ shows `loss=nan, grad_norm=3.61` (DeepSpeed detects NaN, skips optimizer step, returns stale grad_norm).

Install fix:
```bash
cd /workspace/Open-dLLM
/root/.local/bin/uv pip install causal-conv1d flash-linear-attention
```
If pre-built wheels don't exist for SM 12.0 / CUDA 13.0, build from source:
```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE /root/.local/bin/uv pip install causal-conv1d
MAX_JOBS=4 /root/.local/bin/uv pip install git+https://github.com/fla-org/flash-linear-attention
```

**`save_time_interval_minutes` bypasses `save_optimizer_state: false`** — The time-based checkpoint path in `train_torch.py` called `engine.save_checkpoint()` directly, writing 211 GB ZeRO-3 state regardless of `save_optimizer_state`. Fixed by guarding with `if save_time and args.train.save_optimizer_state:`. Always set `save_time_interval_minutes: 0` in cloud configs.

**`anyprecision_adamw` NaN with bf16** — This optimizer stores the second moment `v` in bf16; small gradients cause `v=0` in bf16, giving `update = m/eps` → NaN. Use `optimizer: adamw` (fp32 states) for training stability.

**`repr_align_sub_sample_ratio: 0.25`** — Randomly samples 25% of token positions for cosine-sim alignment loss. Cuts alignment gradient memory ~4×. Required for 2× Blackwell at seq_len 2048 with ZeRO-3.
