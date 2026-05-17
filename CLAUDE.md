# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-dLLM is a diffusion-based large language model framework for training, evaluation, and inference. It converts autoregressive LMs (Qwen, LLaMA, DeepSeek) into discrete diffusion LMs, offering ~4x speedup. The core package is `veomni/`.

## Commands

```bash
# Setup
pip install -e ".[dev]"

# Linting & formatting (ruff, line-length=119, target py38)
make style          # auto-fix
make quality        # check only

# Tests
make test           # pytest tests/

# Pre-commit
make commit         # pre-commit run --all-files
```

No single-file test runner is configured; use `pytest tests/path/test_file.py` directly.

## Architecture

### Training Entry Points (`tasks/`)
Two training scripts, both YAML-driven via the Hydra-style argument parser in `veomni/utils/arguments.py` (three dataclass groups: `ModelArguments`, `DataArguments`, `TrainingArguments`):

- **`tasks/train_torch.py`** — standard MDM training. Also handles the **Repr-Align** path (bidirectional student + frozen causal teacher) when `train.repr_align_wt > 0` and/or `train.enable_masking=true`. Uses FSDP1 or DDP across all visible GPUs.
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
- **FSDP1**: full-shard data parallel via PyTorch FSDP
- **DDP**: standard distributed data parallel
- **Sequence parallel (Ulysses)**: `veomni/distributed/sequence_parallel/` — splits long sequences across GPUs
- **MoE**: `veomni/distributed/moe/` — expert parallelism, fused MoE kernels
- **Parallel plan**: `parallel_plan.py` / `vescale_plan.py` define sharding strategies

### Data (`veomni/data/`)
Supports both plaintext and conversation formats. Key: `build_mapping_dataset()` (map-style), `build_iterative_dataset()` (iterable/streaming). Dynamic batching via `dynamic_batching.py`.

### Loss Functions (`veomni/ops/loss.py`)
Implements cross-entropy losses with fused kernel support: `seed_kernels` > `liger-kernel` > vanilla fallback.

### Checkpointing (`veomni/checkpoint/`)
Primary manager is `bytecheckpoint` with DCP (Distributed Checkpoint) format. `mereg_dcp_to_hf.py` script converts to HF format.

## Evaluation

- **Code completion**: `eval/eval_completion/` — uses lm-evaluation-harness (HumanEval, MBPP)
- **Code infilling**: `eval/eval_infill/` — uses torchrun with DDP
- Both use `accelerate launch` or `torchrun` with custom diffusion generation

## Key Patterns

- Models are loaded via `veomni/models/auto.py`: `build_foundation_model(config_path, weights_path, ...)` which dispatches to per-family loaders in `veomni/models/loader.py`
- Diffusion generation uses `model.diffusion_generate()` with `MDMGenerationConfig` (mask tokens, steps, algorithm selection like `p2`)
- All model classes use `trust_remote_code=True`
- Config files reference HDFS paths for ByteDance internal clusters; local development uses HF model paths

### Two diffusion paths
The repo supports two distinct ways of producing a diffusion LM (don't confuse them):

1. **Repr-Align** (`train_torch.py` with `repr_align_wt > 0`) — flips the AR model's attention mask to bidirectional and adds a cosine-sim alignment loss against a frozen causal teacher's hidden states. **No new parameters** — reuses the existing model weights. 3-4× faster convergence. Built into `modeling_qwen2.py`, `modeling_qwen3.py`, `modeling_qwen3_5_moe.py`.
2. **LDLM** (`train_ldlm.py`) — trains a new Perceiver encoder/decoder + DiT head (1.39B–6.75B params) on top of a **frozen** AR encoder. Latent-space diffusion. Heavier but more expressive. Implementation in `veomni/models/ldlm/` (LDLMAutoencoder).

If the user says "train a diffusion model" without specifying, ask which path they want. Repr-Align is the default recommendation for converting an existing AR model.

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

**Multi-GPU**: Always use `--nproc_per_node=1`. The script places the frozen encoder on GPU 0 via `device_map="auto"` and trainable components (Perceiver, diffusion head) on GPU 1. Do NOT use `--nproc_per_node=2`.

## Local Training Hardware

See **`docs/local_training.md`** for the full inventory and upgrade path analysis of the user's local boxes (HP Z6 G4, MSI 5090+RTX PRO 4000), the 35B-A3B Repr-Align memory budget, the split-compute architecture (anchor precompute on MSI → student train on Z6), and the rent-vs-buy decision tree.

Key facts to keep in mind without re-reading the doc:
- **HP Z6 G4** = Xeon Silver 4108 (Skylake-SP, no PMEM), 48 GB DDR4 mixed, RTX 3090 + Quadro P2000, 6 DIMM slots (1-DPC → Memory Mode Optane impossible regardless of CPU).
- Repr-Align teacher is a frozen anchor (not distillation) → precompute hidden states **once**, cache to the 12 TB drive, reuse forever. Don't build live RPC teacher infra.
- Cache 4–8 selected layers (not all 40) to keep precompute under 7 TB.
- 35B-A3B student state is ~580 GB; no on-hand machine fits this without CPU offload. Default to **renting 8× H100** ($300–500 per epoch) unless a sustained-local-iteration case is made.
