# QLoRA Repr-Align Assessment: Smoke Test vs Production

**Date:** 2026-05-20  
**Run:** `qlorafy-27b-v2` (wandb ID: `14f1q0cj`)  
**Config:** `configs/pretrain/qlorafy_27b_train.yaml`  
**Hardware:** 1× RTX 5090 (32 GB), CUDA_VISIBLE_DEVICES=0


## Current Run Metrics (Steps 1–19)

```
step  loss   grad_norm  qlora_gn  tok/s  mfu   flops_achieved
  1   5.19    59.29      59.29     387   0.16   55T
  2   7.55    54.02      54.02     405   0.17   58T
  3   8.04    87.81      87.81     347   0.15   49T
  5   5.54    19.59      19.59     473   0.20   67T
 10   4.99     5.08       5.08     381   0.16   54T
 15   5.57     3.69       3.69     427   0.18   61T
 19   5.13     2.43       2.43     466   0.20   66T
```

- **Loss:** 5.2 → 5.1 (barely moving in 19 steps)
- **Grad norm:** 59 → 2.4 (stabilizing fast — healthy)
- **QLoRA grad norm:** matches total grad_norm (all trainable params are LoRA)
- **NaN:** 1 occurrence at step 21 (intermittent, recovered)
- **VRAM:** 24 GB allocated / 32 GB reserved on RTX 5090

---

## What This Run Proves (Integration Test)

1. QLoRA + Repr-Align forward/backward works on 27B with text-only `Qwen3_5ForCausalLM`
2. Gradients are finite, loss is decreasing, no systematic NaN
3. Single-GPU 32 GB fits 27B NF4 + LoRA r=32 training
4. Wandb logging captures real metrics (after fixes)
5. Text-only model class saves ~4.8 GB by not instantiating the vision encoder

**This is a smoke test, not a production run.**

---

## Production Gaps

### P0 — Will Not Produce Useful Alignment Without Fixing

| Gap | Current | Production Need | Impact |
|---|---|---|---|
| Anchor cache layers | 4 / 64 (6%) | ≥16, ideally all 64 | Alignment signal covers only 6% of model depth — 94% has no guidance |
| Training data | 569K tokens (1000 examples × ~569 avg tokens) | 50M–500M tokens | ~100–1000× too little data for convergence |
| Evaluation pipeline | None | Held-out perplexity + downstream tasks + bidirectional quality metric | Cannot tell if alignment is working beyond "loss goes down slightly" |

### P1 — Quality Ceiling

| Gap | Current | Production Need | Impact |
|---|---|---|---|
| LoRA rank | 16 (0.3% trainable) | 32–128 (0.6–2.4%) | r=16 can only make small perturbations; full-param Repr-Align moves all 27B weights bidirectional |
| Sequence length | 1024 | 2048–4096 | Short context limits what the model learns about bidirectional dependencies |
| Checkpointing | DCP fails with `Params4bit`; `save_hf_weights: true` for PEFT export only | Full checkpoint/resume | Cannot resume from interruption; cannot select best checkpoint |

### P2 — Operational

| Gap | Current | Production Need | Impact |
|---|---|---|---|
| NaN stability | 1 occurrence in 25 steps | Zero or well-understood | Intermittent; recovered, but needs monitoring over longer runs |
| MFU | 19% | 30–50% (larger micro_batch) | micro_batch=1 × 16 grad_accum is inherently low MFU |
| Baseline comparison | None | Full-param Repr-Align loss curve | Don't know what "good" looks like for QLoRA vs full-param |
| Generation probe | Falls back to `model.generate()` (AR) | Should test bidirectional generation quality | AR generation doesn't validate alignment at all |

---

## Production Requirements vs Current

| Aspect | Current | Production Minimum | Production Ideal |
|---|---|---|---|
| Trainable params | 80M (0.3%) | 80M (r=32) | 500M (r=128) or full-param |
| Alignment layers | 4 / 64 | 16 / 64 | All 64 |
| Training data | 569K tokens | 50M tokens | 500M tokens |
| Training steps | 2000 | 10K | 50K |
| Batch size (tokens/step) | 9K | 32K | 128K |
| Loss target | ? | < 4.0 (ppl < 55) | < 3.5 (ppl < 33) |
| Eval set | none | held-out perplexity | perplexity + downstream tasks |
| Seq length | 1024 | 2048 | 4096 |
| Hardware | 1× RTX 5090 | 2× H100 | 8× H100 |
| Training time | ~11 hrs (2000 steps) | 4–8 hrs (cluster, 10K steps) | 12–24 hrs (cluster, 50K steps) |

### LoRA Rank vs VRAM (Qwen3.6-27B, NF4, seq_len=1024)

| Rank | LoRA Params | Trainable % | LoRA GB | Optim GB | Total Est | Measured | Fits RTX 5090? |
|---|---|---|---|---|---|---|---|
| 16 | 73M | 0.27% | 0.15 | 0.59 | 24.0 GB | 24.0 GB | Yes |
| 32 | 147M | 0.54% | 0.29 | 1.17 | 24.9 GB | 27.8 GB* | Yes (needs `expandable_segments:True`) |
| 64 | 294M | 1.09% | 0.59 | 2.35 | 26.6 GB | OOM | No (28.4 GB + activations > 31.4 GB) |
| 128 | 587M | 2.17% | 1.17 | 4.70 | 30.2 GB | — | No |

*Measured is higher than estimated due to PEFT wrapping overhead and Gated DeltaNet activation tensors.

**r=64 requires either:** (1) 2× GPU with model parallelism, (2) seq_len=512, or (3) a 48+ GB GPU (A6000, A100 80GB).

---

## Minimum Viable Path to Production

| Step | Cost | Time | Blocking? |
|---|---|---|---|
| 1. Precompute anchor cache for all 64 layers on 100K examples | ~$5 on Vast.ai (2× GPU, 2 hrs) | 2 hrs | Yes — without this, alignment is near-zero quality |
| 2. Use full 100K FineWeb dataset (already have it) | $0 | config change | No |
| 3. Bump LoRA to r=32 (r=64 OOMs on 32 GB GPU) | $0 | r=32 fits with `expandable_segments:True` | No |
| 4. Add eval pipeline (perplexity + generation probe) | $0 | ~2 hrs code | Yes — cannot measure progress without it |
| 5. Run 10K+ steps on cloud (8×H100) | ~$30–50 | 4–8 hrs | No (could also run locally over ~55 hrs) |
| 6. PEFT adapter export for checkpointing | $0 | `save_hf_weights: true` already works | No |

**Estimated total to production-quality run:** ~$35–55 + 1 day of engineering.

---

## Known Bugs Fixed This Session

1. **`qlora/grad_norm=0`** — `optimizer.zero_grad()` at `train_torch.py:618` cleared grads before metrics were read. Fixed by capturing LoRA grad norms between backward and zero_grad.
2. **`flops_achieved=0`** — `VeomniFlopsCounter` had no estimator for `model_type: qwen3_5_text`. Added `_estimate_qwen3_5_text_flops` accounting for 75% Gated DeltaNet (linear attention) + 25% full attention layers.
3. **`flops_promised=Infinity`** — `get_device_flops()` returned `float("inf")` for unknown GPUs. Added RTX 5090 (335 TFLOPS) and RTX PRO 4000/6000 Blackwell entries.
4. **`tokens_per_second` in millions** — Divided by 1e6, showing 0.0004M for 400 tok/s. Changed to raw tok/s.
5. **Wandb step stuck at 64** — `resume="allow"` always tried to resume old run. Changed to only resume when a checkpoint actually exists.
6. **Generation probe broken** — Used `diffusion_generate()` which only exists in custom model class. Added fallback to `model.generate()` for standard AR generation.
7. **Text-only model class** — `Qwen3_5ForConditionalGeneration` instantiated 4.74 GiB vision encoder even with `language_model_only=True`. Fixed `qlorafy.py` to load via HF's `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig` without `trust_remote_code`, saving ~4.8 GB VRAM.
8. **Dead code in `qlorafy.py`** — `_get_text_only_model_class` was defined inside `build_qlorafied_model` after the imports, making all subsequent code unreachable (function returned `None`). Rewrote the function properly.
