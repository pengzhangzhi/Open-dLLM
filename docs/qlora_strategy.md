# QLoRA Repr-Align Strategy

## Why QLoRA, Not DeepSpeed Full-Weight

A 27B model in bf16 = 54 GB. With fp32 optimizer states (216 GB) + gradients (54 GB) = **324 GB total**. Our hardware (96 GB RAM + 34 GB + 25 GB GPU) cannot hold this, even with ZeRO-3 + CPU/NVMe offload:

- Model creation inside `zero.Init()` materializes parameters on CPU, consuming 36+ GB RSS per rank → systemd-oomd kills the process
- `pin_memory()` of the fp16 flat buffer fails on GPUs with < 27 GB VRAM
- NVMe offload on an HDD causes 94% iowait → swap death

**QLoRA** (4-bit NF4 + LoRA adapters) works because:
- 4-bit quantization shrinks params from 54 GB → 13.5 GB
- LoRA adapters add only ~0.5% trainable params (negligible memory)
- Single RTX 5090 (34 GB) fits the model (19.2 GB NF4) + activations with room to spare
- Data parallelism (DDP) works without ZeRO because each GPU has its own 4-bit copy

## Working Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  tasks/train_torch.py configs/pretrain/qlorafy_27b_smoke.yaml \
  > /tmp/qlorafy_smoke.log 2>&1 &
```

**Critical: `CUDA_VISIBLE_DEVICES=0`** — Physical GPU 0 is the RTX 5090 (34 GB), GPU 1 is the RTX PRO 4000 (25 GB). Loading on GPU 1 OOMs.

## Config

`configs/pretrain/qlorafy_27b_smoke.yaml` (10 steps) / `qlorafy_27b_train.yaml` (2000 steps):

```yaml
model:
  model_path: /home/johndpope/ds_offload/models/Qwen3.6-27B
  enable_qlorafy: true
  qlorafy_config:
    r: 32
    lora_alpha: 64
    target_modules:
      - q_proj  - k_proj  - v_proj  - o_proj
      - gate_proj  - up_proj  - down_proj
    use_rslora: true

data:
  train_path: /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl
  train_size: 1000
  max_seq_len: 2048       # must match anchor precompute
  datasets_type: mapping

train:
  data_parallel_mode: ddp  # no DeepSpeed for QLoRA
  rmpad: true              # matches anchor tokenization (no BOS)
  rmpad_with_pos_ids: true
  enable_masking: true
  repr_align_wt: 1.0
  anchor_cache_dir: /home/johndpope/ds_offload/anchors/qwen3.6-27b
  align_layers: "16,32,48,64"
  repr_align_sub_sample_ratio: 0.25
  optimizer: adamw
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  save_hf_weights: true    # PEFT adapter export (not DCP)
```

## What Almost Worked

1. **Model loaded** — `build_qlorafied_model()` in `veomni/models/qlorafy.py` loaded the 27B in NF4 (19.2 GB VRAM) via the VL weight-remap path (shard-by-shard key mapping from `model.language_model.*` → `model.*`)
2. **LoRA adapters attached** — PEFT `LoraConfig` with `TaskType.FEATURE_EXTRACTION` on q/k/v/o/gate/up/down projections
3. **W&B connected** — run `lup4kq7l` at https://wandb.ai/snoozie/open-dllm/runs/lup4kq7l
4. **Forward pass reached** — crashed in `CachedTeacher._load_chunk` with anchor cache miss

### Anchor Cache Miss — Root Cause

`precompute_anchor.py` tokenizes as:
```python
tokens = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
# chunked at max_seq_len=2048
```

With `rmpad: false` (old config), the training pipeline added BOS token → hash mismatch. **Fix: use `rmpad: true rmpad_with_pos_ids: true`** so the `CachedTeacher` splits packed sequences by `position_ids` and hashes each chunk without BOS, matching the anchor format.

## Next Steps

| Step | Command | Expected |
|------|---------|----------|
| 1. Launch smoke test | `CUDA_VISIBLE_DEVICES=0 ... qlorafy_27b_smoke.yaml` | 10 steps, loss decreases |
| 2. If anchor miss persists | Fix `precompute_anchor.py` to match training tokenization | Hashes align |
| 3. Launch full training | `CUDA_VISIBLE_DEVICES=0 ... qlorafy_27b_train.yaml` | 2000 steps, ~30 mins |
| 4. Push anchors to S3 | `bash scripts/cloud/push_ckpt_s3.sh` | Persistent cache |

## Hardware Layout (MSI Box)

| GPU | Physical Index | Model | VRAM | CUDA_VISIBLE_DEVICES |
|-----|---------------|-------|------|---------------------|
| RTX 5090 | 0 | GeForce RTX 5090 | 34 GB | `0` |
| RTX PRO 4000 | 1 | RTX PRO 4000 Blackwell | 25 GB | `1` |

System RAM: 96 GB DDR5

## Key Files

| File | Role |
|------|------|
| `veomni/models/qlorafy.py` | NF4 loading + PEFT adapter injection |
| `veomni/models/auto.py` | Dispatch to `build_qlorafied_model()` |
| `configs/pretrain/qlorafy_27b_smoke.yaml` | Smoke test config |
| `configs/pretrain/qlorafy_27b_train.yaml` | Full training config |
| `scripts/precompute_anchor.py` | Anchor cache computation |
| `veomni/models/cached_teacher.py` | Anchor cache lookup + packing split |
