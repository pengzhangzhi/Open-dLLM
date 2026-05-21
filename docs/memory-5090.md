# Memory Reduction Playbook — MSI Box (RTX 5090 + RTX PRO 4000)

**Hardware:** i5-13600KF, 96 GB DDR5, RTX 5090 (32 GB) + RTX PRO 4000 (24 GB) = 56 GB VRAM total  
**NVMe:** Samsung 990 Pro 2 TB (`nvme0n1`, Gen5, ~6 GB/s sequential) — `~/ds_offload` lives here  
**Target model:** Qwen3.6-35B-A3B (`model_type: qwen3_5_moe`, 40 layers, 256 experts / 8 active)

---

## The Problem in Numbers

With current config (`optimizer: adamw`, `ds_offload_optimizer: nvme`, `ds_offload_param: cpu`):

| Component | Size | Where |
|-----------|------|-------|
| bf16 params (35B × 2 B) | 70 GB total → **35 GB/rank** | CPU RAM |
| fp32 optimizer states (35B × 12 B) | 420 GB total → **210 GB/rank** | NVMe |
| bf16 gradients (ZeRO-3 sharded) | 70 GB total → **35 GB/rank** | GPU (ZeRO reduces incrementally) |
| Active GPU mem (1 layer gathered) | ~2–4 GB | GPU |
| Alignment overhead (repr_align) | ~16–30 MB | GPU |

**NVMe I/O per step:** read 210 GB + write 210 GB = **420 GB/rank at ~6 GB/s = ~70 seconds/step minimum.**  
**CPU RAM headroom:** 70 GB params + ~10 GB overhead = 80 GB needed vs 96 GB available. ~16 GB margin.  
**VRAM:** fine — ZeRO-3 keeps only the active layer on GPU (~4 GB peak). Not the bottleneck.

Subsampling (`repr_align_sub_sample_ratio`, `repr_align_num_sample_layers`) saves ~30 MB of the 80 GB requirement — rounding error. **The enemy is fp32 optimizer states on NVMe.**

---

## Fix 1: 8-bit Optimizer (Biggest Single Win, ~3× NVMe reduction)

Switch from fp32 AdamW (12 B/param) to 8-bit Adam (2–3 B/param).

```bash
uv pip install bitsandbytes
```

In `configs/pretrain/qwen3_6_35b_a3b_full_repr_align_ds.yaml`:
```yaml
train:
  optimizer: adamw_8bit   # bitsandbytes 8-bit AdamW
```

veomni's `train_torch.py` would need a dispatch for `adamw_8bit` → `bnb.optim.AdamW8bit`. DeepSpeed's own 8-bit path is `"type": "Adam8bit"` in the raw DS config.

**Effect:**
- Optimizer states: 210 GB/rank → **~60 GB/rank**
- NVMe I/O per step: 420 GB → **~120 GB → ~20 seconds/step**
- CPU RAM: unchanged (params still on CPU)

**Risk:** 8-bit Adam can lose precision on very small gradients. MoE sparse experts see infrequent updates — watch for expert collapse. Monitor `router_aux_loss` and per-expert load balance.

---

## Fix 2: ZeRO-Infinity Tuning (Free, Do This Now)

The current config is missing performance knobs. Add to the DeepSpeed zero block (veomni maps these via `ds_zero_stage` + the raw DS JSON builder in `veomni/distributed/deepspeed/`):

```json
"zero_optimization": {
  "stage": 3,
  "overlap_comm": true,
  "contiguous_gradients": true,
  "offload_optimizer": {
    "device": "nvme",
    "nvme_path": "/home/johndpope/ds_offload/zero_stage_3",
    "pin_memory": true,
    "buffer_count": 4,
    "fast_mode": true
  },
  "offload_param": {
    "device": "cpu",
    "pin_memory": true
  },
  "stage3_max_live_parameters": 2e9,
  "stage3_max_reuse_distance": 2e9,
  "stage3_prefetch_bucket_size": 5e8,
  "stage3_param_persistence_threshold": 1e6,
  "reduce_bucket_size": 5e8
}
```

`pin_memory: true` on both offload buffers cuts PCIe latency for CPU↔GPU transfers.  
`fast_mode: true` uses async NVMe I/O (requires `libaio-dev` + `async_io` DeepSpeed op, already installed).  
Larger `buffer_count` lets multiple NVMe reads pipeline while compute runs.

**Effect:** Hard to quantify without profiling — rough 10–30% throughput improvement from overlap.

---

## Fix 3: Freeze Early Layers (Largest Optimizer State Reduction Without Precision Loss)

Repr-Align trains the full stack, but gradient signal from early layers is weak — the anchor diverges most at mid/late layers. Freeze the bottom third:

```yaml
train:
  freeze_layers: "0,1,2,3,4,5,6,7,8,9,10,11,12,13"  # freeze first 14 of 40
```

Frozen layers contribute zero optimizer state.

**Effect on 35B-A3B (40 layers):**
- Rough param fraction in first 14 layers (dense embed + attention + MoE): ~35% of params
- Optimizer states: 210 GB/rank → **~135 GB/rank**
- NVMe I/O: 420 GB → **~270 GB → ~45 seconds/step**

Combine with 8-bit Adam: 135 GB × 0.3 = **~40 GB/rank → ~13 seconds/step**.

**Risk:** Early layers handle token embedding and positional structure. Freezing them locks in the AR causal attention pattern — fine for Repr-Align since the goal is to shift hidden state geometry in mid/late layers where semantic content lives.

---

## Fix 4: ZeRO-2 Instead of ZeRO-3 (If Params Fit in VRAM)

ZeRO-3 shards params across ranks and gathers them layer by layer — constant NVMe/PCIe pressure. ZeRO-2 keeps full params on each rank but shards only optimizer states + gradients.

| | ZeRO-2 | ZeRO-3 |
|--|--------|--------|
| Params per rank | 70 GB (full) | 35 GB (sharded) |
| Optimizer/rank | 210 GB NVMe | 210 GB NVMe |
| Param transfer per step | None (already local) | 40 layer gather/scatter cycles |

With 96 GB DDR5 and CPU param offload, ZeRO-2 + CPU offload stores 70 GB params on CPU (vs 35 GB for ZeRO-3). That consumes the 16 GB headroom entirely — **likely OOM** for 35B-A3B.

**Verdict for 35B-A3B:** ZeRO-3 is mandatory. ZeRO-2 is viable for ≤27B if RAM holds.

---

## Fix 5: LoRA on Dense Layers (Nuclear Option for Optimizer States)

Apply LoRA (rank 16–32) to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in the 10 full-attention layers. MoE expert weights stay full-rank (they're already sparse).

- Full-attention layers: 10 of 40 (every 4th)
- Attention params per layer: ~4 × (hidden × head_dim × n_heads) = 4 × 2048 × 128 × 16 ≈ 268M params
- LoRA rank 16: trains 2 × 2048 × 16 × 4 = 262K params per layer instead of 268M
- Reduction: **~1000× per adapter layer**

Optimizer states for LoRA adapters only: negligible NVMe pressure.  
MoE expert weights (the bulk of 35B) still need optimizer states unless frozen or 8-bit.

LoRA is not currently wired into veomni's training loop — would require `peft` integration or manual injection.

---

## Realistic Step Time Estimates on MSI Box

After applying fixes (8-bit Adam + freeze first 14 layers + ZeRO-Infinity tuning):

| Config | Optimizer states/rank | NVMe I/O/step | Est. step time |
|--------|----------------------|--------------|----------------|
| Current (fp32 Adam, all layers) | 210 GB | 420 GB | **~70–120 s** |
| 8-bit Adam, all layers | 60 GB | 120 GB | **~20–35 s** |
| fp32 Adam, freeze 14 layers | 135 GB | 270 GB | **~45–75 s** |
| 8-bit Adam + freeze 14 layers | ~40 GB | ~80 GB | **~13–25 s** |
| 8-bit Adam + freeze 14 layers + ZeRO-Infinity tuning | ~40 GB | ~80 GB | **~10–18 s** |

All estimates assume 6 GB/s NVMe sequential. Async overlap may hide some I/O behind compute.

---

## What Subsampling Actually Helps

| Scenario | Subsampling saves | Verdict |
|----------|-----------------|---------|
| 35B-A3B local (ZeRO-3 + NVMe) | ~30 MB vs 80 GB budget | Irrelevant |
| 27B on Vast.ai (97.9 GB VRAM, no offload) | ~1–2 GB alignment activations vs 54 GB model | Marginal but real at long seq |
| 1.7B smoke runs | ~8 MB vs 9 GB model | Irrelevant |

Subsampling's value: **reducing anchor cache reads from disk** (fewer layers loaded per step) and **alignment compute time** (wall clock, not memory). Both are sub-1% of total step time at 35B scale.

---

## Recommended Action Order

1. **Now (zero code change):** Add ZeRO-Infinity tuning knobs to the DS config builder — `pin_memory`, `fast_mode`, `buffer_count`, `stage3_prefetch_bucket_size`. Free 10–30% speedup.

2. **Short term:** Wire `optimizer: adamw_8bit` in `train_torch.py` → `bnb.optim.AdamW8bit`. Largest single win, ~3× NVMe reduction.

3. **Short term:** Set `freeze_layers: "0,1,...,13"` for validation runs. Cuts another ~35% of optimizer states.

4. **Use local MSI box for:** debug runs (is loss finite? are hooks firing? does repr_align loss decrease?), not throughput. Accept 20–60 s/step as the floor.

5. **Use rented H100s for:** actual pretraining. 8× H100 = no NVMe offload needed (optimizer states fit in HBM), steps in 1–3 s, $300–500/epoch.
