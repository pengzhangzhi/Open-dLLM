# PRD: DeepSpeed (ZeRO-3 + NVMe Offload) Integration for veomni

**Status:** draft  •  **Owner:** open-dLLM core  •  **Last updated:** 2026-05-18

---

## 1. Problem

veomni's `data_parallel_mode` today is `{ddp, fsdp1, fsdp2, fsdp2-vescale}`.
For models whose **bf16 weight footprint exceeds combined VRAM** (e.g.
Qwen3.6-35B-A3B at 70 GB on a 5090+PRO 4000 box = 56 GB total), FSDP1 with
`enable_fsdp_offload=true` keeps shards on CPU but still requires the full
state to fit in **CPU RAM minus activation/gradient working set**. On the MSI
box that's 91 GB total, which gives ~20 GB headroom after a 70 GB model — too
tight for a 27B-dense future or any seq-len growth.

The unmet capability is **NVMe-tier offload of params + grads + optimizer
states**. DeepSpeed ZeRO-Infinity already does this and is the lowest-effort
way to get it. FSDP2 has prototype CPU-resident offload but no NVMe tier as of
torch 2.5.

## 2. Goals / Non-goals

### Goals
- Add `data_parallel_mode: deepspeed` (with sub-mode for ZeRO-1/2/3 + offload tier).
- Train Qwen3.6-35B-A3B end-to-end on the local MSI box without OOM, using
  CPU+NVMe offload via the 12 TB drive.
- Preserve the Repr-Align + cached-anchor + quantize_frozen paths unchanged.
- Allow the same YAML configs (modulo one `data_parallel_mode` change) to run
  under DeepSpeed.
- Keep DDP / FSDP1 paths working — no regressions.

### Non-goals (this PRD)
- Pipeline parallelism. DeepSpeed PP integration is out of scope; the existing
  `pipeline_parallel_size=1` constraint stays.
- DeepSpeed Mixture-of-Experts kernels. We're keeping veomni's MoE path.
- Distributed checkpoint format unification. We'll persist DeepSpeed-format
  checkpoints; conversion to HF format is an existing tool
  (`mereg_dcp_to_hf.py`) that may need a sibling for DS format — track as a
  follow-up.

## 3. User-facing surface

### YAML
New field on `TrainingArguments`:

```yaml
train:
  data_parallel_mode: deepspeed
  ds_zero_stage: 3                  # 1, 2, 3
  ds_offload_optimizer: cpu         # null | cpu | nvme
  ds_offload_param: cpu             # null | cpu | nvme  (zero3 only)
  ds_nvme_path: /run/media/johndpope/12TB/open_dllm/ds_offload
  ds_overlap_comm: true
  ds_contiguous_gradients: true
  # Optional: bring-your-own ds_config.json (overrides the above)
  ds_config_path: ""
```

Existing knobs that **change semantics** under DeepSpeed:
- `enable_full_shard` → ignored (ZeRO stage controls sharding).
- `enable_fsdp_offload` → ignored; use `ds_offload_param` / `ds_offload_optimizer`.
- `enable_mixed_precision` → translated into ds_config `bf16.enabled = true`.
- `enable_gradient_checkpointing` → applied directly to model before DS init
  (DeepSpeed wraps but doesn't manage activation ckpt).

### Launch
DeepSpeed is launched via the same `torchrun --nproc_per_node=N` we already
use. We avoid `deepspeed` CLI to keep one entry point.

```
torchrun --nproc_per_node=2 tasks/train_torch.py configs/.../qwen3_6_35b_a3b_one_layer_repr_align_ds.yaml
```

## 4. Integration touchpoints

### 4.1 `veomni/utils/arguments.py`
- Add `data_parallel_mode` literal `"deepspeed"`.
- Add the 7 ds_* fields above (Literal/str/bool/int as appropriate).
- Add validation: `ds_zero_stage in {1,2,3}`; `ds_offload_param != null` ⇒
  `ds_zero_stage == 3`; `ds_offload_*` set to `"nvme"` requires
  non-empty `ds_nvme_path` that exists on disk.

### 4.2 `veomni/distributed/deepspeed_init.py` (new)
- `build_ds_config(args) -> dict`: translate TrainingArguments into a
  DeepSpeed JSON config. Handles `bf16/fp16`, ZeRO config, offload tiers,
  grad accumulation, train micro-batch.
- `init_deepspeed_engine(model, optimizer, args) -> (engine, optimizer, lr_sched_proxy)`:
  call `deepspeed.initialize(model=..., optimizer=..., config=...)`.
- Honor `ds_config_path` override: if set, load that JSON verbatim and only
  patch micro-batch + grad-accum from args.

### 4.3 `veomni/distributed/torch_parallelize.py`
- New branch in `build_parallelize_model`: if `data_parallel_mode == "deepspeed"`,
  skip FSDP wrap and instead call into `deepspeed_init`. Return the
  DeepSpeed engine (`DeepSpeedEngine`) which exposes a `.module` for the
  original model — adjust downstream callers that touch the model directly
  (Cola wrapper, freeze_layers, quantize_frozen).
- Important: `freeze_layers_by_patterns` and `quantize_frozen_linears` must
  run **before** `deepspeed.initialize` so DS sees the right `requires_grad`
  and the quantized tensors when choosing what to shard.

### 4.4 `tasks/train_torch.py`
- Replace `loss.backward()` with `engine.backward(loss)` when running DS.
- Replace `optimizer.step()` / `optimizer.zero_grad()` with `engine.step()`
  (DS handles both internally).
- Gradient clipping: pass via ds_config (`gradient_clipping`) instead of
  PyTorch-side `clip_grad_norm_`.
- Wandb step + global_step semantics unchanged — `engine.global_steps`
  exposes the same counter.

### 4.5 `veomni/checkpoint/checkpointer.py`
- Add a `ds` ckpt manager branch parallel to `dcp` / `bytecheckpoint`.
  Uses `engine.save_checkpoint(path, tag)` and `engine.load_checkpoint(path,
  tag)`. No DCP gather_object → the AffineQuantizedTensor pickle issue we
  fixed doesn't reappear here, but the quantize_frozen dequantize-on-save
  shim still applies symmetrically (DS save also walks state_dict).
- HF-weights export path: DeepSpeed has `zero_to_fp32.py` for stage-3
  consolidation; wire `save_hf_weights=true` to invoke that.

### 4.6 `veomni/optim/optimizer.py`
- `build_optimizer` keeps returning a plain torch optimizer; DS wraps it in
  `init_deepspeed_engine`. No changes to optimizer types
  (`adamw`, `anyprecision_adamw`, `apollo`) — but document that the actual
  optimizer state may live on CPU/NVMe so the in-memory params aren't where
  the state is.
- DeepSpeed has its own `DeepSpeedCPUAdam` that's faster than torch CPU
  Adam. Expose via `optimizer: ds_cpu_adam` as a new option (opt-in).

### 4.7 Repr-Align / CachedTeacher
- No code changes expected. `model.forward(...)` is wrapped by DS but the
  signature is preserved. CachedTeacher operates on the unwrapped
  `engine.module` if needed; in practice it's a sibling module, not inside
  the engine.
- Risk: `casual_input_ids` arg goes through `engine.forward(**micro_batch)`
  which uses `**kwargs` — verify DS doesn't strip unknown kwargs.

### 4.8 `pyproject.toml`
- Add `deepspeed >= 0.15` as an optional extra: `pip install -e ".[deepspeed]"`.
- Lock the version that ships ZeRO-Infinity NVMe with bf16 working
  (recent releases regressed; pin the known-good).

## 5. Phased delivery

| Phase | Scope | Acceptance |
|---|---|---|
| **P0** | YAML + ds_config builder + ZeRO-3 init (no offload). | Qwen3-1.7B smoke runs under DS ZeRO-3 on 2 GPUs, loss within 1% of FSDP baseline. |
| **P1** | CPU offload (param + optimizer). | Qwen3.6-35B-A3B one-layer smoke runs end-to-end on MSI box. Step time documented; expect 30-90 s/step. |
| **P2** | NVMe offload via ds_nvme_path on 12 TB drive. | 35B-A3B with `ds_offload_param: nvme` runs; CPU RAM usage drops to <40 GB; step time documented. |
| **P3** | Checkpoint save/load + HF export via zero_to_fp32. | Save + resume round-trip preserves loss within noise. `save_hf_weights=true` produces a HF-format model that loads with `from_pretrained`. |
| **P4** | DeepSpeedCPUAdam optimizer option + perf tuning (overlap, contiguous grads). | Step time improvement vs P1 measured and recorded. |

P0 + P1 unblock the local 35B-A3B work. P2-P4 are improvements.

## 6. Risks / open questions

1. **MoE compatibility.** veomni's MoE path (`Qwen3_5MoeForCausalLM`) uses
   custom expert parallelism. ZeRO-3 sharding may conflict with the expert
   layout. Mitigation: start with `expert_parallel_size=1`; document
   ZeRO+EP combos as a follow-up.

2. **Cached teacher inside DS engine.** The Repr-Align teacher path holds
   `self.teacher_model` as a submodule. DS may try to shard it. Mitigation:
   add `teacher_model` and `CachedTeacher` to a DS `parameters_to_ignore`
   list, OR detach the teacher as an external module not part of the
   engine.

3. **`init_device: meta` interaction.** DS has its own meta-init path
   (`zero.Init()` context). We need to either route through DS's path or
   ensure veomni's existing meta-load runs before `deepspeed.initialize`.
   Mitigation: spike in P0.

4. **NVMe offload bandwidth.** 12 TB drive's sequential read is ~3 GB/s.
   For ZeRO-3 per-layer param swap on a 70 GB model with ~40 layers,
   that's ~1.75 GB/layer / 3 GB/s = ~0.6 s/layer just for I/O on the
   critical path. Step time likely dominated by NVMe. Mitigation: measure
   in P2; if too slow, increase `aio.block_size`/`aio.queue_depth` or
   accept CPU-only as the cap.

5. **DeepSpeed + torchao quantized tensors.** Not tested. The
   `_dequantize_for_save` shim in checkpointer.py works for DCP; need to
   verify DS save calls into the same `state_dict()` path. Mitigation:
   smoke in P3.

6. **Two parallel codepaths** = maintenance cost. Every future change to
   the training loop touches both. Mitigation: keep DS isolated in
   `deepspeed_init.py` and gate it behind `data_parallel_mode == "deepspeed"`
   so the FSDP path is unchanged; revisit if DS becomes the default.

## 7. Out-of-scope alternatives considered

- **FSDP2 + NVMe offload.** Not in stable torch as of 2.5. Re-evaluate when
  torch 2.7 ships.
- **TBA / FlashTrain** (arxiv 2408.10013). Offloads activations, not
  weights — wrong axis for our bottleneck. Research prototype with complex
  install (kvikio nightly, GPUDirect-Storage, Apex from source).
- **vLLM-style paged attention for training.** Designed for inference KV
  cache, not training memory.

## 8. Open decisions

- Pin which DeepSpeed version? (Current latest is 0.16.x; verify Zero-Infinity
  bf16 path is intact.)
- Default `ds_nvme_path`? Probably leave null and force user to set it
  (avoids accidental writes to system disk).
- Do we want to expose `ds_config_path` escape hatch in P0 or defer? Probably
  P0 — power users will want to hand-tune.
- Resume from FSDP-saved checkpoints? **No** — different layouts. Document a
  one-time bf16 export → DS load path for migration.

---

**Effort estimate (calendar time, single engineer):** P0 ~2 days, P1 ~1 day,
P2 ~2 days (mostly NVMe tuning), P3 ~1-2 days, P4 ~1 day. Total ~1 week
to get a working 35B-A3B local training loop with NVMe-offload.
