# Local Training — Hardware Options

This doc tracks the local hardware available for training Open-dLLM (Repr-Align path) on Qwen3.6, and the upgrade / split-compute paths that make a 35B-A3B run feasible.

## Available Machines

### HP Z6 G4 (`hp-z6.local` → `192.168.1.101`)

Verified via SSH `dmidecode` 2026-05-17.

| Component | Spec |
|---|---|
| Chassis | HP Z6 G4 Workstation (BIOS P60 v02.94, 2024-05-17) |
| CPU | **1× Xeon Silver 4108** (Skylake-SP, 1st-gen Xeon SP, 8c / 16t @ 1.8 GHz) |
| RAM | 48 GB DDR4 ECC RDIMM, mixed kit, **all throttled to 2133 MT/s** |
| DIMM slots | 6 total (1 DPC), 5 populated, slot CPU0-DIMM4 empty |
| GPU 0 | NVIDIA RTX 3090 (24 GB GDDR6X, Ampere, BF16 OK) |
| GPU 1 | NVIDIA Quadro P2000 (5 GB, Pascal — display only, can't participate in training) |
| Storage | Samsung 980 Pro 1TB NVMe (Gen3 link due to Skylake) |
| External | `/run/media/johndpope/12TB/` data + checkpoints drive |
| Network | 1 GbE built-in (`enp5s0`), Tailscale |
| OS | Ubuntu 25.10, kernel 6.14 |

**Platform limits — important**:
- Skylake-SP **does not support Optane PMEM at all**. CPU upgrade to Cascade Lake-SP M/L SKU (e.g. 8260M) is mandatory for any PMEM plan.
- Z6 G4 is **1 DPC** (one DIMM per channel). Even with PMEM-capable CPU, **Memory Mode PMEM is impossible** — only App Direct + Linux kmem tiering.
- Max DDR4 with single CPU: 768 GB (6× 128 GB LRDIMM).
- PCIe Gen3 throughout — link to GPU caps at 16 GB/s.

### MSI Box (`msi`, current working machine)

Verified locally 2026-05-18.

| Component | Spec |
|---|---|
| Chassis / Board | MSI MPG Z690 CARBON WIFI (MS-7D30), AMI BIOS 1.91 (2022-10-11) |
| CPU | i5-13600KF (14c / 20t, Raptor Lake) |
| RAM | 96 GB DDR5 (Z690 ceiling 128 GB) |
| GPU 0 | RTX PRO 4000 Blackwell (24 GB), PCIe Gen 5 ×8 under load |
| GPU 1 | RTX 5090 (32 GB), PCIe Gen 5 ×8 under load |
| Combined GPU memory | 56 GB |
| `nvme0n1` | Samsung 990 Pro 2 TB — **OS drive**, 1.3 TB free, used for `~/ds_offload` cache for now |
| `nvme1n1` | Samsung 990 Pro 2 TB — **user data, do not touch** (mounted `/run/media/johndpope/2TB`) |
| `sda` | Seagate IronWolf 12 TB SATA — **currently 100% full**, ~250 MB/s (NAS HDD) |
| Network | LAN + Tailscale |
| OS | Ubuntu 26.04 LTS, kernel 7.0 |

**Notes**:
- Z690 is a consumer chipset → **no Optane PMEM support**. PMEM is Cascade Lake-SP / Ice Lake-SP only.
- Both PCIe slots are ×8 (Z690 CPU has 16 Gen5 lanes bifurcated between two GPU slots); still Gen5 = 32 GB/s per card.
- Repr-Align teacher cache lives on `nvme0n1` at `/home/johndpope/ds_offload/anchors/...` until a dedicated 1 TB NVMe is dropped in.
- 12 TB HDD needs cleanup before it's useful for cache or checkpoints (Trash alone is 1.6 TB).

## Smoke Test Procedure (Qwen3-1.7B, MSI)

Validates the precompute-anchor + cached-teacher pipeline end-to-end at $0 cost on existing hardware before committing to a 35B-A3B run.

**Step 1 — precompute teacher anchors** (5090, ~minutes for 1000 examples):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \
    --layers 7,14,21,28 \
    --max_seq_len 2048 \
    --batch_size 8 \
    --max_examples 1000
```

Writes ~1000 `.safetensors` shards (~8 MB each = ~8 GB total for 4-layer cache) plus a `manifest.json` capturing the cache contract (model, tokenizer, hidden size, layers).

**Step 2 — train student against cached anchors** (5090, ~hours):

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/qwen3_1_7b_repr_align_smoke.yaml
```

Config: `configs/pretrain/qwen3_1_7b_repr_align_smoke.yaml`. Outputs to `/home/johndpope/ds_offload/checkpoints/qwen3-1.7b-repr-align-smoke`.

**What to watch on wandb**:
- `loss_components/mdm` — should decrease from random init level
- `loss_components/repr_align` — should *decrease* (cosine distance shrinks → alignment grows toward 1.0)
- `loss_components/teacher` — N/A (CachedTeacher returns no loss)
- Total `loss` — sum of mdm + repr_align * 1.0

**What success looks like**: both mdm and repr_align losses trend down across a few hundred steps on the 1000-example set. It will overfit (1000 examples is tiny), that's expected — the goal is "pipeline runs and the loss signal is correct", not convergence on real data.

**Why this validates the bigger story**:
- `CachedTeacher` correctly substitutes for `copy.deepcopy(model)` (loader path)
- `align_layers` subsetting works on both student and teacher hiddens (modeling path)
- Hash-based cache lookup matches across precompute and train (CachedTeacher correctness)
- Standard FSDP / mixed precision / gradient checkpointing all still work
- Once this passes, the same code paths run for Qwen3.6-35B-A3B — only the hardware constraint differs

## Memory Budget — 35B-A3B Repr-Align (Full Fine-tune)

```
Student BF16 params         70 GB
Student BF16 grads          70 GB
Student FP32 master        140 GB
Adam m (FP32)              140 GB
Adam v (FP32)              140 GB
Teacher BF16 (frozen)       70 GB   ← can live on a different machine
Activations (checkpointed)  ~20 GB
──────────────────────────────────
Total state               ~650 GB
Student-only state        ~580 GB
```

No machine on hand can fit this on GPU alone. CPU/disk offload mandatory.

## Architecture: Split-Compute (Anchor Precompute + Student Train)

Because Repr-Align is **realignment, not distillation**, the "teacher" is a frozen snapshot of the student's own init. Its hidden states for any input are deterministic and never change for the whole run → **cache once, read forever**.

```
ONE-TIME (on MSI / 5090):
  load Qwen3.6-35B-A3B
  for batch in FineWeb 100K:
      h = qwen3_6(batch.input_ids, output_hidden_states=True)
      save_to_disk(batch.id, h[selected_layers])   # → 12TB drive

EVERY STEP (on Z6 or rented box):
  batch = dataloader.next()
  anchor_h = mmap_read(batch.id)                   # ~ms, NVMe
  student_h = student(batch.input_ids)
  loss = mdm_loss + λ · cosine_align(student_h, anchor_h)
```

**Cache size budget** (FineWeb 100K @ seq_len 2048, hidden_dim 2048, BF16):

| Layers cached | Size | Fits on 12 TB? |
|---|---|---|
| 1 layer | 840 GB | ✅ trivially |
| **4 layers (10/20/30/39)** | **3.4 TB** | **✅ recommended** |
| 8 layers (every 5th) | 6.7 TB | ✅ |
| All 40 | 33.6 TB | ❌ |

Repr-Align literature suggests 3–8 mid/late layers retain most of the alignment signal vs all-layer.

**Why not live RPC between Z6 and MSI**: anchor outputs are static — recomputing them every step is pure waste. Precompute also decouples the two machines: MSI can be powered off during the actual training run.

## Upgrade Paths for the Z6

### Path A — Quick win: matched RDIMM, keep current CPU

| Item | Cost (AUD est.) |
|---|---|
| 6× 32 GB DDR4-2666 ECC RDIMM matched | $300–600 |
| **Total** | **$300–600** |
| Result | 192 GB @ full 2666 MT/s, ~25% bandwidth gain |
| 35B-A3B local? | ❌ Still way short of 580 GB student state |
| Use case | General Z6 perf bump; doesn't change training feasibility |

### Path B — Pure DDR4 max-out (single CPU)

| Item | Cost (AUD est.) |
|---|---|
| 6× 128 GB DDR4-2666 ECC LRDIMM | $1,800–3,600 |
| Keep Xeon Silver 4108 | $0 |
| **Total** | **$1,800–3,600** |
| Result | 768 GB DRAM, no PMEM, simple setup |
| 35B-A3B local? | ✅ Fits 580 GB student with ~190 GB headroom |
| Caveat | 8 cores → optimizer step is CPU-bound and slow (~30–60s/step) |

### Path C — CPU + DDR4 max (recommended for serious local training)

| Item | Cost (AUD est.) |
|---|---|
| Xeon **8260M** (Cascade Lake-SP, 24c/48t, PMEM-capable) | $300–600 |
| 6× 128 GB DDR4-2933 ECC LRDIMM | $1,800–3,600 |
| **Total** | **$2,100–4,200** |
| Result | 24 cores @ 2.4 GHz, 768 GB DDR4 @ full 2933 |
| 35B-A3B local? | ✅ Best simple path. ~60–90 s/step, ~10 days/epoch |
| Notes | DDR4-2933 only at full speed with Cascade Lake; Skylake caps at 2666 |

### Path D — Optane via App Direct (complex, mostly not worth it)

| Item | Cost (AUD est.) |
|---|---|
| Xeon 8260M (mandatory — Skylake can't address PMEM) | $300–600 |
| 3× 256 GB Optane PMEM 100 (e.g. NMA1XBD256GQS) | $900–1,800 |
| 3× 128 GB DDR4-2933 LRDIMM | $900–1,800 |
| **Total** | **$2,100–4,200** |
| Result | ~384 GB DRAM + ~768 GB PMEM as kmem NUMA node = ~1.15 TB |
| Caveat | Memory Mode impossible on 1-DPC Z6 G4. App Direct only. Setup via `ipmctl`/`ndctl`/`daxctl`. Optane is discontinued. |
| Worth it? | Roughly same price as Path C, more capacity, way more complexity. **Pick C unless you need >768 GB local RAM**. |

### Path E — Switch chassis for Memory Mode

| Chassis | Used AUD | Why |
|---|---|---|
| HP Z8 G4 | $1,200–3,000 | Z6's 2-DPC sibling, 24 slots, true Memory Mode |
| Dell Precision 7920 Tower | $900–2,500 | Best value workstation route |
| Lenovo ThinkStation P920 | $1,000–2,800 | 16 slots, 2-DPC |
| Dell R740 / HP DL380 Gen10 (rack server) | $400–900 | Cheapest path but loud / needs rack |

Only worth it if you want Memory Mode PMEM at scale (≥1 TB).

### Path F — Don't upgrade, rent

| Item | Cost (USD) |
|---|---|
| 8× H100 80 GB on Lambda / RunPod, ~2 days for one full FineWeb 100K epoch | $300–500 |
| Cumulative break-even vs Path C (~$2-3K) | 5–10 full training runs |

Right answer if you're doing one-shot validation. Right answer if you want to defer hardware commit until pipeline is proven.

## Decision Tree

```
Want to train Qwen3.6-35B-A3B Repr-Align locally?
│
├─ "Just validate the pipeline works" → smoke test on Qwen3-4B with current Z6 ($0)
│                                       no upgrade needed; runs on 3090 + 48 GB RAM
│
├─ "Run once, get a real model" → Path F (rent 8× H100, $300–500/epoch)
│
├─ "Iterate on Repr-Align continuously for weeks/months locally" → Path C
│                                                                  ($2-4K, ~10 days/epoch)
│
└─ "Want max local memory for multi-model inference too" → Path E
                                                          (chassis swap, $4-6K total)
```

## Open Questions

- [ ] Pull SSH diagnostics from the MSI box (CPU, RAM, PCIe gen, NVMe layout)
- [ ] Measure `iperf3` bandwidth between Z6 and MSI (decides if any cross-machine streaming is even worth considering)
- [ ] Run smoke test on Qwen3-4B Repr-Align with current Z6 hardware — validates entire training pipeline at $0 cost before any hardware decision
- [ ] If Path C committed: identify specific 8260M and 128 GB LRDIMM listings on AU eBay; verify Z6 G4 BIOS revision recognises 8260M (P60 v02.94 should, but confirm against HP QuickSpecs)
- [ ] If Path A interim: spec exact matched 6× 32 GB RDIMM kit; current 5× mixed kit forces 2133 MT/s

## Related Docs

- `docs/representation_alignment.md` — Repr-Align method
- `docs/cola_ldm.md` — Cola DLM architecture (parked)
- `docs/cola_ldm_roadmap.md` — validation experiments, pathways
