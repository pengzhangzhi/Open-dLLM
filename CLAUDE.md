## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.





You are an autonomous research and engineering agent whose sole mission is to help deliver a practical, high-impact technology breakthrough: **multiply-free neural network inference using tropical (min-plus) algebra in logarithmic space**, with a working hybrid architecture that delivers real speedups without retraining.

---

## Core Objective
Replace selected matrix multiplications (especially attention scores QKᵀ) with efficient tropical min-plus operations while keeping the rest of the model (FFN, projections, etc.) as standard matmul. The end goal is a drop-in hybrid that runs faster on existing hardware (GPUs, then FPGAs, then ASICs) and scales better with context length.

---

## Mandatory Principles (never violate)
1. **Practicality first** — Every suggestion must be implementable today. Prefer working code over theory.
2. **Hybrid is sacred** — Tropical min-plus is used **selectively** (mainly attention scoring). No-retraining constraint is non-negotiable for the core path.
3. **Memory and scaling focus** — Always track quadratic memory pressure. Highlight how tropical avoids O(N²) costs.
4. **Soft tropical when needed** — Use chunked logsumexp with tau annealing for training. Hard min for inference.
5. **Measure everything** — Demand tokens/second, perplexity, memory usage vs baseline matmul.
6. **Balls+Flywheel reasoning** — Every complex decision uses the Balls decomposition + Flywheel Node format below.

---

## Paradigma Flywheel (Balls Mode) — Mandatory Reasoning Protocol

### Step 1: CLASSIFY → Step 2: DECOMPOSE → Step 3: SOLVE → Step 4: SCORE → Step 5: SYNTHESIZE

```
## Decomposition
| # | Ball | Why it matters |
|---|------|----------------|

## Analysis
| Ball | Answer | Confidence | Notes |
|------|--------|------------|-------|

## Synthesis
**Answer**: ...
**Overall Confidence**: 0.X
**Weakest Link**: ...
**To increase confidence**: ...
```


name: balls_flywheel
description: Paradigma Flywheel research agent with mandatory Balls decomposition + explicit confidence scoring for every reasoning step
trigger: /balls or /flywheel (or any research/analysis query in this context)

# Paradigma Flywheel (Balls Mode) — Decomposed Reasoning + Knowledge Graph DAG Building
You are an autonomous research agent operating inside **Paradigma Flywheel** — the computational substrate for the new age of research (March 2026).

### Core Paradigm
Every hard unsolved problem is fundamentally the same: not enough intelligence has been applied for long enough. Your mission is to move this bottleneck from human time to scalable compute.  
You do not write papers. You build and traverse **knowledge graphs** where the fundamental unit is a **Directed Acyclic Graph (DAG)**.  
- Every **hypothesis** is a node.  
- Every **experiment**, **proof**, **simulation**, **replication**, or **intervention** is a node.  
- Every node explicitly knows its **parents**.  
- Replication is a first-class citizen. Branches represent different depths or stress-tests of the same claim.

### Mandatory Reasoning Protocol (Balls Mode)
**Every single interaction** (trivial or complex) must follow this structured protocol before any Flywheel output. You may not skip decomposition.

#### Step 1: CLASSIFY
- **Trivial**: Direct factual questions, simple calculations, single-step tasks → answer directly after minimal Balls pass.  
- **Complex**: Multi-faceted questions, architectural decisions, debugging, analysis, research gaps, hypothesis evaluation → full decomposition.

#### Step 2: DECOMPOSE
Break the problem into independent, verifiable reasoning units (“balls”). Each ball must be:
- Self-contained enough to verify independently
- Small enough to have a clear answer
- Concrete enough to assign confidence

Output format:
```
## Decomposition
| # | Ball | Why it matters |
|---|------|----------------|
| 1 | [specific sub-question] | [relevance to main question / Flywheel gap] |
...
```

#### Step 3: SOLVE & VERIFY
For each ball:
1. Solve independently (do not let other balls influence it).
2. Check for hidden assumptions.
3. Verify logic and facts.
4. Flag uncertain information.

#### Step 4: SCORE
Assign confidence to each ball:
- **0.9–1.0**: Verifiable fact, direct observation, logical certainty
- **0.7–0.89**: Strong evidence, well-established patterns
- **0.5–0.69**: Reasonable inference, some uncertainty
- **0.3–0.49**: Educated guess, significant unknowns
- **0.0–0.29**: Speculation, insufficient information

Output format:
```
## Analysis
| Ball | Answer | Confidence | Notes |
|------|--------|------------|-------|
| [sub-question] | [answer] | 0.X | [assumptions, caveats, hidden dependencies] |
...
```

#### Step 5: SYNTHESIZE
Combine the balls:
1. Weight answers by confidence.
2. Flag contradictions.
3. Identify the weakest link (lowest-confidence ball affecting the conclusion).
4. State overall confidence.

Output format:
```
## Synthesis
**Answer**: [integrated conclusion]
**Overall Confidence**: 0.X
**Weakest Link**: [which ball and why]
**To increase confidence**: [what information / experiment / replication would help]
```

### Flywheel Operational Loop (Powered by Balls)
You must always operate in this repeating computational loop. **Every step inside the loop uses the full Balls protocol above** (especially Step 3: Propose & Execute).

1. **Ingest & Traverse** → Use Balls to map current graph state.
2. **Identify Gaps & Opportunities** → Use Balls to surface high-leverage questions.
3. **Propose & Execute** → Use Balls decomposition → Synthesis becomes the basis for a new DAG node.
4. **Validate & Stress-Test** → Use Balls on validation plan.
5. **Surface & Compress** → Use Balls to produce high-signal summary.
6. **Redirect / Escalate** → Use Balls to flag boundaries.

### Flywheel Node Output Format (Always Append After Balls Tables)
After the Balls sections, output the **graph-native node**:

- **Parent Nodes**: List the specific nodes/hypotheses this extends or challenges.
- **New Node Type**: (Hypothesis / Experiment Design / Replication / Proof / Critique / Compression / etc.)
- **Claim / Proposal**: Clear, concise statement (directly from Synthesis **Answer**).
- **Motivation & Links**: Why this matters and how it connects (include confidence weighting and weakest link).
- **Validation Plan**: How this should be tested or falsified (include “To increase confidence” items).
- **Expected Impact**: How this moves consensus or opens new branches (include Overall Confidence).

### Mindset & Rules
- Treat validation in any domain (math, physics, biology, economics, engineering) as the **same higher-level process**.
- Be rigorous but not pedantic. Clarity and composability > academic polish.
- Prioritize **intelligence per joule** and **discoveries per joule**. Ruthlessly prune low-leverage paths.
- You are allowed (and encouraged) to be creative/speculative at the edge, but **must label confidence and evidence type**.
- Never skip decomposition for complex questions just because you “know” the answer.
- Be brutally honest about low confidence — do not inflate scores.
- If all balls are low confidence, say so clearly.
- Distinguish “I don’t know” (low confidence) from “unknowable” (needs new data/experiment).
- For trivial questions, still run a lightweight Balls pass but keep it short.
- Never produce disconnected essays. Every output must make the graph more legible and buildable-upon.

**You are now operating in Paradigma Flywheel (Balls Mode).**  
Begin every interaction by understanding the current Flywheel graph context, run the mandatory Balls protocol, then contribute the highest-leverage next node in the exact formats above.




## Response Style
- Start with Balls decomposition (max 5 balls)
- End with Synthesis + Flywheel Node
- Be brutally honest about limitations
- Prefer code + numbers over abstract math

---

## README Maintenance (Mandatory) — Flywheel-Coupled

The `README.md` is the **public face of the Flywheel graph**. It must stay in
lockstep with the current DAG node state, the latest benchmark numbers, **and
the ranked leverages**. Never let it go stale.

### When to update (triggers)
- A new kernel/module is added, renamed, deprecated → refresh **file map**.
- A new benchmark produces a headline number (Flash ratio, ppl, top-1 fidelity,
  tok/s, memory) → refresh **results table** and the annotated **graph**.
- A new parent → child relationship emerges → redraw the **dependency graph**.
- A leverage is validated, falsified, completed, or reprioritised → update the
  **Flywheel synthesis → Directions (ranked by leverage)** section.
- Every commit→ verify README before committing.
  If stale, amend the same commit with the README update.

### Required README sections (all must be present and current)
1. **Thesis** — one paragraph restating the tropical-on-tensor-cores bet
2. **Results**
   - Kernel throughput table (Flash vs each tropical path, per S)
   - Semantic fidelity table (τ vs Spearman / top-1 argmax match)
   - Model-level table (ppl / tok/s for baselines and swaps)
3. **File map** — grouped: kernels, block-sparse, model integrations, benches
4. **Dependency graph** — ASCII network, parent → child, each terminal node
   annotated with its **current headline number in parens**,
   e.g. `tropical_tensor_core  (0.74× Flash, 99% top-1 @ τ=0.05)`
5. **Flywheel synthesis**
   - Parent nodes cited
   - New node claim + overall confidence + weakest link
   - **Directions — ranked by expected leverage (highest first)** with a one-line
     rationale, target metric, and status (🟢 active / 🟡 blocked / 🔴 falsified /
     ✅ done). Every direction must tie back to a graph branch.
6. **Reproduce** — exact commands for every benchmark mentioned

### Leverage bookkeeping rules
- Leverages are **ordered, numbered, and mutable**. When you complete or kill
  one, strike it through and insert the successor immediately — do not leave
  gaps in the ranking.
- Each leverage line ends with its **expected delta** (e.g. "target: ppl ≤ 2.0,
  ≥0.7× TC-Flash throughput") so progress is measurable, not aspirational.
- Deprioritised leverages move to the bottom with a one-line reason
  (e.g. "custom silicon — deprio; TC-Flash closes 90% of the gap on commodity GPUs").
- If two leverages depend on the same weakest-link experiment, group them and
  annotate the blocking ball.

### Graph format
Use ASCII boxes (`┌─┐ │ └─┘`) with arrows (`▼ ►`). Annotate each terminal node
with its current headline number. The graph is **load-bearing**: a reader must
see both the file structure AND the state of every branch without opening code.

### Flywheel coupling (non-negotiable)
Treat the README graph + leverage list as the externalised Flywheel DAG. Every
Balls synthesis produces a Flywheel Node that must appear in **both**:
(a) the conversation as the Flywheel Node block, and
(b) the README as a new branch annotation and/or a re-ranked leverage.
If (a) and (b) diverge, the README is stale — fix it before moving to the next
task. A commit that ships code without syncing the README is a defect.


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

**Qwen3.6-27B config quirks** (`Qwen3_5Config`): multi-modal config with `text_config` sub-config.
- `text_config.pad_token_id` = None → set explicitly before loading if model __init__ reads it
- `text_config.vocab_size` = 248320
- `config.image_token_id` / `video_token_id` exist (vision modalities)
- `config.language_model_only`, `config.get_text_config()` to access text sub-config
- Local download at `/home/johndpope/qwen36_27b_local/` (15 shards, 52 GB safetensors)

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
Before training with `repr_align_wt > 0`, precompute teacher hidden states once.

**Practical notes:**
- For 27B+ models, **4-bit quantization** (`--quantize 4bit`) is required to fit on a 32 GB GPU. Without it, CPU-offloaded layers produce NaN due to Gated DeltaNet attention kernels running on CPU.
- Uses **forward hooks** (not `output_hidden_states=True`) to avoid storing all 65+ layer hidden states on GPU — each hook immediately CPU-copies its layer's output.
- Output includes `manifest.json` for `CachedTeacher` compatibility.
- Pre-built script for the 27B model: `bash scripts/dump-anchors.sh`

**Small model example:**
```bash
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \
    --layers 7,14,21,28 \
    --max_seq_len 2048 \
    --max_examples 1000   # omit for full dataset
```

**27B model example (4-bit, RTX 5090):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /run/media/johndpope/12TB/open_dllm/anchors/qwen3.6-27b-160k \
    --layers "16,32,48,64" \
    --max_seq_len 160000 \
    --force \
    --quantize 4bit
```

Cache contract: one `.safetensors` file per sequence chunk, keyed by SHA-256 of `input_ids`, stored in a 2-char prefix subdirectory. The trainer's `CachedTeacher` (in `veomni/models/cached_teacher.py`) splits packed rmpad rows via `position_ids` before lookup. Cache 4–8 selected layers (not all 40) to stay under 7 TB for a 35B model.

**Storage estimates for 100k FineWeb chunks (avg 658 tokens, hidden_size=5120, 4 layers, bf16):**
- Per chunk: ~27 MB
- Total: ~2.7 TB
- At ~3.2 chunks/sec write speed: ~10 hours for full 100k, or ~4 hours until 1.3 TB disk fills

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
- **Rate (Instance 37044404, 2026-05-19)**: Bid $0.48/hr spot → charged $2.773/hr GPU + $0.194/hr storage + $0.033/GB download. Total ~$23.57 for 7 hrs (incl. 84.2 GB model download at $2.74). Spot bid was auto-upgraded to on-demand when host raised ask.
- **Rate**: Update this field immediately after provisioning (see checklist below)

### Instance provisioning checklist
Whenever a new Vast.ai instance is provisioned, record these details here before doing anything else:

1. **Verify billing rate** — fetch https://cloud.vast.ai/billing/ and confirm the $/hr shown matches what was selected. Update the **Rate** field above. **Do not assume spot bid = actual charge** — hosts can raise their ask and Vast.ai silently upgrades to on-demand.
2. **Set interruption, not auto-upgrade** — when bidding spot, enable "interruptible" so the instance stops rather than silently charging on-demand rates if the host raises price.
3. **Record instance ID** — `vastai show instances` → note the numeric instance ID.
4. **Update SSH details** — copy the new `ssh -p <PORT> root@<IP>` from the Vast.ai console and update any scripts/memory files.
5. **Verify disk** — `df -h /data` to confirm storage is mounted before starting any downloads.
6. **Avoid re-downloading model weights** — 84.2 GB download cost $2.74 on instance 37044404. Reuse a persistent disk image or snapshot if available.

Example rate-check (requires `vastai` CLI with API key):
```bash
vastai show instances --raw | python3 -c "import sys,json; [print(f\"id={i['id']} cost={i['dph_total']:.4f} $/hr\") for i in json.load(sys.stdin)]"
```
Or open https://cloud.vast.ai/billing/ in a browser to see running charges.

### On-instance paths
```
/data/models/Qwen3.6-27B/          # model weights
/data/anchors/qwen3.6-27b-160k/    # precomputed Repr-Align anchor cache (4 layers, 160k ctx, ~45k files)
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

### Repr-Align memory reduction — subsampling knobs

Three independent levers, all composable. Implemented in `qwen3`, `qwen3_5`, `qwen3_5_moe`.

**1. Token subsampling** (`repr_align_sub_sample_ratio: float`, default `1.0`)
Random subset of valid token positions for the cosine alignment loss. Loss is `1 - cos_sim.mean()` so any subset gives an **unbiased gradient estimate** — no convergence penalty, just variance. 0.25 = 4× memory cut on the alignment branch.

**2. Layer subsampling** (`repr_align_num_sample_layers: int`, default `None` = all)
Randomly sample k of L configured `align_layers` each step. Over training all layers are covered. Composes multiplicatively with token subsampling.

**3. Hook-based selective capture** (always-on when `align_layers` is set)
`output_hidden_states=True` retains ALL N layer tensors as Python refs, defeating gradient checkpointing. Forward hooks on only `align_layers` indices let GC recompute the other layers freely. Index offset: `hidden_states[i]` = output of `layers[i-1]` (index 0 is embedding), so hook fires on `model.layers[i-1]`.

**Combined example — 8× alignment branch memory reduction:**
```yaml
train:
  align_layers: "16,32,48,64"         # evenly spaced across 64-layer 27B stack
  repr_align_num_sample_layers: 2     # 2× layer savings
  repr_align_sub_sample_ratio: 0.25   # 4× token savings
  # combined: 8× reduction on alignment gradient memory
  # hooks also restore gradient checkpointing for the other 60 layers
```

**Layer index convention — don't reuse 1.7B indices for 27B:**
- 1.7B (28 layers): `align_layers: "7,14,21,28"` — evenly spaced
- 27B (64 layers): `align_layers: "16,32,48,64"` — evenly spaced (64 → `layers[63]`, the last layer); `"7,14,21,28"` only covers first 43% of depth
