
# 🔥 Open-dLLM: Open Diffusion Large Language Models

🌍 Languages: [English](README.md) | [中文](README_cn.md) | [日本語](README_ja.md)

👉 TL;DR: **Open-dLLM** is the most open release of a diffusion-based large language model to date —  
including **pretraining, evaluation, inference, and checkpoints**.  


#### Representation Alignment

Open-dLLM supports **representation alignment** for adapting autoregressive LMs into diffusion LMs with 4x speedup. This feature is based on our recent paper, [**Don’t Retrain—Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment**](https://arxiv.org/pdf/2605.06885). Check out [Representation Alignment Tutorial](docs/representation_alignment.md).


<p align="center">
  <a href="https://github.com/pengzhangzhi/Open-dLLM">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40" alt="GitHub"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg" width="40" alt="Notion"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/fredzzp/open-dcoder-0.5B">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40" alt="Hugging Face"/>
  </a>
</p>

<p align="center">
  <b>💻 Code</b> &nbsp; | &nbsp; <b>📖 Blog</b> &nbsp; | &nbsp; <b>🤗 Model</b>
</p>


## 🎥 Demo

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif" 
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center"><i>QuickSort generation using Open-dCoder (0.5B)</i></p>

<p align="center">
  <a href="https://youtu.be/d8WrmvUhO9g">
    <img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube" alt="YouTube link"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.bilibili.com/video/BV1ZveSz3E1J/">
    <img src="https://img.shields.io/badge/Bilibili-视频-blue?logo=bilibili" alt="Bilibili link"/>
  </a>
</p>

---

## ✨ Highlights

- 🏋️ **Pretraining pipeline + open datasets**  
- ⚡ **Inference scripts** — easy sampling & generation  
- 📊 **Evaluation suite** — HumanEval, MBPP, Infilling (lm-eval-harness + custom metrics)  
- 📦 **Weights + checkpoints** on Hugging Face  
- 🤝 **Transparent configs** for full reproducibility  

---

## Why Open-dLLM?

Most diffusion LLM repos (e.g., LLaDA, Dream) only release **inference scripts + weights**, which limits reproducibility.  
**Open-dLLM** is the first to open-source the **entire stack** for diffusion LLMs.

👉 With Open-dLLM, you can go from **raw data → training → checkpoints → evaluation → inference**, all in one repo.

---

## 🔎 Transparency Comparison of Diffusion LLM Releases

| Project                                                                 | Data | Training Code | Inference | Evaluation | Weights |
|-------------------------------------------------------------------------|:---:|:-------------:|:---------:|:----------:|:-------:|
| **Open-dLLM / Open-dCoder (ours)**                                      | ✅  | ✅            | ✅        | ✅         | ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                               | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)    | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)             | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |

✅ = fully available · ❌ = not provided · ⚠️ = partial/limited

---

## ⚙️ Install

We use `micromamba` for environment management (feel free to adapt to `conda`):

```bash
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# install the newest torch with cu121
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8
# optional
pip install pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .
pip install lm-evaluation-harness/ human-eval-infilling/
````

---

## 🚀 Quickstart: Sampling

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# Prompt
prompt = "Write a quick sort algorithm in python."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generation config
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 For full logging, history tracking, and file output:

```bash
python sample.py
```

---

## 📊 Benchmarking

We release a fully open-source **evaluation suite** for diffusion-based LLMs (dLLMs), covering both **standard code generation tasks** and **code infilling tasks**.

Benchmarks include: **HumanEval / HumanEval+**, **MBPP / MBPP+**, **HumanEval-Infill**, **SantaCoder-FIM**.

---

#### Standard Code Generation

| Method                       | HumanEval |          | HumanEval+ |          | MBPP     |          | MBPP+    |          |
| ---------------------------- | --------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- |
|                              | Pass\@1   | Pass\@10 | Pass\@1    | Pass\@10 | Pass\@1  | Pass\@10 | Pass\@1  | Pass\@10 |
| LLaDA (8B)                   | 35.4      | 50.0     | 30.5       | 43.3     | 38.8     | 53.4        | 52.6     | 69.1        |
| Dream (7B)                   | 56.7      | 59.2     | 50.0       | 53.7     | 55.4     | 56.2        | 71.5     | 72.5        |
| Mask DFM (1.3B)              | 9.1       | 17.6     | 7.9        | 13.4     | 6.2      | 25.0     | –        | –        |
| Edit Flow (1.3B)             | 12.8      | 24.3     | 10.4       | 20.7     | 10.0     | 36.4     | –        | –        |
| **Open-dCoder (0.5B, Ours)** | **20.8**  | **38.4** | **17.6**   | **35.2** | **16.7** | **38.4** | **23.9** | **53.6** |

> *Despite being only 0.5B parameters, Open-dCoder competes with much larger dLLMs in code completion tasks.*

---

#### Code Infilling

| Method                                | HumanEval Infill Pass@1 | SantaCoder Exact Match |
| ------------------------------------- | ----------------------: | ---------------------: |
| LLaDA-8B                              |                    48.3 |                  35.1  |
| Dream-7B                              |                    39.4 |                  40.7  |
| DiffuCoder-7B                         |                    54.8 |                  38.8  |
| Dream-Coder-7B                        |                    55.3 |                  40.0  |
| **Open-dCoder (0.5B, Ours)**          |                    32.5 |                  29.6  |
| **Open-dCoder (0.5B, Ours)** Oracle Length |               77.4 |                  56.4  |

> *We followed the average fixed length evaluation setting in [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/) to get the results.*

---

## 🧪 Evaluation

Install evaluation packages:

```bash
pip install -e lm-evaluation-harness human-eval-infilling
```

#### Code Completion (HumanEval, MBPP)

```bash
cd eval/eval_completion
bash run_eval.sh
```

#### Code Infilling

```bash
cd eval/eval_infill
bash run_eval.sh
```

---

## 🏋️ Pretraining

* **Data**: Concise, high-quality code corpus [**FineCode**](https://huggingface.co/datasets/fredzzp/fine_code), hosted on Hugging Face.
* **Initialization**: Following *Dream*, continued pretraining from **Qwen2.5-Coder**, adapting it into the diffusion framework.
* **Loss**: Masked Diffusion Model (MDM) objective — masking ratios uniformly sampled from `[0,1]`, reconstructed with cross-entropy loss.

### Download Data

```bash
python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
```

### Training

```bash
export TOKENIZERS_PARALLELISM=false
NNODES=1
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}



torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=data/data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=16 \
  --train.global_batch_size=512 \
  --train.output_dir=logs/Qwen2.5-Coder-0.5B_mdm \
  --train.save_steps=10000
```
example of multi-node training with repr alignment loss:
```bash

export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:=1}
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK   --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT  tasks/train_torch.py \
configs/pretrain/qwen2_5_coder_500M.yaml --data.train_path=data/data \
--data.num_workers=0 \
--data.prefetch_factor=1 \
--train.ckpt_manager=dcp \
--train.micro_batch_size=3 \
--train.global_batch_size=240 \
--train.repr_align_wt=10.0 \
--model.model_path=Qwen/Qwen2.5-Coder-3B-Instruct \
--train.save_steps=10000 \
--train.output_dir=logs/Qwen2.5-Coder-3B-Instruct_mdm_repr_align-10
```

### QLoRA Repr-Align (27B on a single 32 GB GPU)

For 27B+ models that don't fit in GPU memory at full precision, use **QLoRA Repr-Align**: NF4 quantized base (frozen) + LoRA adapters (trainable). Fits in ~25 GB VRAM with r=32.

#### How NF4 quantization works

No separate quantization step is needed. `bitsandbytes` quantizes weights on-the-fly during `from_pretrained()` via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. The original bf16 weights on disk are never modified — quantization happens in GPU memory at load time. The 55 GB bf16 checkpoint becomes ~7 GB in VRAM.

#### Step-by-step

1. **Download model weights** (~55 GB):
```bash
huggingface-cli download Qwen/Qwen3.6-27B --local-dir /path/to/Qwen3.6-27B
```

2. **Prepare training data** (plaintext JSONL with a `text` field):
```bash
python -c "
from datasets import load_dataset
import json
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
with open('data.jsonl', 'w') as f:
    for i, ex in enumerate(ds):
        if i >= 100000: break
        f.write(json.dumps({'text': ex['text']}) + '\n')
"
```

3. **Precompute teacher anchor cache** (one-time). This runs the frozen teacher model on your training data and caches hidden states for selected layers. The cached anchors are reused every training step — no live teacher needed during training.

For a **smoke test** (1000 examples, 4 layers, ~2 min):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /path/to/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/qwen3.6-27b \
    --layers 16,32,48,64 \
    --max_seq_len 1024 \
    --max_examples 1000
```

For **production** (100K examples, all 64 layers — recommended for best alignment quality). This requires a GPU with ≥32 GB VRAM or a cloud instance:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /path/to/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/qwen3.6-27b-all64 \
    --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64 \
    --max_seq_len 1024
```

> **Note**: The `--layers` argument uses 1-indexed layer numbers. `--max_examples` limits the number of training examples cached. Omit it to cache the full dataset. The cache is stored as one `.safetensors` file per sequence chunk, keyed by SHA-256 of `input_ids`. Re-running with the same arguments skips already-cached chunks.

4. **Run training** (single 32 GB GPU):
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup .venv/bin/torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/qlorafy_27b_train.yaml \
    > /tmp/qlorafy_train.log 2>&1 &
echo $! > /tmp/qlorafy_train.pid

# Monitor:
tail -f /tmp/qlorafy_train.log
```

Before launching, edit `configs/pretrain/qlorafy_27b_train.yaml` to point to your local paths:
```yaml
model:
  model_path: /path/to/Qwen3.6-27B           # step 1
data:
  train_path: /path/to/data.jsonl              # step 2
  eval_size: 50                               # hold out 50 examples for perplexity eval
train:
  anchor_cache_dir: /path/to/anchors/qwen3.6-27b  # step 3
  eval_every: 100                              # run eval every 100 steps
  wandb_project: your-wandb-project
  wandb_name: qlorafy-27b-run1
```

#### What the config does

| Setting | Value | Why |
|---------|-------|-----|
| `enable_qlorafy: true` | NF4 base + LoRA r=32 | 27B → ~7 GB in VRAM, r=32 fits 32 GB GPU |
| `language_model_only` | Auto-set by qlorafy.py | Loads text-only `Qwen3_5ForCausalLM`, skips 4.7 GB vision encoder |
| `repr_align_wt: 1.0` | Alignment loss weight | Bidirectional adaptation |
| `align_layers: "16,32,48,64"` | 4 of 64 layers | Matches anchor cache; use all 64 for production |
| `repr_align_sub_sample_ratio: 0.25` | 25% of tokens | 4× gradient memory reduction |
| `save_epochs: 0` | Skip DCP checkpoint | DCP can't serialize `Params4bit` |
| `eval_size: 50` | Hold out 50 examples | Perplexity eval every `eval_every` steps |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Required for r=32 | Reduces memory fragmentation on 32 GB GPU |

#### LoRA rank vs VRAM

| Rank | LoRA Params | Trainable % | Fits RTX 5090 (32 GB)? |
|------|-------------|-------------|------------------------|
| 16 | 73M | 0.27% | Yes (24 GB) |
| 32 | 147M | 0.54% | Yes, needs `expandable_segments:True` (28 GB) |
| 64 | 294M | 1.09% | No — OOMs during first forward pass |
| 128 | 587M | 2.17% | No |

#### Training results (RTX 5090, 32 GB, r=32)

| Metric | Value |
|--------|-------|
| Peak VRAM | ~28 GB allocated |
| Speed | ~19 s/step (micro_batch=1, 16 grad_accum) |
| Throughput | ~440 tok/s, MFU ~19% |
| Loss (20 steps) | 5.2 → 5.1 (stabilizing) |
| Grad norm | 59 → 2.4 (rapidly converging) |
| Eval | Perplexity logged to wandb every 100 steps |

> **Checkpoint limitation**: DCP (`torch.distributed.checkpoint`) cannot serialize `Params4bit` objects from bitsandbytes. Set `save_epochs: 0` during training. To save LoRA weights, use `save_hf_weights: true` (exports PEFT adapter weights only, not the NF4 base).
>
> **Wandb metrics logged**: `training/loss`, `training/grad_norm`, `training/lr`, `qlora/grad_norm`, `qlora/param_norm`, `qlora/grad_to_param_ratio`, `eval/loss`, `eval/perplexity`, `flops_achieved(T)`, `flops_promised(T)`, `mfu`, `tokens_per_second`, `system/vram_allocated_gb`, `system/vram_reserved_gb`. Generation probe every 100 steps via `generation/sample`.

### Uploading Checkpoints to Hugging Face

```python
from huggingface_hub import HfApi

REPO_ID = "fredzzp/open-dcoder-0.5B"
LOCAL_DIR = "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=LOCAL_DIR)
```

---

## 🔄 Two Paths to Diffusion: Repr-Align vs. LDLM

Open-dLLM supports **two approaches** for converting an autoregressive LM into a diffusion LM. Which one you choose depends on your compute budget and goals.

### Recommended: Representation Alignment (Light)

**Paper**: [Don't Retrain—Align: Adapting AR LMs to Diffusion LMs via Representation Alignment](https://arxiv.org/html/2605.06885v1)

The key insight: AR models like Qwen already learn strong language representations. You don't need to retrain from scratch — just **preserve** those representations while switching from causal (left-to-right) to bidirectional (any-order) generation.

**How it works**:
1. Load a pretrained AR model (e.g., Qwen3.6-35B-A3B)
2. Flip the attention mask from causal → bidirectional (this is the "student")
3. Keep a frozen copy as the "teacher" (causal attention, clean input)
4. Train with two losses:
   - **Masked denoising loss**: Randomly mask tokens → student predicts them using bidirectional context
   - **Representation alignment loss**: Cosine similarity between student and teacher hidden states at every layer

**Why it's faster**:
- No new architecture to train — uses the existing model weights directly
- **3-4× faster convergence** vs. training from scratch (per the paper)
- Works on tiny datasets (as low as 0.8B tokens)
- Optional `freeze_layers: "mlp"` gives ~2× throughput with minimal quality loss

**Quick start** (2 GPUs):
```bash
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
  --model.model_path=Qwen/Qwen3.6-35B-A3B \
  --train.enable_masking=true \
  --train.repr_align_wt=1.0 \
  --train.micro_batch_size=1 \
  --train.global_batch_size=16 \
  --train.output_dir=/run/media/johndpope/12TB/open_dllm/checkpoints/35b_a3b_repr_align \
  --train.save_steps=500
```

### Repr-Align: Layer + Token Subsampling (Memory Optimization)

Repr-Align alignment loss scales with the number of layers and sequence length — at 27B with 64 layers and long sequences, computing cosine similarity for every layer every step becomes non-trivial. Two independent knobs reduce this cost.

**What the knobs do:**

| Knob | YAML field | Effect |
|------|-----------|--------|
| Token subsampling | `repr_align_sub_sample_ratio: 0.25` | Random 25% of positions each step → 4× fewer alignment gradient tokens |
| Layer subsampling | `repr_align_num_sample_layers: 4` | Random 4 of N configured layers each step → N/4 fewer alignment losses |

Both are unbiased gradient estimates — every position/layer is covered over time. The hook-based implementation (not `output_hidden_states=True`) means gradient checkpointing is preserved for non-alignment layers.

**Validated setup — all layers in pool, subsampled:**

```bash
# Step 1: precompute anchor cache for all 28 layers, 20-example smoke set
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /tmp/smoke_20.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b-all28-smoke20 \
    --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
    --max_seq_len 2048
# → 20 chunks, 1 GB, ~1s

# Step 2: run training smoke test (5 steps)
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc_per_node=1 \
    tasks/train_torch.py \
    configs/pretrain/qwen3_1_7b_alllayers_subsample_smoke.yaml
```

Config (`configs/pretrain/qwen3_1_7b_alllayers_subsample_smoke.yaml`):
```yaml
train:
  align_layers: "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28"
  repr_align_num_sample_layers: 4    # 4 of 28 sampled each step
  repr_align_sub_sample_ratio: 0.25  # 25% of tokens
  data_parallel_mode: deepspeed
  ds_zero_stage: 2
  ds_offload_optimizer: cpu
  optimizer: adamw
  enable_gradient_checkpointing: true
```

**Measured results — Qwen3-1.7B, RTX 5090, 5 steps:**

| Step | loss | repr_align | grad_norm |
|------|------|-----------|-----------|
| 1 | 13.12 | 0.56 | 0.00 |
| 2 | 13.19 | 0.62 | 127.02 |
| 3 | 11.88 | 0.72 | 177.57 |
| 4 | 11.69 | 0.63 | 177.57 |
| 5 | 8.66 | 0.52 | 177.57 |

| Config | Peak VRAM |
|--------|-----------|
| All 28 layers, sample 4, ratio 0.25 | **9.34 GB** |
| Alignment OFF (baseline) | **9.34 GB** |

**Finding: at 1.7B scale, the subsampling is effectively free.** The alignment tensors (4 layers × ~500 tokens × 2048 hidden × bf16 ≈ 8 MB) are negligible against the 9+ GB model + optimizer footprint. No measurable VRAM difference.

**Where the savings are expected to matter — 27B (unverified):**

At 27B, each layer hidden state is 5120-wide. Full alignment on all 64 layers at seq=2048 would be:
- 64 layers × 5120 × 2048 × 2 bytes = **1.3 GB** of alignment activations per step
- With gradient accumulation, these accumulate

With 4-of-64 layer sampling + 0.25 token ratio:
- 4 × 5120 × 512 × 2 bytes = **21 MB** → ~60× reduction

> **This 60× figure is calculated, not measured.** Whether it translates to a real training OOM difference on the cloud 27B setup (2× RTX PRO 6000, ZeRO-3) has not been validated. The 1.7B results confirm correctness (no NaN, gradient coverage) but not VRAM impact. Verification requires running cloud_27b.yaml with and without subsampling and comparing step logs.

**Bugs fixed in this work:**
- `all_reduce` on a single-element tuple returned a scalar, crashing single-component loss configs (e.g. pure MDM with no alignment) at step 2. Fixed in `tasks/train_torch.py`.

### Alternative: LDLM — Latent Diffusion (Heavy)

**Paper**: [Latent Diffusion Language Models](https://arxiv.org/abs/2605.07933)

Trains **new components from scratch** (Perceiver encoder/decoder + diffusion head) on top of a frozen AR encoder. More expressive but significantly more expensive — requires training 1.39B-6.75B new parameters.

See the full LDLM section below for details.

### Comparison

| | Repr-Align | LDLM |
|---|---|---|
| **New parameters** | 0 (reuses AR model) | 1.39B–6.75B |
| **Training speed** | 3-4× faster | Baseline |
| **Data needed** | As low as 0.8B tokens | More data beneficial |
| **Architecture change** | Attention mask only | New Perceiver + DiT head |
| **When to use** | Default choice for converting existing models | When you need latent-space diffusion |

> **Bottom line**: If you have an off-the-shelf AR model and want diffusion capabilities with minimal compute, use **Repr-Align**. It's already built into the Qwen3.6 model implementations (`modeling_qwen3_5_moe.py`, `modeling_qwen3.py`, `modeling_qwen2.py`).

### d3LLM-Style Trajectory Distillation (Masking Curriculum)

Open-dLLM implements [**d3LLM**](https://arxiv.org/abs/2601.07568) (ICML 2026) trajectory-guided masking for Repr-Align training. Instead of random masking, the unmasking order from a teacher diffusion generation run determines the training-time mask pattern. Tokens that the teacher unmasked early are predicted first; tokens unmasked late are predicted later.

**The problem with random masking**: Standard Repr-Align uses uniformly random masks during training. Random masks give the student model no signal about *which* tokens can be safely predicted with limited context — every position is equally likely to be masked, regardless of its predictability.

**d3LLM's fix**: Pre-compute a *decoding trajectory* (the order in which tokens are unmasked during inference) by running `diffusion_generate()` on each training sample. During training, mask according to the trajectory step closest to the target mask ratio. This aligns training-time masking with inference-time decoding behavior.

**Pipeline**:

1. **Extract trajectories** (one-time pre-processing):
```bash
python -m veomni.ops.trajectory_extractor \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /path/to/train.jsonl \
    --output_dir /path/to/trajectories \
    --max_seq_len 2048 \
    --steps 256
```

2. **Train with trajectory-guided masking**:
```yaml
train:
  enable_masking: true
  repr_align_wt: 1.0
  trajectory_data_path: /path/to/trajectories/trajectories.jsonl
  trajectory_min_mask_ratio: 0.0
  trajectory_max_mask_ratio: 0.8
  trajectory_progressive_block_sizes: "16,24,32"
  trajectory_use_blockwise: true
  trajectory_entropy_weight: 1.0
```

**Key differences from the replay buffer**:
- Replay buffer stores **past batches** to prevent forgetting (uniform sampling)
- Trajectory distillation uses the **teacher's inference-time unmasking order** to guide masking (curriculum learning)
- They are **complementary** — both can be enabled simultaneously (the replay buffer replays alignment loss, while trajectory distillation changes the masking pattern)

---

## ⚡ Multi-Block Decoder (d3LLM Inference)

The **multi-block decoder** implements d3LLM's pipelined parallel decoding ([ICML 2026](https://arxiv.org/abs/2601.07568)) — the inference-side counterpart to trajectory-guided masking. Instead of denoising the full sequence in one block, it divides the generation region into blocks and processes them in a pipeline, achieving up to **~5× speedup over AR decoding**.

### How it works

1. **Block-causal attention**: Each block attends to the prompt + all previous blocks + itself (bidirectional within block). Implemented in `create_block_causal_mask()`.
2. **Block state machine**: Tracks each block's progress through 4 states: Inactive → Activated → Fully-Activated → Completed.
   - `block_add_threshold` (0.5): new block added when last block is ≥50% decoded
   - `decoded_token_threshold` (0.5): next block activated when previous is ≥50% decoded
3. **Entropy-thresholded decoding**: Tokens with entropy < `entropy_threshold` get decoded each step. A forced-progress mechanism ensures at least 1 token per fully-activated block per step.
4. **EOS early stopping**: Detects EOS and immediately marks all subsequent tokens as EOS (not mask), updating block states accordingly.

### Code

```
veomni/models/transformers/qwen2/multi_block_generation.py
```

Key components:

| Component | Description |
|-----------|-------------|
| `MultiBlockDecoderConfig` | Config with `block_size`, `entropy_threshold`, `block_add_threshold`, `decoded_token_threshold`, `early_stop` |
| `MultiBlockDecoderMixin` | Mixin class with `generate_multi_block()` entry point |
| `create_block_causal_mask()` | Full-sequence block-causal attention mask |
| `_sample_multi_block()` | Pipelined parallel decoding loop |

Mixed into:
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Qwen3_5ForCausalLM`
- `Qwen3_5MoeForCausalLM`

### Usage

```python
from veomni.models.transformers.qwen2.multi_block_generation import MultiBlockDecoderConfig

gen_config = MultiBlockDecoderConfig(
    mask_token_id=MASK_ID,
    steps=64,
    block_size=32,
    entropy_threshold=0.9,
    max_length=prompt_len + max_new_tokens,
    temperature=0.0,
    early_stop=True,
    eos_token_id=tokenizer.eos_token_id,
)
result = model.generate_multi_block(input_ids, gen_config)
```

### Current status

| Feature | Status |
|---------|--------|
| Pipelined parallel decoding | ✅ Working |
| Block-causal attention mask | ✅ Working |
| Entropy-thresholded token selection | ✅ Working |
| Forced progress (≥1 token/block/step) | ✅ Working |
| EOS early stopping | ✅ Working |
| KV-cache optimization | 🔴 Blocked (HF cache incompatible with block-causal masks) |
| Trajectory-aware decoding | 📝 Future |

---

Open-dLLM supports **LDLM** (Latent Diffusion Language Model, [arXiv:2605.07933](https://arxiv.org/abs/2605.07933)) — a Perceiver-based latent diffusion approach that jointly trains a latent encoder, diffusion model, and decoder on top of a frozen pre-trained LM. The key insight: reshaping the frozen encoder's hidden states into a diffusion-friendly latent space via a trainable Perceiver, yielding latents that are easy to both denoise and decode into tokens.

#### Architecture Comparison: Paper vs. Our Implementation

The paper trains on GPT-2 small (dim=768) with 4–64× A100s. We adapt LDLM to Qwen3.6 models (dim=2048–5120) on 2 consumer GPUs, requiring significant depth compression.

| Component | Paper (GPT-2, dim=768) | Ours 35B-A3B (dim=2048) | Ours 27B (dim=5120) |
|-----------|------------------------|------------------------|---------------------|
| Frozen encoder | GPT-2 small (124M), layer -3 | Qwen3.6-35B-A3B MoE (3B active), layer -3 | Qwen3.6-27B dense, layer -3 |
| Latent encoder (Perceiver) | 6 layers, 12 heads (~50M) | **4 layers**, 8 heads | **4 layers**, 8 heads |
| Latent decoder (Perceiver) | 6 layers, 12 heads (~50M) | **4 layers**, 8 heads | **4 layers**, 8 heads |
| Token decoder (Transformer) | 3 layers (~66M) | **2 layers** | **2 layers** |
| Diffusion model (DiT) | 12 layers, 12 heads (~132M) | **3 layers**, 8 heads | **4 layers**, 8 heads |
| Latent dim | 768 (matches GPT-2) | **2048** (matches Qwen3.6-35B) | **5120** (matches Qwen3.6-27B) |
| Trainable params (total) | ~300M | ~1.39B | ~6.75B |
| σ_dec | 3.0 | 3.0 | 3.0 |
| Self-conditioning | 50% | 50% | 50% |
| Warmup schedule | Sigmoid (k=10, c=0.8) | Sigmoid (k=10, c=0.8) | Sigmoid (k=10, c=0.8) |
| Noise schedule | Tangent (d=3) | Tangent (d=3) | Tangent (d=3) |

> **Key differences**: Our latent dim is 2.7–6.7× larger than the paper's (dictated by the Qwen3.6 encoder's hidden size), but our Perceiver/DiT depths are 2–4× shallower (dictated by GPU memory). The paper uses ~300M trainable params on 4–64× A100s; we use 1.39B–6.75B on 2 consumer GPUs. The larger latent dim means each layer is more expensive (parameters scale as dim²), but fewer layers partially compensates. The `latent_dim` parameter in `LDLMAutoencoder` could be set to a smaller value (e.g., 768) to add a projection bottleneck — this is not yet explored.

#### Training Recipe (from the paper)

The paper identifies 4 critical components for successful joint training (ablations show each substantially impacts generation quality):

1. **MSE decoder loss** (L_h, Eq. 2): MSE between hidden states h and decoder output h_hat, with decoder-input noise σ_dec·ε. MSE is preferred over CE because it doesn't force latents to be well-separated — it allows nearby latents to map to averaged hidden states, producing a smoother latent geometry for diffusion.

2. **Diffusion-to-encoder warmup** (Eq. 29-30): At training start, L_diff and L_h pull the latent space in opposite directions. The warmup multiplies L_diff gradients to the encoder by γ(s), which increases from ~0 to 1 via a sigmoid schedule over S_wu steps. The encoder first learns to reconstruct, then the diffusion objective gradually shapes the latent space.

3. **Adaptive timestep sampling** (Eq. 5): Dynamically adjusts the noise schedule so that the denoising loss grows linearly with the sampled timestep — all timesteps contribute equally to training. A running EMA of loss per timestep bin is maintained and used to compute sampling probabilities proportional to dL/du.

4. **Decoder-input noise** (σ_dec = 3.0): Gaussian noise injected into the decoder input during training (only training, not inference). Three roles: (a) prevents unused latent dimensions from consuming capacity, (b) makes the decoder robust to diffusion model errors, (c) normalizes input variance across timesteps for better diffusion parameterization.

**Total objective**: `L = L_diff · γ(s) + L_h + L_w`, where L_w is the token CE loss with stop-gradient on h_hat (so it doesn't affect the latent encoder).

#### Recreating the Benchmarks

```bash
# 1. Install dependencies (see Install section above)
pip install -e .

# 2. Download the encoder model (only needed for training; benchmark downloads automatically)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-35B-A3B')   # ~22GB download
# snapshot_download('Qwen/Qwen3.6-27B')     # ~54GB download
"

# 3. Run inference benchmark on a single GPU
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py    # Qwen3.6-35B-A3B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py        # Qwen3.6-27B
```

> **Hardware used**: NVIDIA RTX 5090 (32GB VRAM), 96GB system RAM, Python 3.11, PyTorch 2.12+, CUDA 12.9.

#### Inference Throughput (Qwen3.6 LDLM, untrained, RTX 5090 32GB)

| Model | Dim | Trainable Params | Diffusion Steps | Throughput |
|-------|-----|-------------------|-----------------|------------|
| Qwen3.6-35B-A3B | 2048 | 1.39B | 10 | **3,238 tok/s** |
| Qwen3.6-35B-A3B | 2048 | 1.39B | 4 | **~6,500 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 10 | **745 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 4 | **~1,500 tok/s** |

> For comparison, autoregressive generation on the same hardware achieves ~30-50 tok/s for a 27B model.

#### Assumptions & Caveats

- **Untrained weights**: These benchmarks use randomly initialized Perceiver/decoder/diffusion-head weights. A trained model will have identical throughput but produce coherent output. Quality benchmarks (perplexity, HumanEval) will be published after training completes.
- **No encoder in the loop**: The frozen Qwen3.6 encoder is **not used during generation** — it's only needed for training (to produce latent targets). At inference, the diffusion head denoises random noise, then the Perceiver decoder maps latents to tokens. The encoder is deleted before benchmarking (`del autoencoder.token_encoder`).
- **Seq len = 64**: The benchmark uses a short sequence length (64 tokens). Longer sequences will reduce throughput proportionally. The 4-step throughput numbers are linear extrapolations from the 10-step measurements.
- **Batch size = 1**: Single-sequence generation only. Throughput scales near-linearly with batch size for the 35B-A3B (dim=2048 fits easily in VRAM), less so for the 27B (dim=5120).
- **CPU RAM requirement**: While the encoder is not used at inference, it **must** fit in system RAM during training (~54GB for 27B, ~22GB for 35B-A3B in bf16). The Qwen3.6 architecture uses Triton kernels (flash-linear-attention) that cannot run on CPU, so the encoder forward pass during training requires GPU offloading — a multi-GPU setup is recommended for training.
- **Qwen3.6 requires `trust_remote_code=True`**: The model uses custom architecture code (`Qwen3_5ForConditionalGeneration`) that is not in standard transformers releases. Ensure your `transformers` version supports it (>=4.54).
- **35B-A3B is MoE**: Only 3B of its 35B parameters are active per token, giving it a much smaller hidden dim (2048) than the 27B dense model (5120). This is why the LDLM trainable components are 5x smaller and 4x faster.
- **Not an apples-to-apples comparison with AR models**: The diffusion model generates all tokens in parallel across N diffusion steps, while AR generates one token at a time. The "tok/s" metric favors diffusion for short sequences but does not reflect output quality, which depends on training convergence.
- **Architecture depth vs. paper**: Our Perceiver/DiT depths are 2–4× shallower than the paper's (4 vs. 6 Perceiver layers, 3–4 vs. 12 DiT layers). This is a memory constraint, not a design choice. The latent dim (2048/5120) is 2.7–6.7× larger than the paper's 768, meaning each layer has ~7–44× more parameters. Future work could add a projection bottleneck (`latent_dim=768`) to reduce this and enable deeper architectures.

#### How to Train a Qwen3.6 LDLM

1. **Download the base model** (27B dense or 35B-A3B MoE):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B', local_dir='./qwen36_27b_local')
# or for MoE:
# snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir='./qwen36_35b_a3b_local')
"
```

2. **Prepare training data** (e.g., FineWeb):
```bash
python -c "
from datasets import load_dataset
import json
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
with open('data.jsonl', 'w') as f:
    for i, ex in enumerate(ds):
        if i >= 100000: break
        f.write(json.dumps({'text': ex['text']}) + '\n')
"
```

3. **Run the benchmark** (verify setup before training):
```bash
# 27B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py
# 35B-A3B MoE
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py
```

4. **Start training** (single GPU):
```bash
# 27B — single GPU (encoder on CPU, trainable on GPU 0)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_27b_ldlm.yaml
# 35B-A3B MoE — single GPU
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

5. **Start training** (2 GPUs, e.g. RTX 5090 + RTX 4000):
```bash
# 35B-A3B MoE — frozen encoder on GPU 0, trainable components on GPU 1
torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

> **Note**: Use `--nproc_per_node=1` always — the script handles multi-GPU placement internally (encoder on GPU 0 via `device_map="auto"`, trainable Perceiver/diffusion head on GPU 1). Do NOT use `--nproc_per_node=2` or both processes will collide on GPU 1.
>
> **GPU Memory**: With 2 GPUs, the frozen encoder runs on GPU 0 (~22GB VRAM for 35B-A3B, ~54GB for 27B) and trainable components run on GPU 1. With 1 GPU, the encoder stays on CPU and only trainable components use GPU VRAM. The 35B-A3B MoE variant has a smaller hidden dim (2048 vs 5120), making it significantly faster and more memory-efficient — ideal for consumer GPUs.

---

## 💻 System Requirements

### Minimum (for 0.5B–1.7B models)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× GPU with ≥8GB VRAM (e.g., RTX 3060) |
| RAM | 16 GB |
| Storage | 50 GB free |
| CUDA | 12.x+ |

### Recommended (for 7B–27B Repr-Align)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× RTX 3090/4090/5090 (≥24GB VRAM) |
| RAM | **192 GB** (27B ZeRO-3 with CPU offload peaks at ~170GB during init) |
| Storage | 200 GB NVMe (for model weights + anchor cache + DS swap) |
| CUDA | 12.9+ (for Blackwell/RTX 5090) |

### Cloud alternative (for 27B+)

| Provider | Instance | VRAM | Cost | Notes |
|----------|----------|------|------|-------|
| Lambda Labs | 8×H100 80GB | 640 GB | ~$30-50/hr | Full 27B train in <1 hr |
| RunPod | 4×A100 80GB | 320 GB | ~$12-20/hr | Sufficient with ZeRO-3 |
| Vast.ai | 2×A6000 48GB | 96 GB | ~$2-4/hr | Budget option, needs NVMe offload |

### RAM budget for 27B Repr-Align (DeepSpeed ZeRO-3)

| Component | Size | Device |
|-----------|------|--------|
| Model params (bf16) | ~54 GB | CPU (offloaded) |
| Optimizer states (bf16) | ~108 GB | NVMe (offloaded) |
| DeepSpeed buffers | ~10-20 GB | RAM |
| **Peak during init** | **~170 GB** | RAM (before swap-out) |

> **Key insight**: The bottleneck for local 27B training is **system RAM, not GPU VRAM**. DeepSpeed ZeRO-3 + NVMe offload handles the GPU side, but `deepspeed.initialize()` materializes the full model + optimizer in RAM before swapping to NVMe. 96GB RAM is insufficient; 128GB is marginal; 192GB is comfortable.

---

## 🔬 Hardware Investigation Notes

### What works

- **Qwen3-1.7B Repr-Align**: Passes on single RTX 5090 with DeepSpeed ZeRO-3 + CPU offload (~4.8GB VRAM, ~15s/step). Full wandb logging. Anchor precompute takes ~30s for 1000 examples.
- **Qwen3.6-27B anchor precompute**: Works across both GPUs (RTX 5090 + RTX 4000) with `device_map=auto` + CPU overflow. 1000 examples × 4 layers = 27GB cache in ~20 min.
- **DeepSpeed ZeRO-3 + NVMe optimizer offload**: Successfully writes ~180GB of optimizer/param state to NVMe. The init completes; the RAM peak is the bottleneck.

### What doesn't work (yet)

- **27B full-layer Repr-Align on 96GB RAM**: OOM killed during `deepspeed.initialize()`. The init peak (~170GB) exceeds available RAM + swap (104GB total). Two investigation paths remain open:
  1. **Upgrade to 192GB RAM**: Replace 2×16GB sticks with 2×64GB. Cost: ~$2,000 AUD. Should comfortably fit the init peak.
  2. **Lazy init with `init_device: meta`**: Skip materializing params in RAM during model construction. DeepSpeed's `zero.Init` context + `remote_device="cpu"` still allocates params on CPU. A true meta-device init would defer allocation until DeepSpeed can partition + swap directly, never holding the full model in RAM. This requires changes to the weight loading path in `veomni/models/loader.py`.

- **2-GPU ZeRO-3 with 96GB RAM**: Each rank holds a partition (~27GB params), totaling 54GB in RAM. Optimizer adds another ~108GB. Total exceeds RAM even before considering DeepSpeed buffers.

### Hobby RAM vs Cloud H100 — cost comparison

| Approach | Upfront Cost | Per-run Cost | Time for 10 steps | Notes |
|----------|-------------|-------------|-------------------|-------|
| **RAM upgrade** (96→192GB) | ~$2,000 AUD | $0 (electricity) | ~15-30 min | One-time, reusable for future models |
| **Rent 8×H100** (Lambda) | $0 | ~$30-50/hr | <5 min | Faster, but per-experiment cost adds up |
| **Rent 4×A100** (RunPod) | $0 | ~$12-20/hr | <10 min | Sweet spot for 27B |
| **Rent 2×A6000** (Vast.ai) | $0 | ~$2-4/hr | ~20 min | Cheapest cloud option |

> **Break-even**: At $2,000 AUD for RAM vs ~$30/hr for H100s, the RAM upgrade pays for itself after ~65 hours of training. If you're iterating frequently (hyperparameter sweeps, multiple models), the RAM upgrade is more economical. If you only need a few one-off runs, cloud is cheaper.

### DeepSpeed NVMe offload gotchas

- **`DS_SKIP_CUDA_CHECK=1`**: Required when system CUDA toolkit (13.1) doesn't match PyTorch's compiled CUDA (13.0). Without it, DeepSpeed can't compile async_io extensions needed for NVMe offload.
- **`buffer_size`**: Must exceed the largest combined partition size. Default 100M elements is too small for 27B (622M for embed_tokens alone). Set to 2B elements: `offload["buffer_size"] = 2_000_000_000`.
- **Pre-build async_io**: Run `DS_SKIP_CUDA_CHECK=1 python -c "import deepspeed.ops.op_builder as b; b.AsyncIOBuilder().load()"` once before training.
- **`torch.empty` on `"nvme"` device**: DeepSpeed's `_post_init` patch in `veomni/distributed/deepspeed_init.py` must allocate on `"cpu"` not `"nvme"` — PyTorch doesn't recognize `nvme` as a device. The async_io swapper handles the NVMe transfer separately.

---

## 🙏 Appreciation

This project builds on incredible prior work:

* **Frameworks & Tooling**: [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **Open-source dLLMs**: [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream)
* **Pioneering dLLMs**: [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/), [Seed Diffusion](https://seed.bytedance.com/seed_diffusion), [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model)
* **Foundational research**: [MD4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html), [MDLM](https://arxiv.org/abs/2406.07524), [DPLM](https://github.com/bytedance/dplm)

We stand on the shoulders of these projects, and hope Open-dLLM contributes back to the diffusion LLM community.




## 📚 Citation

If you use **Open-dLLM** or **Open-dCoder** in your research, please cite us:

```bibtex
@misc{opendllm2025,
  title        = {Open-dLLM: Open Diffusion Large Language Models},
  author       = {Fred Zhangzhi Peng, Shuibai Zhang, Alex Tong, and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/pengzhangzhi/Open-dLLM}},
  note         = {Blog: \url{https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74}, 
                  Model: \url{https://huggingface.co/fredzzp/open-dcoder-0.5B}}
}
