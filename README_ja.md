
# 🔥 Open-dLLM: オープン拡散大規模言語モデル

🌍 言語: [English](README.md) | [中文](README_cn.md) | [日本語](README_ja.md)

👉 TL;DR: **Open-dLLM**は、拡散ベースの大規模言語モデルの最もオープンなリリースです —
**事前学習、評価、推論、チェックポイント**のすべてを含みます。

このリポジトリでは、Open-dLLMの**コード生成バリアント**である**Open-dCoder**を紹介します。


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
  <b>💻 コード</b> &nbsp; | &nbsp; <b>📖 ブログ</b> &nbsp; | &nbsp; <b>🤗 モデル</b>
</p>


## 🎥 デモ

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif"
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center"><i>Open-dCoder (0.5B)を使用したクイックソート生成</i></p>

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

## ✨ ハイライト

- 🏋️ **事前学習パイプライン + オープンデータセット**
- ⚡ **推論スクリプト** — 簡単なサンプリングと生成
- 📊 **評価スイート** — HumanEval、MBPP、Infilling（lm-eval-harness + カスタムメトリクス）
- 📦 **重みとチェックポイント**をHugging Faceで公開
- 🤝 **透明性のある設定**により完全な再現性を実現

---

## なぜOpen-dLLMなのか？

ほとんどの拡散LLMリポジトリ（例：LLaDA、Dream）は**推論スクリプトと重み**のみをリリースしており、再現性が制限されています。
**Open-dLLM**は、拡散LLMの**全スタック**をオープンソース化した初めてのプロジェクトです。

👉 Open-dLLMを使えば、**生データ → 学習 → チェックポイント → 評価 → 推論**のすべてを1つのリポジトリで実現できます。

---

## 🔎 拡散LLMリリースの透明性比較

| プロジェクト                                                                 | データ | 学習コード | 推論 | 評価 | 重み |
|-------------------------------------------------------------------------|:---:|:-------------:|:---------:|:----------:|:-------:|
| **Open-dLLM / Open-dCoder（本プロジェクト）**                                      | ✅  | ✅            | ✅        | ✅         | ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                               | ❌  | ❌            | ✅        | ⚠️ 限定的 | ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                | ❌  | ❌            | ✅        | ⚠️ 限定的 | ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)    | ❌  | ❌            | ❌        | ❌         | ❌（APIのみ） |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)             | ❌  | ❌            | ❌        | ❌         | ❌（APIのみ） |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) | ❌  | ❌            | ❌        | ❌         | ❌（APIのみ） |

✅ = 完全利用可能 · ❌ = 提供なし · ⚠️ = 部分的/限定的

---

## ⚙️ インストール

環境管理には`micromamba`を使用しています（`conda`にも対応可能）:

```bash
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# cu121対応の最新torchをインストール
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8 \
  pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .
```

---

## 🚀 クイックスタート: サンプリング

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# プロンプト
prompt = "Write a quick sort algorithm in python."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 生成設定
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 完全なロギング、履歴追跡、ファイル出力を行う場合:

```bash
python sample.py
```

---

## 📊 ベンチマーク

拡散ベースのLLM（dLLM）向けの完全オープンソース**評価スイート**をリリースしています。**標準的なコード生成タスク**と**コード補完（Infilling）タスク**の両方をカバーしています。

ベンチマークには次が含まれます: **HumanEval / HumanEval+**、**MBPP / MBPP+**、**HumanEval-Infill**、**SantaCoder-FIM**。

---

#### 標準的なコード生成

| 手法                       | HumanEval |          | HumanEval+ |          | MBPP     |          | MBPP+    |          |
| ---------------------------- | --------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- |
|                              | Pass\@1   | Pass\@10 | Pass\@1    | Pass\@10 | Pass\@1  | Pass\@10 | Pass\@1  | Pass\@10 |
| LLaDA (8B)                   | 35.4      | 50.0     | 30.5       | 43.3     | 38.8     | 53.4        | 52.6     | 69.1        |
| Dream (7B)                   | 56.7      | 59.2     | 50.0       | 53.7     | 55.4     | 56.2        | 71.5     | 72.5        |
| Mask DFM (1.3B)              | 9.1       | 17.6     | 7.9        | 13.4     | 6.2      | 25.0     | –        | –        |
| Edit Flow (1.3B)             | 12.8      | 24.3     | 10.4       | 20.7     | 10.0     | 36.4     | –        | –        |
| **Open-dCoder (0.5B、本プロジェクト)** | **20.8**  | **38.4** | **17.6**   | **35.2** | **16.7** | **38.4** | **23.9** | **53.6** |

> *わずか0.5Bパラメータにもかかわらず、Open-dCoderはコード補完タスクにおいて、はるかに大規模なdLLMと競合します。*

---

#### コード補完（Infilling）

| 手法                                | HumanEval Infill Pass@1 | SantaCoder Exact Match |
| ------------------------------------- | ----------------------: | ---------------------: |
| LLaDA-8B                              |                    48.3 |                  35.1  |
| Dream-7B                              |                    39.4 |                  40.7  |
| DiffuCoder-7B                         |                    54.8 |                  38.8  |
| Dream-Coder-7B                        |                    55.3 |                  40.0  |
| **Open-dCoder (0.5B、本プロジェクト)**          |                    32.5 |                  29.6  |
| **Open-dCoder (0.5B、本プロジェクト)** Oracle Length |               77.4 |                  56.4  |

> *結果取得には、[DreamOn](https://hkunlp.github.io/blog/2025/dreamon/)の平均固定長評価設定に従いました。*

---

## 🧪 評価

評価パッケージをインストール:

```bash
pip install -e lm-evaluation-harness human-eval-infilling
```

#### コード補完（HumanEval、MBPP）

```bash
cd eval/eval_completion
bash run_eval.sh
```

#### コード補完（Infilling）

```bash
cd eval/eval_infill
bash run_eval.sh
```

---

## 🏋️ 事前学習

* **データ**: 簡潔で高品質なコードコーパス[**FineCode**](https://huggingface.co/datasets/fredzzp/fine_code)をHugging Faceでホスティング
* **初期化**: *Dream*に従い、**Qwen2.5-Coder**から継続事前学習を行い、拡散フレームワークに適応
* **損失**: Masked Diffusion Model（MDM）目的関数 — マスク比率を`[0,1]`から一様サンプリングし、クロスエントロピー損失で再構成

### データのダウンロード

```bash
python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
```

### 学習

```bash
export TOKENIZERS_PARALLELISM=false
NNODES=${NNODES:=1}
NPROC_PER_NODE=4
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

### Hugging Faceへのチェックポイントアップロード

```python
from huggingface_hub import HfApi

REPO_ID = "fredzzp/open-dcoder-0.5B"
LOCAL_DIR = "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=LOCAL_DIR)
```

---

## 🙏 謝辞

このプロジェクトは、素晴らしい先行研究の上に成り立っています:

* **フレームワークとツール**: [VeOmni](https://github.com/ByteDance-Seed/VeOmni)、[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **オープンソースdLLM**: [LLaDA](https://github.com/ML-GSAI/LLaDA)、[Dream](https://github.com/HKUNLP/Dream)
* **先駆的なdLLM**: [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)、[Seed Diffusion](https://seed.bytedance.com/seed_diffusion)、[Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model)
* **基盤研究**: [MD4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html)、[MDLM](https://arxiv.org/abs/2406.07524)、[DPLM](https://github.com/bytedance/dplm)

私たちはこれらのプロジェクトの肩の上に立っており、Open-dLLMが拡散LLMコミュニティに貢献できることを願っています。




## 📚 引用

**Open-dLLM**または**Open-dCoder**を研究で使用される場合は、以下のように引用してください:

```bibtex
@misc{opendllm2025,
  title        = {Open-dLLM: Open Diffusion Large Language Models},
  author       = {Fred Zhangzhi Peng, Shuibai Zhang, Alex Tong, and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/pengzhangzhi/Open-dLLM}},
  note         = {Blog: \url{https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74},
                  Model: \url{https://huggingface.co/fredzzp/open-dcoder-0.5B}}
}
```
