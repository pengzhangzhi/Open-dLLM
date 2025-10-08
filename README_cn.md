# 🔥 Open-dLLM: 开源扩散式大语言模型

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 TL;DR: **Open-dLLM** 是迄今为止最开放的扩散式大语言模型发布 —— 我们开源了 **预训练、评测、推理以及模型权重**。

本仓库介绍了 **Open-dCoder**，它是 Open-dLLM 的 **代码生成版本**。

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
  <b>💻 代码</b> &nbsp; | &nbsp; <b>📖 博客</b> &nbsp; | &nbsp; <b>🤗 模型</b>
</p>

---

## 🎥 演示

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif"
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center"><i>使用 Open-dCoder (0.5B) 生成快速排序算法</i></p>

<p align="center">
  <a href="https://youtu.be/d8WrmvUhO9g">
    <img src="https://img.shields.io/badge/YouTube-视频-red?logo=youtube" alt="YouTube link"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.bilibili.com/video/BV1ZveSz3E1J/">
    <img src="https://img.shields.io/badge/Bilibili-视频-blue?logo=bilibili" alt="Bilibili link"/>
  </a>
</p>

---

## ✨ 亮点

- 🏋️ **完整预训练流程 + 开源数据集**
- ⚡ **推理脚本** —— 简单运行采样和生成
- 📊 **评测套件** —— HumanEval、MBPP、代码（支持 lm-eval-harness + 自定义指标）
- 📦 **模型权重**（已上传到 Hugging Face）
- 🤝 **透明配置**，可完全复现

---

## 为什么选择 Open-dLLM？

目前大多数扩散式 LLM 仓库（例如 LLaDA、Dream）只开源了 **推理代码和权重**，限制了复现性。
**Open-dLLM** 是第一个开源 **全栈** 的扩散式 LLM：

👉 从 **原始数据 → 训练 → 权重 → 评测 → 推理**，全流程一个仓库搞定。

---

## 🔎 扩散式 LLM 开放程度对比

| 项目                                                                               | 数据 | 训练代码 | 推理 |  评测   |    权重     |
| ---------------------------------------------------------------------------------- | :--: | :------: | :--: | :-----: | :---------: |
| **Open-dLLM / Open-dCoder (ours)**                                                 |  ✅  |    ✅    |  ✅  |   ✅    |     ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                                          |  ❌  |    ❌    |  ✅  | ⚠️ 部分 |     ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                           |  ❌  |    ❌    |  ✅  | ⚠️ 部分 |     ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)               |  ❌  |    ❌    |  ❌  |   ❌    | ❌ (仅 API) |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)                        |  ❌  |    ❌    |  ❌  |   ❌    | ❌ (仅 API) |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) |  ❌  |    ❌    |  ❌  |   ❌    | ❌ (仅 API) |

✅ = 完全开源 · ❌ = 未提供 · ⚠️ = 部分/有限

---

## ⚙️ 安装

我们推荐使用 `micromamba` 管理环境（也可改用 `conda`）：

```bash
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# 安装最新 torch (cu121)
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

pip install --upgrade --no-cache-dir \
  tensordict torchdata byte-flux triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8 \
  pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .
```

---

## 🚀 快速开始：采样

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer + 模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# 输入提示
prompt = "用Python写一个快速排序算法。"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 生成配置
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 更多日志记录与文件输出：

```bash
python sample.py
```

---

## 📊 基准测试

我们开源了完整的 **评测套件**，覆盖 **标准代码生成任务** 和 **代码填充任务**：

- HumanEval / HumanEval+
- MBPP / MBPP+
- HumanEval-Infill
- SantaCoder-FIM

结果表格与 README 中一致，这里不再重复。

---

## 🏋️ 预训练

- **数据**: 开源高质量代码语料 [**FineCode**](https://huggingface.co/datasets/fredzzp/fine_code)
- **初始化**: 基于 **Qwen2.5-Coder** 继续预训练，从自回归 → 扩散
- **目标函数**: Masked Diffusion Model (MDM)，mask 比例均匀采样 `[0,1]`

---

## 🙏 致谢

本项目建立在以下工作之上：

- **框架与工具**: [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **开源 dLLM**: [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream)
- **先锋探索**: [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/), [Seed Diffusion](https://seed.bytedance.com/seed_diffusion), [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model)
- **基础研究**: [MD4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html), [MDLM](https://arxiv.org/abs/2406.07524), [DPLM](https://github.com/bytedance/dplm)

我们希望 **Open-dLLM** 能回馈社区，推动扩散式大语言模型研究。

---

## 📚 引用

如果您在研究中使用 **Open-dLLM** 或 **Open-dCoder**，请引用：

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
