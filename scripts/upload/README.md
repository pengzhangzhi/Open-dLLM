---
license: apache-2.0
language:
  - code
library_name: transformers
tags:
  - masked-diffusion
  - code-generation
  - qwen2
---

## Open Diffusion Large Language Models for Code Generation

This repository contains the weights and custom code for the **{repo_id}** model, a masked diffusion model for code generation based on the Qwen2 architecture.

This model uses bidirectional attention and must be used with the custom `diffusion_generate` method.

## How to Use

First, make sure you have the latest `transformers` library installed.

```bash
pip install transformers torch huggingface_hub
```

You can then use the model for generation. Note: You must pass trust_remote_code=True to load the custom model architecture.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
device = "cuda" if torch.cuda.is_available() else "cpu"

# trust_remote_code=True is essential
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

prompt = "def fibonacci(n):"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# The model will use the generation_config.json from the repo by default
# You can also override parameters here
outputs = model.diffusion_generate(
    inputs=input_ids,
    max_new_tokens=100,
    steps=16,
    temperature=0.8
)

# Decode the output
prompt_len = input_ids.shape[1]
generated_text = tokenizer.decode(outputs.sequences[0][prompt_len:], skip_special_tokens=True)

print("--- Generated Code ---")
print(generated_text)
```
