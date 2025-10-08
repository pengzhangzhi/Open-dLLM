# sample.py

import torch
from transformers import AutoTokenizer

# Import the custom generation config
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

# Import your custom model class directly
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM


# 1. Define paths and parameters
model_path = "fredzzp/open-dcoder-0.5B"  # "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"
# You might need to use the original tokenizer path if it's not saved with the checkpoint
tokenizer_path = model_path
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Tokenizer and Model
# trust_remote_code=True is essential because you are loading a custom model implementation
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to(device).eval()

# Set the mask token if not already set. This is crucial for generation.
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model.resize_token_embeddings(len(tokenizer))
    print("Added new [MASK] token.")

# 3. Prepare generation config and inputs
prompt = """
Write a quick sort algorithm in Python.
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Create a generation configuration object
generation_config = MDMGenerationConfig(
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id,  # Usually same as eos for decoder-only
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    steps=128,
    temperature=0.5,
    top_k=200,
    alg="p2",
    alg_temp=0.5,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_history=True,
)

# 4. Generate text using the diffusion_generate method
print("\nStarting generation...")
with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=generation_config)
print("Generation complete.")


# 5. Decode and print the output
prompt_len = input_ids.shape[1]
generated_sequences = outputs.sequences
generated_text = tokenizer.decode(generated_sequences[0][prompt_len:], skip_special_tokens=True)

print("\n--- Prompt ---")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
print("\n--- Generated Code ---")
print(generated_text)
