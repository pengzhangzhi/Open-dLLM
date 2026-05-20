"""
Forward pass smoke test for Qwen3.6-27B using the local Q4_K_XL GGUF.
17 GB quantized — fits on RTX PRO 4000 (25 GB) without touching the 5090.

Requires: llama-cpp-python built with CUDA support
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
        uv pip install llama-cpp-python --no-binary llama-cpp-python

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/test_27b_forward.py
"""
import time
import torch

GGUF_PATH = (
    "/home/johndpope/.cache/huggingface/hub/"
    "models--unsloth--Qwen3.6-27B-MTP-GGUF/blobs/"
    "4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095"
)
PROMPT = "The capital of France is"
N_TOKENS = 32
GPU_LAYERS = 99   # offload all layers to GPU


def vram_gb():
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 1e9, total / 1e9


def main():
    from llama_cpp import Llama

    used_before, total = vram_gb()
    print(f"VRAM before load: {used_before:.1f} / {total:.1f} GB")

    print(f"Loading Q4_K_XL GGUF ({GGUF_PATH[-20:]}...) with {GPU_LAYERS} GPU layers ...")
    t0 = time.time()
    llm = Llama(
        model_path=GGUF_PATH,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=512,
        verbose=True,
    )
    load_time = time.time() - t0
    used_after, _ = vram_gb()
    print(f"Loaded in {load_time:.1f}s — VRAM: {used_after:.1f} / {total:.1f} GB (model ≈ {used_after - used_before:.1f} GB)")

    print(f"\nRunning forward pass: '{PROMPT}' → {N_TOKENS} tokens ...")
    t1 = time.time()
    out = llm(PROMPT, max_tokens=N_TOKENS, echo=True)
    elapsed = time.time() - t1

    used_peak, _ = vram_gb()
    text = out["choices"][0]["text"]
    print(f"\nOutput: {text!r}")
    print(f"Time: {elapsed:.2f}s  |  Peak VRAM: {used_peak:.1f} GB")
    print("OK — no OOM.")


if __name__ == "__main__":
    main()
