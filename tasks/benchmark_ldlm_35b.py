import time

import torch

from veomni.models.ldlm.autoencoder import DiffusionHead, LDLMAutoencoder


SEQ_LEN = 64
DEPTH = 4
DECODER_NUM_LAYERS = 2
DIFFUSION_HEAD_DEPTH = 4
NUM_DIFFUSION_STEPS = 10
WARMUP_ITERS = 3
BENCH_ITERS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ENCODER_NAME = "Qwen/Qwen3.6-35B-A3B"

print(f"Building LDLM autoencoder (encoder={ENCODER_NAME})...")
autoencoder = LDLMAutoencoder(
    encoder_model_name=ENCODER_NAME,
    seq_len=SEQ_LEN,
    depth=DEPTH,
    decoder_input_noise_std=3.0,
    encoder_hidden_layer=-3,
    decoder_num_layers=DECODER_NUM_LAYERS,
)

diffusion_head = DiffusionHead(
    dim=autoencoder.dim,
    depth=DIFFUSION_HEAD_DEPTH,
)

autoencoder.latent_encoder = autoencoder.latent_encoder.to(device)
autoencoder.latent_decoder = autoencoder.latent_decoder.to(device)
autoencoder.token_decoder = autoencoder.token_decoder.to(device)
autoencoder.lm_head = autoencoder.lm_head.to(device)
diffusion_head = diffusion_head.to(device)

del autoencoder.token_encoder
torch.cuda.empty_cache()

trainable = sum(p.numel() for p in list(autoencoder.latent_encoder.parameters())
    + list(autoencoder.latent_decoder.parameters())
    + list(autoencoder.token_decoder.parameters())
    + list(autoencoder.lm_head.parameters())
    + list(diffusion_head.parameters()))
print(f"Trainable params: {trainable/1e6:.1f}M")
print(f"Vocab size: {autoencoder._vocab_size}, Dim: {autoencoder.dim}")

autoencoder.eval()
diffusion_head.eval()

@torch.no_grad()
def generate(num_steps=NUM_DIFFUSION_STEPS, seq_len=SEQ_LEN):
    z = torch.randn(1, seq_len, autoencoder.dim, device=device)
    for i in range(num_steps):
        t = torch.full((1,), (i + 1) / num_steps, device=device)
        alpha_bar = 1.0 - t[:, None, None] ** 2
        sigma = (1.0 - alpha_bar).sqrt()
        pred = diffusion_head(z, t)
        z = alpha_bar.sqrt() * pred + sigma * torch.randn_like(pred) * 0.1

    dec_out = autoencoder.decode(z, training=False)
    logits = dec_out["logits"]
    tokens = logits.argmax(dim=-1)
    return tokens

print(f"\nWarmup ({WARMUP_ITERS} iters)...")
for _ in range(WARMUP_ITERS):
    _ = generate()

torch.cuda.synchronize()

print(f"Benchmarking ({BENCH_ITERS} iters, {NUM_DIFFUSION_STEPS} diffusion steps, seq_len={SEQ_LEN})...")
times = []
for i in range(BENCH_ITERS):
    torch.cuda.synchronize()
    t0 = time.time()
    tokens = generate()
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0
    times.append(elapsed)
    tok_per_sec = SEQ_LEN / elapsed
    print(f"  iter {i+1}: {elapsed:.3f}s, {tok_per_sec:.1f} tok/s")

avg_time = sum(times) / len(times)
avg_tok_sec = SEQ_LEN / avg_time
print(f"\n{'='*50}")
print(f"Model: {ENCODER_NAME}")
print(f"Avg latency: {avg_time:.3f}s")
print(f"Avg throughput: {avg_tok_sec:.1f} tok/s")
print(f"Seq len: {SEQ_LEN}, Diffusion steps: {NUM_DIFFUSION_STEPS}")
print(f"Throughput per diffusion step: {avg_tok_sec * NUM_DIFFUSION_STEPS:.1f} tok/s (raw)")
print(f"{'='*50}")

save_dir = "/run/media/johndpope/12TB/open_dllm/ldlm_model"
import os


os.makedirs(save_dir, exist_ok=True)
state = {
    "latent_encoder": autoencoder.latent_encoder.state_dict(),
    "latent_decoder": autoencoder.latent_decoder.state_dict(),
    "token_decoder": autoencoder.token_decoder.state_dict(),
    "lm_head": autoencoder.lm_head.state_dict(),
    "diffusion_head": diffusion_head.state_dict(),
    "config": {
        "encoder": ENCODER_NAME,
        "seq_len": SEQ_LEN,
        "depth": DEPTH,
        "decoder_num_layers": DECODER_NUM_LAYERS,
        "diffusion_head_depth": DIFFUSION_HEAD_DEPTH,
        "dim": autoencoder.dim,
        "vocab_size": autoencoder._vocab_size,
    },
}
torch.save(state, os.path.join(save_dir, "ldlm_35b_a3b_untrained.pt"))
print(f"Saved to {save_dir}/ldlm_35b_a3b_untrained.pt")
