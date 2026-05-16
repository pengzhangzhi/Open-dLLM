# Cola DLM — Continuous Latent Diffusion LM on top of Repr-Align

Paper: **Cola DLM** (arXiv:[2605.06548](https://arxiv.org/abs/2605.06548)) —
"Continuous Latent Diffusion Language Model." The paper proposes a
**Text VAE** that compresses tokens into multi-scale continuous latents,
plus a **block-causal DiT** that models the global semantic prior over
those latents, and a conditional decoder that maps latents back to text.

This Open-dLLM port adapts the recipe as an **opt-in auxiliary head on
top of Repr-Align** (not a stand-alone model). The Repr-Aligned LM
plays the role of both the Text VAE's hidden-state source and the
"conditional decoder" — we keep its LM head untouched. The only new
trainable pieces are the Text VAE *encoder* (a hierarchical Perceiver)
and the block-causal DiT denoiser.

The LDLM stack in `veomni/models/ldlm/` is **not touched** by this
change.

---

## 1. Architecture

```
input_ids (masked)        casual_input_ids (clean)
       │                          │
       ▼                          ▼
┌─────────────────┐        ┌─────────────────┐
│ Student LM      │        │ Teacher LM      │
│ (bidirectional) │        │ (causal, frozen)│
└────────┬────────┘        └────────┬────────┘
         │                          │
         │  hidden_states[L-3]      │ hidden_states (all layers)
         │                          │
         ├─── MDM + path loss       │
         │                          │
         ├─── Repr-Align cosine ◄───┘  (existing)
         │
         ▼                                                 ┐
  (detach by default)                                      │
         │                                                 │
         ▼                                                 │
┌────────────────────────────────────────────────────┐     │
│ TextVAEEncoder (Cola DLM)                          │     │
│  ├ global Perceiver → z_global  (B, G=16, dim)     │     │
│  ├ local  Perceiver → z_local_raw (B, L=64, dim)   │     │  Cola DLM
│  └ fusion: concat(z_local, mean(z_global)) → Linear│     │  auxiliary
│                                                    │     │  head
│ z = concat([z_global, z_local])  shape (B, G+L, dim)│    │
└────────────────┬───────────────────────────────────┘     │
                 │                                         │
                 ▼                                         │
       ┌─────────────────────┐                             │
       │ Block-causal mask   │                             │
       │  - global prefix is │                             │
       │    bidirectional    │                             │
       │  - local: causal    │                             │
       │    across chunks of │                             │
       │    block_size (=16),│                             │
       │    bidirectional    │                             │
       │    within a chunk   │                             │
       └──────────┬──────────┘                             │
                  │                                        │
                  ▼                                        │
┌────────────────────────────────────────────────────────┐  │
│ BlockCausalDiT (paper-style, lifted from official repo)│  │
│  - sinusoidal TimestepEmbedding → 3-layer SiLU MLP     │  │
│  - per-block AdaLN-Zero conditioning (scale+shift+gated│  │
│    residual; zero-init → block ≈ identity at step 0)   │  │
│  - DiT trunk: MaskedSelfAttn → MLPGeluTanh             │  │
│  - final AdaLN + zero-init projection out              │  │
│                                                        │  │
│  cola_prediction = "v"  (paper default, Flow Matching):│  │
│       z_t = (1-t)·z + t·ε,    target u_t = ε - z       │  │
│       L_cola = MSE(v_pred, u_t)                        │  │
│  cola_prediction = "x0" (legacy cosine schedule):      │  │
│       ᾱ = 1 - t²,   z_t = √ᾱ·z + √(1-ᾱ)·ε              │  │
│       L_cola = MSE(x0_pred, z)                         │  │
└────────────────────────────────────────────────────────┘  ┘

Total loss = L_mdm + L_path + L_aux_moe
           + repr_align_wt * L_align
           + cola_wt       * L_cola
```

Key properties:

- **Default does not change the student's training objective.** With
  `cola_detach_student=true` (default), the student is still shaped only
  by MDM + Repr-Align; the Cola head just learns to compress + denoise
  the resulting hidden states. Flip to `false` after warmup if you want
  Cola gradients to also shape the student.
- **Two latent scales.** 16 global tokens for semantics, 64 local
  tokens for detail, fused so each local latent carries global context.
  Mirrors the paper's hierarchical Text VAE split.
- **Block-causal is a strict superset of full-causal and full-
  bidirectional.** `cola_block_size=1` → pure causal,
  `cola_block_size=cola_num_local` → pure bidirectional. The default
  `16` gives short blocks: diffusion refines tokens in parallel inside
  a block, with strict left-to-right progression across blocks.

---

## 2. Files

| Path | Role |
|------|------|
| `veomni/models/cola_ldm/modules.py` | `PerceiverResampler`, `MaskedSelfAttention`, `CrossAttention`, `FeedForward`, `PreNorm`. Self-contained (no LDLM import). |
| `veomni/models/cola_ldm/text_vae.py` | `TextVAEEncoder` — global + local Perceivers + fusion. |
| `veomni/models/cola_ldm/block_causal_dit.py` | `make_block_causal_mask`, `BlockCausalDiT` (denoiser), `ColaDLMHead` (encoder + DiT + loss). |
| `veomni/models/cola_ldm/wrapper.py` | `ColaReprAlignWrapper` — `nn.Module` that wraps a base Repr-Align LM, runs the head from `hidden_states[cola_source_layer]`, and adds the loss. Pass-through for everything else (FSDP, optimizer, checkpointer). |
| `configs/pretrain/qwen3_6_35b_a3b_cola_ldm.yaml` | Ready-to-run 2-GPU config for 35B-A3B. |
| `tasks/train_torch.py` | Wrapping happens here, behind `if args.train.cola_wt > 0`. |
| `veomni/utils/arguments.py` | New `TrainingArguments` fields (`cola_wt`, `cola_num_global`, …). |

---

## 3. Run it

Two-GPU launch, FSDP1 sharding both the student and the Cola head:

```bash
export TOKENIZERS_PARALLELISM=false
torchrun --nnodes=1 --nproc-per-node=2 tasks/train_torch.py \
    configs/pretrain/qwen3_6_35b_a3b_cola_ldm.yaml
```

Knobs you'll touch most often (all under `train:` in YAML, or override
on the command line as `--train.cola_wt=…`):

| Knob | Default | Effect |
|------|---------|--------|
| `cola_wt` | `0.0` | Master switch. `0` disables Cola entirely. Start at `0.1–0.5`. |
| `cola_num_global` | `16` | Global semantic latents. More → richer summary, more compute. |
| `cola_num_local` | `64` | Local detail latents. Should be ≥ `cola_block_size`. |
| `cola_block_size` | `16` | Local chunk size for the block-causal mask. `1` = fully causal, `cola_num_local` = fully bidirectional. |
| `cola_encoder_depth` | `2` | Depth per Perceiver. Quadratic in trainable params; bump cautiously. |
| `cola_diffusion_depth` | `4` | DiT depth. |
| `cola_heads` | `8` | Attention heads (must divide `dim`). |
| `cola_source_layer` | `-3` | Which student hidden layer to compress. `-3` matches LDLM; try `-1` for end-state. |
| `cola_detach_student` | `true` | If `false`, Cola loss backprops into the student too. |
| `cola_log_hist_every` | `200` | Wandb histogram interval (`0` disables). |
| `cola_prediction` | `"v"` | DiT prediction target. `"v"` = Flow Matching velocity (paper alignment, `target = ε - z`). `"x0"` = x0-prediction MSE with cosine schedule (legacy / ablations). |

---

## 4. WandB metrics

The wrapper populates `outputs.loss_components` (the existing training
loop DP-reduces and logs each entry as `losses/<name>`):

| Scalar | Meaning |
|--------|---------|
| `losses/cola_diff` | Raw MSE diffusion loss (pre-weight). |
| `losses/cola_t_mean` | Mean sampled timestep this step (sanity-check the sampler). |
| `losses/cola_z_norm`, `cola_z_std`, `cola_z_mean`, `cola_z_max` | Combined latent geometry. Watch for collapse (`std → 0`) or divergence (`max → ∞`). |
| `losses/cola_z_global_norm`, `cola_z_local_norm`, `cola_z_global_std`, `cola_z_local_std` | Per-scale latent stats. If one scale's norm collapses to ~0, that stream is unused. |
| `losses/cola_pred_cosine` | Cosine between the DiT's prediction and its target (`target = ε - z` under FM, `target = z` under x0). Climbs from ~0 toward ~1 as the DiT learns. |
| `losses/cola_pred_snr` | `std(target) / std(target - pred)`. Pure SNR of the denoiser, prediction-objective-agnostic. |

Plus, only on rank 0:

| Metric | Cadence | Meaning |
|--------|---------|---------|
| `cola/grad_norm` | every step | L2 norm of the Cola head's grads (separate from the LM's overall grad norm). |
| `cola_hist/z_global` | every `cola_log_hist_every` | Histogram of all global latent activations. |
| `cola_hist/z_local` | every `cola_log_hist_every` | Histogram of all local latent activations. |

Per-layer Repr-Align cosine scalars already come from the Qwen3.5 / 3.6
modeling forward via `loss_components["repr_align"]` — unchanged.

Suggested WandB dashboard sections:
- **Repr-Align Health** → `losses/repr_align`, `losses/mdm`, `losses/path`
- **Cola Diffusion** → `losses/cola_diff`, `losses/cola_pred_cosine`, `losses/cola_pred_snr`, `losses/cola_t_mean`
- **Latent Geometry** → `losses/cola_z_*`, `cola_hist/*`
- **Training Stability** → `training/loss`, `training/grad_norm`, `cola/grad_norm`, `training/lr`

---

## 5. What's lifted from the official Cola-DLM repo

The DiT internals were cherry-picked from `/home/johndpope/Documents/GitHub/Cola-DLM` (the ByteDance Seed reference release for arXiv:2605.06548) and trimmed to what an auxiliary head actually needs:

- **`TimestepEmbedding`** — sinusoidal projection (`diffusers` convention, `flip_sin_to_cos=False`, `downscale_freq_shift=0`) followed by a 3-layer SiLU MLP. Replaces the original `Linear(1, dim)` time injection.
- **`AdaLN`** — AdaLN-Zero conditioning per DiT block: SiLU → Linear produces a (shift, scale) pair for the pre-norm and a gate for the residual; the final Linear is zero-initialised so each block starts as identity.
- **`MLPGeluTanh`** — GELU(`approximate="tanh"`) MLP, matching the paper's FFN.
- **Flow Matching loss + schedule** — `z_t = (1-t)·z + t·ε`, velocity target `u_t = ε - z`, MSE in velocity space (paper Eq. 2.1.7). Toggle via `cola_prediction`.
- **Zero-init output projection** — first-step output ≈ 0, stabilising warmup.

What was **not** lifted (deliberately):

- The NA variable-length mask (`create_na_block_causal_mask`) and KV cache — only matter at inference. The fixed `(G + L)` latent shape during training makes the simpler `make_block_causal_mask` sufficient.
- The HuggingFace `PreTrainedModel` / `PretrainedConfig` wrappers — the auxiliary head doesn't need standalone serialisation.
- RoPE on the latent axis — easy to add (`rotary-embedding-torch`) but not required; the block-causal mask alone gives the necessary structural inductive bias.
- The full Cola Text VAE (`modeling_cola_vae.py`) — replaced by our Perceiver-based `TextVAEEncoder`, since the upstream Repr-Aligned LM already produces high-quality bidirectional hidden states.

## 6. Extension points (Phase 2 / 3)

The head is structured so future upgrades don't require touching the
wrapper or trainer. Subclass `ColaDLMHead` and override one of:

| Method | Default | Swap in… |
|--------|---------|----------|
| `sample_timesteps(B, device, dtype)` | uniform on [0, 1] | Adaptive sampler (per-bucket loss EMA), importance-sampled t, etc. |
| `noise_schedule(t)` | FM (`a=1-t, b=t`) when `prediction_type="v"`; cosine when `"x0"` | Rectified flow, EDM, ImageNet-style v-prediction with a different schedule. |
| `compute_loss(pred, target)` | `F.mse_loss` | Huber, SPG (sandwiched policy gradient), weighted by SNR(t). |

`make_block_causal_mask(num_global, num_local, block_size)` is a free
function — swap it for a Swin-style local-window mask, periodic global
tokens, or hybrid AR-prefix patterns without touching the DiT.

Self-conditioning (feed previous `z_pred` as extra channel) is a
single-line addition to `BlockCausalDiT.forward` if/when you want it;
the LDLM implementation already shows the pattern. RoPE can be lifted
straight from `cola_dlm/modeling_cola_dit.py` (`TextRotaryEmbedding`).

---

## 7. Sanity checklist before kicking off a long run

1. `torchrun --nproc-per-node=2 ... cola_wt=0` → existing Repr-Align run; should match prior baseline.
2. Same, with `cola_wt=0.5` → confirm `losses/cola_diff` drops, `losses/cola_pred_cosine` climbs, `losses/repr_align` curve isn't dragged off.
3. Inspect `losses/cola_z_global_norm` and `losses/cola_z_local_norm` after ~1k steps. Both should be O(1)–O(10), neither at 0.
4. Flip `cola_detach_student: false` only after the head has clearly converged on cosine.
