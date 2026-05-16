# Cola DLM on Open-dLLM — Roadmap & Validation Plan

> Companion document to [`cola_ldm.md`](cola_ldm.md), which explains the
> current architecture and how to run it. This doc explains **what is
> still missing**, the pathways to fill the gaps, and the wandb
> experiments that tell you which pathway is paying off.

---

## TL;DR

We shipped Phase 1: a Cola-paper-aligned auxiliary head
(`TextVAEEncoder` + `BlockCausalDiT` with Flow Matching) plugged on top
of Repr-Align. It trains; it does **not** yet generate text on its own.

The next decisions are:

1. **Validate Phase 1.** Does the Cola loss actually help the
   Repr-Aligned student, or is it a free-floating regulariser? Wandb
   tells us this in the first few thousand steps.
2. **Pick a generation pathway** based on what Phase 1 shows:
   - **A.** Discard Cola at inference — student carries everything.
   - **B.** Add a `TextVAEDecoder` and generate via the Cola DiT prior.
   - **C.** Use Cola latents as conditioning for the student's MDM
     `diffusion_generate()`.
   - **D.** Port the full official Cola Text VAE end-to-end.

Each pathway has a concrete build list, an expected outcome, and a set
of wandb panels that decide success.

---

## 1. Where Qwen3.6 fits at every stage

|                    | Training                                                                           | Inference (today)                                              |
| ------------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Qwen3.6 student    | Updated by MDM + Repr-Align (+ Cola if `cola_detach_student=false`)                | **Runs all generation** via `model.diffusion_generate()`       |
| Qwen3.6 teacher    | Frozen; provides per-layer cosine targets                                          | Discarded                                                      |
| Cola head (~600 M) | Trained from random init                                                           | **Currently unused** — see §3 for paths to wire it in          |

So "leveraging Qwen3.6" is honest: the Repr-Aligned student is the
inference model on every pathway. The pathways differ in *what extra
machinery wraps that student* at sampling time.

---

## 2. Phase 1 validation — does Cola help at all?

**Goal**: decide whether keeping the Cola head around past training is
worth the engineering cost.

### Setup

Run the same dataloader / seed twice:

| Run | Config override                                              | What we measure                       |
| --- | ------------------------------------------------------------ | ------------------------------------- |
| R0  | `--train.cola_wt=0`                                          | Baseline Repr-Align curves            |
| R1  | `--train.cola_wt=0.5` (default in `qwen3_6_35b_a3b_cola_ldm.yaml`) | Repr-Align + Cola auxiliary           |

Train both for **≥5 k global steps** on the FineWeb-100k sample. The
Cola head's `proj_out` is zero-init so R1 starts numerically identical
to R0 for the first step; any divergence after is signal.

### WandB panels to compare R0 vs R1

| Panel                  | Metric                                                                                                                                | What "good" looks like                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Repr-Align health**  | `losses/repr_align`, `losses/mdm`, `losses/path`                                                                                       | R1 reaches ≤ R0 curves at the same step count. **Hard fail** if R1 is meaningfully worse — Cola is destabilising training.                                              |
| **Cola convergence**   | `losses/cola_diff`, `losses/cola_pred_cosine`, `losses/cola_pred_snr`                                                                  | `cola_diff` drops monotonically; `cola_pred_cosine` climbs from ~0 toward ~0.7+ within a few k steps; `cola_pred_snr` climbs above 1.0.                                  |
| **Latent geometry**    | `losses/cola_z_norm`, `cola_z_global_norm`, `cola_z_local_norm`, `cola_z_std`, `cola_z_global_std`, `cola_z_local_std`                | Both `_global` and `_local` norms stay O(1)–O(10). Hard fail if either collapses to ~0 (stream unused) or explodes (>1e3).                                              |
| **Latent shape**       | `cola_hist/z_global`, `cola_hist/z_local` (every `cola_log_hist_every=200` steps)                                                     | Distribution stays roughly Gaussian-ish, no spike at 0, no heavy tails developing.                                                                                       |
| **Optimisation**       | `training/grad_norm`, `cola/grad_norm`, `training/loss`                                                                               | `cola/grad_norm` is comparable to LM `grad_norm`. Any saturation at `max_grad_norm` (1.0) on every step means LR is too high for the head.                              |

### Decision rules (after ~5 k steps)

| Outcome                                                                  | Interpretation                                                                  | Next pathway                                                                            |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| R1 `losses/repr_align` consistently **below** R0 by ≥10%                 | Cola is a useful auxiliary; the student's hidden states are more structured       | **Pathway A or B**. Cheapest: A. Most upside: B.                                          |
| R1 ≈ R0 on `losses/repr_align`, but Cola converges (`cola_pred_cosine > 0.5`) | Cola is a free-floating prior — useful if/when we want it for generation         | **Pathway B or C**, depending on appetite for code                                       |
| R1 `losses/repr_align` **worse** than R0                                 | Cola is competing with Repr-Align objectives                                     | Lower `cola_wt`, confirm `cola_detach_student=true`, retry. If still bad, drop the head |
| `cola_pred_cosine` stuck near 0 / `cola_z_*_norm` collapses              | DiT or encoder not learning — check schedule, LR, init, FSDP wrapping            | Debug before any pathway                                                                |

### Ablation matrix (run after Phase 1 if results are promising)

| Knob                  | Values to sweep                              | Hypothesis                                                                                                  |
| --------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `cola_prediction`     | `"v"`, `"x0"`                                | Paper claims `v` (Flow Matching) converges faster and is more numerically stable                            |
| `cola_wt`             | `0.1`, `0.25`, `0.5`, `1.0`                  | Find the largest weight that doesn't drag `losses/repr_align`                                                |
| `cola_block_size`     | `1`, `4`, `16`, `64`                         | `1` = fully causal (closest to AR), `64` = fully bidirectional (closest to LDLM). Sweet spot is empirical    |
| `cola_detach_student` | `true`, `false` (only after R1 looks healthy) | `false` lets Cola gradients shape the student. Big upside if stable, big risk if not                         |
| `cola_source_layer`   | `-1`, `-3`, `-6`                             | Earlier layers = more abstract; later = closer to lm_head. LDLM uses `-3`                                    |

---

## 3. Generation pathways

### Pathway A — Discard Cola at inference (no new code)

**What you do.** Treat the Cola head as a training-only regulariser.
At inference, load just the student and call
`model.diffusion_generate()` as today.

**Cost.** Zero. Existing `eval/eval_completion/run_eval.sh` works
unchanged. Cola head weights are dropped from the HF checkpoint by
filtering `state_dict` keys.

**When to choose.** Phase 1 shows R1 ≤ R0 on `losses/repr_align`
(Cola helped training) **AND** you don't want to build a decoder.

**Validation experiments.**

| Test                       | How                                                                                | Pass criterion                                                              |
| -------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **MDM perplexity**         | `lm-evaluation-harness` on `wikitext-2`, both R0 and R1 student checkpoints       | R1 student perplexity ≤ R0 student perplexity                               |
| **Code completion**        | `eval/eval_completion/run_eval.sh` on HumanEval pass@1 / pass@10                  | R1 ≥ R0; ideally R1 within ~5 % of LLaDA-8B / Dream-7B numbers              |
| **Generation samples**     | `python sample.py` with 5 fixed prompts, both checkpoints                          | Subjective: R1 samples are at least as coherent as R0                       |

### Pathway B — Add a `TextVAEDecoder` (recommended next build)

**What you do.** Mirror the encoder: a small Perceiver that takes
`(B, G+L, dim)` latents and produces `(B, T, dim)` hidden states, then
feed those through the **existing** Qwen3.6 `lm_head` to get logits.

**New module.** `veomni/models/cola_ldm/text_vae.py::TextVAEDecoder`
(~200 M params at dim=2048).

**New training term.** Reconstruction loss
`L_recon = CE(lm_head(decoder(encoder(h))), input_ids)`.
Wired into `ColaDLMHead.forward` with its own weight
`cola_recon_wt: float = 1.0`.

**New inference path.** `cola_generate(prompt, num_blocks)`:
1. Encode the prompt to a prefix latent `z^pre = encoder(LM(prompt).hidden_states[-3])`.
2. Sample noise per block; integrate `BlockCausalDiT` Euler steps to
   transport noise → clean block latent.
3. Concatenate `[z^pre, generated_blocks]`, run `decoder(z)` →
   hidden states → `lm_head` → tokens.

**Effort.** ~1 day of build, ~1 day of debug. Decoder code can be
copied almost verbatim from the encoder. The new inference loop is the
main work.

**WandB validation panels.**

| Panel                       | New metrics                                                                                                | What "good" looks like                                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Reconstruction quality**  | `losses/cola_recon`, `cola_recon_token_acc` (argmax accuracy of decoder logits vs `input_ids`)             | `cola_recon` drops below 2.5 (ln(vocab)/2) within ~5 k steps; `cola_recon_token_acc > 50%` within ~10 k steps                            |
| **Round-trip fidelity**     | `cola_roundtrip_cos`: cosine between original hidden state `h` and `decoder(encoder(h))`                    | Climbs to >0.8 by ~20 k steps                                                                                                            |
| **Generation quality**      | Custom eval: sample 100 generations every 5 k steps, score perplexity under the frozen teacher              | Generation perplexity descends, converging on or below the student's MDM perplexity                                                      |
| **End-to-end HumanEval**    | `cola_generate(...)` swapped for `diffusion_generate(...)` in `eval/eval_completion/`                       | Pass@1 within 50 % of pathway A as a first milestone; competitive after tuning                                                            |

**Decision rules.**
- If `cola_recon_token_acc` plateaus < 30 %, the decoder is too small or
  the encoder bottleneck is too aggressive — bump `cola_num_local` or
  decoder depth.
- If round-trip cosine plateaus < 0.6, encoder/decoder are not aligned
  — check that decoder receives the same latent that the DiT denoises
  to (i.e. clean, not `z_t`).
- If `cola_generate` perplexity diverges from MDM perplexity, the DiT
  is producing latents the decoder hasn't seen during training — sample
  noise from the decoder's training distribution.

### Pathway C — Cola latents as conditioning for `diffusion_generate()`

**What you do.** Keep the existing MDM generation loop on the student.
Inject Cola latents as soft conditioning into every LM forward.

**Concretely**:
1. Sample `z_0` with the trained DiT prior.
2. At each MDM denoising step, prepend `z_0` (projected through a small
   `nn.Linear(dim, dim)` adapter) as a soft-prompt prefix to the
   student's input embeddings.
3. Run the existing `diffusion_generate()` algorithm.

**New module.** `ColaSoftPromptAdapter` — a `nn.Linear(dim, dim)` per
block of `z_0`, or one shared. Trained jointly with everything else.

**New training term.** Optional: re-train end-to-end with the soft
prompt active during the MDM forward, so the student learns to use
the conditioning. Without this, the inference-time injection is OOD.

**Effort.** ~2 days build + retraining time. The conditioning hook
inside Qwen3.6 attention is the fiddly part.

**WandB panels.**

| Panel                       | Metric                                                                                                                  | Good                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Conditioning utility**    | `losses/mdm_with_cond` vs `losses/mdm_no_cond` (toggle the adapter on/off per micro-batch)                              | `mdm_with_cond` is lower, monotonically increasing the gap as training progresses                  |
| **Sample diversity**        | Pairwise BLEU / self-BLEU on 100 generations per prompt, with vs without `z_0`                                          | Conditioning gives controllably different generations from different `z_0` samples                |
| **Long-range coherence**    | Perplexity of held-out continuations under the frozen teacher, with vs without conditioning                              | Conditioning improves continuation perplexity by ≥5 %                                              |

**When to choose.** You want the Cola DiT's strengths (block-causal
planning, long-range coherence) without the engineering cost of a full
decoder. Trade-off: requires retraining the student.

### Pathway D — Full official Cola Text VAE port

**What you do.** Implement `ColaTextVAEModel` from
`/home/johndpope/Documents/GitHub/Cola-DLM/cola_dlm/modeling_cola_vae.py`
verbatim, run two-stage training (Stage 1: Text VAE alone with
reconstruction + KL + BERT-masking; Stage 2: joint VAE + DiT with Flow
Matching), and use the official inference pipeline.

**Effort.** ~1 week build, ~order-of-magnitude more code than Pathway
B. Replaces, rather than complements, the Repr-Align path.

**When to choose.** Phase 1 shows Cola is helping a lot AND you want
paper-grade results AND you're prepared to break compatibility with
the Repr-Align training loop.

**Validation.** Reproduce the paper's RQ4 8-task benchmark
(LAMBADA, MMLU, OBQA, HellaSwag, RACE, SIQA, SQuAD, Story Cloze) via
`scripts/run_benchmark.sh` from the reference repo, comparing against
the released ByteDance checkpoint.

---

## 4. Decision tree

```
            ┌──────────────────────────────┐
            │ Phase 1: train R0 vs R1      │
            │  (5–10 k steps, FineWeb-100k)│
            └──────────────┬───────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
   R1 better          R1 ≈ R0             R1 worse
   on repr_align      on repr_align       on repr_align
       │                   │                   │
       ▼                   ▼                   ▼
   Want Cola at        Want Cola at        Lower cola_wt;
   inference?          inference?          if still bad,
       │                   │               drop the head
   ┌───┴────┐          ┌───┴────┐
   │        │          │        │
   No       Yes        No       Yes
   │        │          │        │
   ▼        ▼          ▼        ▼
   A     B (or D)      A       B or C
                       ↑
                  "free-floating prior";
                  retain if cheap, build
                  on it only if needed
```

---

## 5. Concrete work list (Pathway B, the recommended next build)

In dependency order. Each item ships with the wandb metrics that prove
it works.

1. **`TextVAEDecoder` module** — `veomni/models/cola_ldm/text_vae.py`.
   Mirror of `TextVAEEncoder`: cross-attends learned queries
   (`num_queries=max_seq_len`) onto `(B, G+L, dim)` latents, returns
   `(B, T, dim)` hidden states. ~200 M params at dim=2048.
   - **Test**: shape contract + a forward pass on random latents
     yields finite outputs.
   - **Wandb on first integration run**: `losses/cola_recon` curve.

2. **Reconstruction loss in `ColaDLMHead.forward`** — wire
   `lm_head_weight` (passed in from the wrapper) to compute
   `L_recon = F.cross_entropy(decoder_logits, input_ids)`. Return both
   `L_cola_diff` and `L_recon`, sum with `cola_recon_wt`.
   - **Wandb**: `losses/cola_recon`, `losses/cola_recon_token_acc`.

3. **Wrapper plumbing** — `ColaReprAlignWrapper` forwards the lm_head
   weight reference to the head (it's tied across LM + decoder).
   Surface `cola_recon` in `loss_components`.
   - **Test**: a 100-step run on FineWeb shows `cola_recon` strictly
     decreasing.

4. **`cola_generate` inference function** — new file
   `veomni/models/cola_ldm/inference.py`. Implements the paper's
   three-step recipe: prefix-encode, block-wise DiT integration,
   decoder forward → lm_head → tokens.
   - **Test**: greedy generation from a fixed seed prompt produces the
     same tokens twice (determinism).

5. **`tasks/cola_generate.py` CLI** — small wrapper around
   `cola_generate` so you can run `python tasks/cola_generate.py
   --ckpt <path> --prompt "..."` standalone.
   - **Test**: produces non-degenerate text on 5 fixed prompts at
     ~5 k, ~20 k, and ~50 k training steps.

6. **HumanEval evaluation harness** — fork
   `eval/eval_completion/eval_single.py` to use `cola_generate` instead
   of `diffusion_generate`. Compare pass@1 / pass@10 against the
   Pathway A baseline.
   - **Wandb**: log `eval/cola_humaneval/pass@1`, `pass@10` after
     every `save_steps`.

**Estimated total effort**: 2–3 days of focused work. Each step is
independently testable.

---

## 6. WandB dashboard layout

Suggested sections, in priority order:

1. **Phase 1 sanity** — `losses/repr_align`, `losses/mdm`,
   `losses/path`, `training/loss`.
2. **Cola health** — `losses/cola_diff`, `losses/cola_pred_cosine`,
   `losses/cola_pred_snr`, `losses/cola_t_mean`.
3. **Latent geometry** — `losses/cola_z_*`, `cola_hist/*`.
4. **Optimisation** — `training/grad_norm`, `cola/grad_norm`,
   `training/lr`.
5. **(Pathway B)** **Decoder** — `losses/cola_recon`,
   `cola_recon_token_acc`, `cola_roundtrip_cos`.
6. **(Pathway B/C)** **End-to-end eval** —
   `eval/cola_humaneval/pass@1`, `eval/cola_humaneval/pass@10`,
   sample-quality screenshots logged via `wandb.Html`.

---

## 7. Open questions to answer with experiments, not by argument

- Does the Repr-Aligned student benefit more from Cola's compressed
  latents at layer −3, −1, or somewhere else?
- Is the block-causal mask actually adding value over a fully
  bidirectional mask on the local stream? (`cola_block_size=cola_num_local`)
- Does Flow Matching converge meaningfully faster than x0-prediction
  on this setup, or is the x0 baseline good enough at our scale?
- Can `cola_detach_student=false` be turned on without destabilising
  Repr-Align?

Each is a single ablation; each is a single wandb panel diff. Don't
prejudge.

---

## 8. Inference-speed calibration (do not overclaim)

Measured on the **official Cola-DLM release** (`/home/johndpope/Documents/GitHub/Cola-DLM`,
their 1.8B DiT + Text VAE, no Open-dLLM involvement) on an
**RTX PRO 4000 Blackwell**, bf16, `batch=3`, `max_new_tokens=32`:

| Setting                                          | Throughput                | Per-token wall      |
| ------------------------------------------------ | ------------------------- | ------------------- |
| `timestep_num=16`, `block_size=4` (defaults)     | 37.8 tok/s aggregate / 12.6 per prompt | ~80 ms / token |
| `timestep_num=8` (extrapolated)                  | ~75 tok/s aggregate       | ~40 ms / token      |
| `timestep_num=4` (extrapolated)                  | ~150 tok/s aggregate      | ~20 ms / token      |

**Why it's not blazing.** At defaults you pay
`timestep_num / block_size = 4` DiT forwards per generated token. The
diffusion-LM win is **fewer forwards than AR** *only* when
`timestep_num < block_size`. The official defaults are tuned for
quality, not throughput.

**What this means for pathways B/C/D on our 35B-A3B target:**

| Hardware            | Settings                       | Expected per-prompt tok/s | Notes                                           |
| ------------------- | ------------------------------ | ------------------------- | ----------------------------------------------- |
| RTX PRO 4000        | `timestep_num=16, block_size=4` (defaults) | ~4–6                | 35B-A3B per-forward ≈ 3× the paper's 1.8B       |
| RTX 5090            | same                           | ~10–15                    | ~2.5× the PRO 4000 per-forward                  |
| RTX 5090            | `timestep_num=8, block_size=8` | ~25–35                    | Approximate AR parity on per-token wall         |
| RTX 5090            | `timestep_num=4, block_size=16`| ~60–90                    | Headline-number territory; expect quality drop |

**Bench command (official repo)**:

```bash
CUDA_VISIBLE_DEVICES=<id> .venv/bin/python scripts/bench_tps.py \
    --timestep_num 8
```

**Decision implication.** Throughput is **not** a reason to choose
Pathway B/C/D over Pathway A. Choose B/C/D only if Phase 1 + the
pathway-specific validation (quality, controllability, long-range
coherence) show a clear benefit. Until generation quality wins are
demonstrated, the Repr-Aligned student via `diffusion_generate()`
is both simpler and faster.

---

## 9. What this doc deliberately does NOT cover

- Inference-time KV cache / NA variable-length mask. Those become
  important once Pathway B is shipped; until then the simpler
  fixed-shape mask is fine.
- Distillation / SPG / RL post-training. Phase 3 from the original
  proposal — out of scope until a working generation path exists.
- Multi-modal (text+image) extensions. The reference repo has a
  preliminary text+image module; not relevant to this codebase yet.
