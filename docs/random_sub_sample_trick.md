# Random Sub-Sampling Trick for Repr-Align Memory Reduction

## Origin

Inspired by the FSRT paper (CVPR 2024, [arXiv:2404.09736 §5.58](https://arxiv.org/pdf/2404.09736#page=5.58)):

> Instead of computing the perceptual loss on the full feature map, randomly crop a
> 128×128 sub-square and compute loss only there. Over training, every spatial region
> gets covered. Memory drops ~4× because gradient graphs are proportional to area.

We apply the same principle to **Repr-Align** training in **Open-dLLM**
([arXiv:2605.06885](https://arxiv.org/pdf/2605.06885)) — targeting the **Qwen3**
model family specifically.

---

## Where the Memory Bottleneck Is

In `modeling_qwen3.py` ~lines 1003-1015, the alignment loss is computed over **all valid
tokens × all alignment layers**:

```python
loss_mask = (labels != IGNORE_INDEX)  # (L,)
if loss_mask.any():
    # h is shape [1, L, D], concatenate selected layers and permute to [L, num_layers, D]
    student_stacked = torch.cat([h[..., :-1, :] for h in student_hidden_states], dim=0).permute(1, 0, 2)
    student_stacked = student_stacked[loss_mask]  # [V, num_layers, D]
    teacher_stacked = torch.cat([h[..., :-1, :] for h in teacher_hidden_states], dim=0).permute(1, 0, 2)
    teacher_stacked = teacher_stacked[loss_mask]  # [V, num_layers, D]

    repr_align_loss = repr_align_loss_fn(student_stacked, teacher_stacked)
```

For Qwen3 (Qwen3.6 35B-A3B is a MoE, but for dense variants like Qwen3-14B or
Qwen3-32B: num_layers ≈ 40-64, D = 5120-7168):

```
V × num_layers × D × 2 (student + teacher)
≈ 4096 × 40 × 7168 × 2 × 2 bytes
≈ 4.7 GB just for the stacked hidden states at alignment time
```

But the **real cost** is the gradient graph: all intermediate activations from both
student *and* teacher forward passes are retained for backprop through the alignment
loss. With `output_hidden_states=True`, every layer's output must be preserved.

**Bottleneck**: even though Qwen3 already has `self.align_layers` to select *which*
layers participate (line 999-1001), it still uses **all valid tokens V** in the
sequence. For long sequences (8K+), V dominates.

---

## The Trick: Sub-Sample Tokens for the Alignment Loss

The alignment loss is a **per-position** cosine similarity, averaged to a scalar:

```python
def repr_align_loss_fn(z1, z2):
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    cosine_sim = (z1_norm * z2_norm).sum(dim=-1)  # per-position, per-layer
    return 1.0 - cosine_sim.mean()                 # averaged to scalar
```

Since the reduction is `mean()`, sub-sampling any subset of the `[V, num_layers, D]`
tensor produces an **unbiased estimate** of the same loss gradient. Over enough steps,
every token position sees gradients.

### Implementation sketch

In `modeling_qwen3.py`, replace the alignment block (lines 1011-1015) with:

```python
# Compute alignment loss with random token sub-sampling
sub_sample_ratio = 0.25  # hyperparameter — tune for memory vs. convergence

num_valid = student_stacked.size(0)  # V
num_sample = max(1, int(num_valid * sub_sample_ratio))
sample_indices = torch.randperm(num_valid, device=student_stacked.device)[:num_sample]

z1 = student_stacked[sample_indices]    # [num_sample, num_layers, D]
z2 = teacher_stacked[sample_indices]    # [num_sample, num_layers, D]

repr_align_loss = repr_align_loss_fn(z1, z2)
```

**What this changes:**
| Before | After |
|---|---|
| Align loss on all V tokens | Align loss on V × ratio tokens |
| Gradient graph retains all V tokens' hidden states | Gradient graph retains only sub-sampled tokens' hidden states |
| Peak memory dominated by full hidden state stack | Peak memory reduced ~4× for the alignment branch |
| Forward/CE loss unchanged (still full sequence) | Forward/CE loss unchanged |

### What stays full (no saving from this trick)

- **Student forward pass** (bidirectional attention, all layers) — still runs on full
  sequence because the MDM token loss needs full logits
- **Teacher forward pass** (causal attention) — still runs on full sequence
- **Token-level CE / MDM loss** (`causallm_loss_function`) — needs full logits

These dominate compute anyway. The saving is specifically in **gradient memory for the
alignment branch**, which can be the tipping point between fitting or OOM'ing on
longer sequences.

---

## Combining with existing layer selection

Qwen3 already has `align_layers` (line 999) to select layers. The sub-sampling is
**orthogonal** — they compose nicely:

```
Before:  V tokens × L full layers                      → O(V·L·D)   memory
Now:     V×ratio tokens × L (from align_layers) layers → O(V·ratio·L·D) memory
```

If you also want to randomise *which* layers per step (beyond the static
`align_layers` selection), see the alternative below.

---

## Alternative: Sub-Sample by Layer, Not Token

Instead of sampling tokens, randomly sample alignment layers from `align_layers`:

```python
# Align on a random subset of layers each step
num_align_layers = len(align_layers)  # e.g., 8 configured
num_sample_layers = 2                 # use 2 per step
layer_indices = torch.randperm(num_align_layers, device=student_stacked.device)[:num_sample_layers]

z1 = student_stacked[:, layer_indices, :]  # [V, num_sample, D]
z2 = teacher_stacked[:, layer_indices, :]
```

Each step aligns only 2 of 8 layers. Over training, all configured layers get aligned.

**You can combine both** — sub-sample tokens and layers:

```python
z1 = student_stacked[sample_indices][:, layer_indices, :]  # [V', L', D]
```

Max savings: `0.25 × 0.25 = 16×` reduction in alignment gradient memory.

---

## Qwen3-Specific Details

Location in `modeling_qwen3.py`:
- `repr_align_loss_fn` at line 59
- `align_layers` config at line 839-842 (selects which layer outputs to return from
  `output_hidden_states`)
- Alignment computation block at lines 1003-1015 (target for the patch)
- Loss weighting at line 1076: `loss = loss + repr_align_wt * repr_align_loss`

---

## Key Insight

The loss is `1.0 - cos_sim.mean()` — a **mean-pooled per-position, per-layer
measure**. Mean pooling is linear, so:

```
E[sub_sampled_loss] = full_loss
```

The gradient from any single step is noisy, but unbiased. This is the same argument
that makes dropout, RandAugment, and RandomErasing work — stochastic regularization
with unbiased expectation.

---

## Trade-offs

| Parameter | Low value | High value |
|---|---|---|
| `sub_sample_ratio` | Faster steps, noisier gradients | Slower steps, cleaner gradients |
| Layer sampling count | Max memory saving | Min memory saving |

Recommendation: start with `sub_sample_ratio=0.25` and the existing `align_layers`
static selection (don't randomise layers yet). Monitor loss curves. If convergence
looks identical to full alignment, keep the setting. If noisier, dial up gradually.
Add per-step layer randomisation only if you need more memory headroom.
