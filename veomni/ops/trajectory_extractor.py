"""Pre-compute unmasking trajectories for d3LLM-style pseudo-trajectory distillation.

Runs diffusion_generate() on each training sample and records which tokens
are unmasked at each denoising step.  The saved trajectories are consumed
during Repr-Align training to replace random masking with teacher-guided
masking (see DataCollatorWithTrajectoryMasking).

Usage:
    python -m veomni.ops.trajectory_extractor \\
        --model_path Qwen/Qwen3-1.7B \\
        --data_path /path/to/data.jsonl \\
        --output_dir /path/to/trajectories \\
        --max_seq_len 2048 --steps 256
"""

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

from veomni.models.auto import build_foundation_model
from veomni.models.transformers.qwen2.generation_utils import mdm_generate


@torch.no_grad()
def extract_trajectory(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    max_new_tokens: int = 128,
    steps: int = 64,
    temperature: float = 0.7,
    top_k: int = 200,
    alg: str = "entropy",
    alg_temp: float = 0.6,
) -> list:
    """Run one diffusion decode and return the per-step unmasking trajectory.

    The trajectory is a list of token-ID sequences (Python lists) at each
    denoising step, starting from the step-0 state and ending at the final
    decoded sequence.  Tokens that are *still masked* at step ``i`` have
    value ``mask_token_id`` in ``trajectory[i]``.

    This is analogous to d3LLM's ``generate_teacher_model_trajectory()`` but
    uses the codebase's existing ``mdm_generate`` (standalone, no mixin).
    Because we cannot rely on ``output_history`` being efficient, we run a
    custom loop that records ``x`` at each step.
    """
    device = input_ids.device
    pad_token_id = getattr(model.config, "pad_token_id", None)

    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
    gen_attention_mask = (
        (x != pad_token_id).long() if pad_token_id is not None else None
    )
    timesteps = torch.linspace(1, 1e-3, steps + 1, device=device)

    trajectory = []
    for i in range(steps):
        mask_index = x == mask_token_id
        if not mask_index.any():
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=x, attention_mask=gen_attention_mask, is_causal=False
            )
        logits = outputs.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]
        t, s = timesteps[i], timesteps[i + 1]

        probs = torch.softmax(mask_logits.float(), dim=-1)
        if temperature > 0:
            probs = torch.softmax(mask_logits.float() / temperature, dim=-1)

        if top_k and top_k > 0:
            top_k_val = min(top_k, probs.size(-1))
            indices_to_remove = probs < torch.topk(probs, top_k_val)[0][
                ..., -1, None
            ]
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        if alg == "entropy":
            log_probs = torch.log(probs.clamp(min=1e-10))
            confidence = (probs * log_probs).sum(dim=-1)
        else:
            confidence = torch.gather(
                probs, -1, probs.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)

        x0 = (
            torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
                confidence.shape
            )
            if temperature > 0
            else probs.argmax(dim=-1)
        )

        num_masked = mask_index.sum(dim=-1, keepdim=True)
        gamma = 1 - s / t
        num_to_unmask = (num_masked * gamma).long()

        full_confidence = torch.full_like(
            x, -float("inf"), device=device, dtype=confidence.dtype
        )
        full_confidence[mask_index] = confidence

        if alg_temp and alg_temp > 0:
            scaled_logits = full_confidence / alg_temp
            uniform = torch.rand_like(scaled_logits).clamp_(
                min=1e-20, max=1 - 1e-20
            )
            gumbel_noise = -torch.log(-torch.log(uniform))
            scores = scaled_logits + gumbel_noise
            _, unmask_indices = torch.topk(
                scores, num_to_unmask.max(), dim=1
            )
        else:
            _, unmask_indices = torch.topk(
                full_confidence, num_to_unmask.max(), dim=1
            )

        rows = torch.arange(x.size(0), device=device).unsqueeze(1)
        unmask_selection = torch.zeros_like(x, dtype=torch.bool)
        unmask_selection[rows, unmask_indices] = True
        unmask_selection = unmask_selection & (
            torch.cumsum(unmask_selection.long(), dim=-1) <= num_to_unmask
        )

        proposals = torch.full_like(x, fill_value=mask_token_id)
        proposals[mask_index] = x0
        x[unmask_selection] = proposals[unmask_selection]

        trajectory.append(x[0].cpu().tolist())

    return trajectory


def extract_and_save(
    model_path: str,
    data_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    max_new_tokens: int = 128,
    steps: int = 64,
    max_examples: Optional[int] = None,
    batch_size: int = 1,
):
    """Run trajectory extraction over a dataset and save to disk.

    Output: one JSONL file, each line:
        {"idx": int, "trajectory": [[tok_id, ...], ...], "nfe": int}
    """
    os.makedirs(output_dir, exist_ok=True)

    model = build_foundation_model(
        model_path,
        weights_path=model_path,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    model.cuda().eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    mask_token_id = tokenizer.mask_token_id

    # Build a plain dataset without the full data pipeline complexity
    from datasets import load_dataset as hf_load_dataset
    import os as _os
    _ext = _os.path.splitext(data_path)[1].lstrip(".")
    _ext = "json" if _ext == "jsonl" else _ext
    dataset = hf_load_dataset(_ext, data_files=data_path, split="train")

    limit = len(dataset)
    if max_examples is not None:
        limit = min(limit, max_examples)

    results = []
    for idx in range(limit):
        sample = dataset[idx]
        text = sample.get("text", sample.get("content", sample.get("input_ids", "")))
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            tokens = tokens[:max_seq_len]
        else:
            tokens = text[:max_seq_len]

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()
        traj = extract_trajectory(
            model,
            input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=min(max_new_tokens, max_seq_len // 2),
            steps=steps,
        )
        results.append({"idx": idx, "trajectory": traj, "nfe": len(traj)})
        print(f"  [{idx+1}/{limit}] nfe={len(traj)}")

    out_path = os.path.join(output_dir, "trajectories.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} trajectories to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract d3LLM-style trajectories from a Repr-Align model"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    extract_and_save(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        max_examples=args.max_examples,
    )
