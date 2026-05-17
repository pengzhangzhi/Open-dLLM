"""Drop-in replacement for the live Repr-Align teacher.

Repr-Align's "teacher" is a frozen snapshot of the student's initialisation —
its hidden states for a given input are deterministic and never change for the
whole training run. Recomputing them every step (the current `make_teacher=True`
path via `copy.deepcopy(model)`) is pure waste.

`CachedTeacher` reads anchor tensors pre-computed by `scripts/precompute_anchor.py`
and surfaces them via the same interface that `modeling_qwen3.py` already calls
on `self.teacher_model` — namely `forward(input_ids, ...)` returning an object
with `.hidden_states` (a tuple indexed by layer) and `.loss`.

Only the layer indices listed in the cache manifest are populated; other entries
in the tuple are `None`. `modeling_qwen3.py` must subset to `align_layers` before
stacking, which it does when `align_layers` is set on the model.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import safe_open
from torch import nn


class _CachedTeacherOutput:
    """Minimal stand-in for `BaseModelOutputWithPast` — only the attrs that
    `modeling_qwen3.py` reads from teacher outputs."""

    __slots__ = ("hidden_states", "loss")

    def __init__(self, hidden_states: tuple, loss: Optional[torch.Tensor] = None):
        self.hidden_states = hidden_states
        self.loss = loss


def _hash_row(input_ids_row: torch.Tensor) -> str:
    return hashlib.sha256(input_ids_row.cpu().numpy().tobytes()).hexdigest()[:16]


class CachedTeacher(nn.Module):
    def __init__(self, cache_dir: str, num_hidden_layers: int, hidden_size: int):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Anchor cache manifest missing at {manifest_path}. "
                f"Run scripts/precompute_anchor.py first."
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        # Sanity-check the cache matches the student we're plugging it into.
        if self.manifest["num_hidden_layers"] != num_hidden_layers:
            raise ValueError(
                f"Anchor cache has num_hidden_layers={self.manifest['num_hidden_layers']} "
                f"but student model has {num_hidden_layers}. Re-run precompute_anchor."
            )
        if self.manifest["hidden_size"] != hidden_size:
            raise ValueError(
                f"Anchor cache has hidden_size={self.manifest['hidden_size']} "
                f"but student model has {hidden_size}. Wrong cache for this model."
            )
        self.cached_layers: list[int] = list(self.manifest["layers"])
        self.num_hidden_layers = num_hidden_layers
        # +1 because HF includes embedding output at index 0
        self._tuple_len = num_hidden_layers + 1

    def _shard_path(self, h: str) -> Path:
        return self.cache_dir / h[:2] / f"{h}.safetensors"

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Return a mock CausalLMOutputWithPast-like object with sparse
        `hidden_states` populated only at indices in `self.cached_layers`."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        bsz = input_ids.size(0)
        device = input_ids.device

        # Gather per-row, per-layer tensors from disk.
        per_layer_rows: dict[int, list[torch.Tensor]] = {li: [] for li in self.cached_layers}
        for b in range(bsz):
            h = _hash_row(input_ids[b])
            p = self._shard_path(h)
            if not p.exists():
                raise KeyError(
                    f"Anchor cache miss for hash {h} (row {b} of batch). "
                    f"Was this example included in the precompute run? Path: {p}"
                )
            with safe_open(str(p), framework="pt") as f:
                for li in self.cached_layers:
                    per_layer_rows[li].append(f.get_tensor(f"hidden_layer_{li}"))

        # Stack to [B, S, D] per layer, move to model device.
        stacked = {
            li: torch.stack(per_layer_rows[li], dim=0).to(device=device)
            for li in self.cached_layers
        }

        # Build the full-length tuple, putting cached tensors at the right
        # indices and None elsewhere. modeling_qwen3.py with align_layers set
        # will only touch the cached indices.
        hidden_states = tuple(
            stacked[li] if li in stacked else None for li in range(self._tuple_len)
        )
        return _CachedTeacherOutput(hidden_states=hidden_states, loss=None)
