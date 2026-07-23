"""Drop-in replacement for the live Repr-Align teacher.

Repr-Align's "teacher" is a frozen snapshot of the student's initialisation —
its hidden states for a given input are deterministic and never change for the
whole training run. Recomputing them every step (the current `make_teacher=True`
path via `copy.deepcopy(model)`) is pure waste.

`CachedTeacher` reads anchor tensors pre-computed by `scripts/precompute_anchor.py`
and surfaces them via the same interface that `modeling_qwen3.py` already calls
on `self.teacher_model` — namely `forward(input_ids, position_ids, ...)` returning
an object with `.hidden_states` (a tuple indexed by layer) and `.loss`.

The trainer packs multiple short chunks into a single row via rmpad, with
`position_ids` resetting at each chunk boundary. The cache is keyed per-chunk
(each precomputed example is one unpadded chunk), so this module uses
`position_ids` to split the incoming packed row into chunks, hashes and looks
up each chunk, then stitches the hidden states back into a packed
[1, total_len, D] tensor that matches the student's layout.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import safe_open
from torch import nn

from ..utils.seqlen_pos_transform_utils import pos2culen


class _CachedTeacherOutput:
    """Minimal stand-in for `BaseModelOutputWithPast` — only the attrs that
    `modeling_qwen3.py` reads from teacher outputs.

    No __slots__: DeepSpeed ZeRO-3 forward hooks call vars(output) on nn.Module
    outputs; __slots__ removes __dict__ and would raise TypeError there.
    """

    def __init__(self, hidden_states: tuple, loss: Optional[torch.Tensor] = None):
        self.hidden_states = hidden_states
        self.loss = loss


def _hash_chunk(input_ids_chunk: torch.Tensor) -> str:
    return hashlib.sha256(input_ids_chunk.cpu().numpy().tobytes()).hexdigest()[:16]


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

    def _load_chunk(self, h: str, chunk: Optional[torch.Tensor] = None) -> dict[int, torch.Tensor]:
        p = self._shard_path(h)
        if not p.exists():
            preview = chunk[:16].tolist() if chunk is not None else None
            raise KeyError(
                f"Anchor cache miss for hash {h}. Path: {p}. "
                f"chunk_len={None if chunk is None else chunk.numel()} first16_ids={preview}"
            )
        out: dict[int, torch.Tensor] = {}
        with safe_open(str(p), framework="pt") as f:
            for li in self.cached_layers:
                out[li] = f.get_tensor(f"hidden_layer_{li}")
        return out

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None, **kwargs):
        """Return a mock CausalLMOutputWithPast-like object with sparse
        `hidden_states` populated only at indices in `self.cached_layers`.

        Splits the packed `input_ids` into chunks using `position_ids` (whose
        zeros mark chunk starts), hashes each chunk, looks up its cached
        hidden states, and concatenates them back to match the student layout.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids is None:
            raise ValueError(
                "CachedTeacher requires position_ids to split packed rmpad rows into chunks."
            )
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # rmpad collator yields bsz=1 with all chunks packed along seq dim.
        # Anything else would mean per-row padding, which this cache doesn't model.
        if bsz != 1:
            raise ValueError(
                f"CachedTeacher only supports bsz=1 packed rows, got bsz={bsz}. "
                "If you need padded multi-row batches, change the data pipeline or "
                "use the live teacher path."
            )

        # cu_seqlens = [0, len_0, len_0+len_1, ..., seq_len]
        cu_seqlens = pos2culen(position_ids).tolist()

        per_layer_slices: dict[int, list[torch.Tensor]] = {li: [] for li in self.cached_layers}
        for i in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            chunk = input_ids[0, s:e]
            h = _hash_chunk(chunk)
            chunk_layers = self._load_chunk(h, chunk=chunk)
            for li in self.cached_layers:
                t = chunk_layers[li]
                if t.size(0) != (e - s):
                    raise ValueError(
                        f"Cached chunk for hash {h} has length {t.size(0)} but "
                        f"packed row slice expects {e - s}."
                    )
                per_layer_slices[li].append(t)

        # Concatenate along seq dim to rebuild a packed [1, seq_len, D] tensor.
        stacked: dict[int, torch.Tensor] = {}
        for li in self.cached_layers:
            cat = torch.cat(per_layer_slices[li], dim=0).unsqueeze(0).to(device=device)
            if cat.size(1) != seq_len:
                raise ValueError(
                    f"Reconstructed cached layer {li} has length {cat.size(1)} "
                    f"but packed row is {seq_len}."
                )
            stacked[li] = cat

        hidden_states = tuple(
            stacked[li] if li in stacked else None for li in range(self._tuple_len)
        )
        return _CachedTeacherOutput(hidden_states=hidden_states, loss=None)
