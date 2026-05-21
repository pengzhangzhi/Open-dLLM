"""Trajectory dataset for d3LLM-style pseudo-trajectory distillation.

Loads pre-computed unmasking trajectories (one per training sample) and
provides O(1) lookup by sample index during training.

The trajectory data is a JSONL file where each line contains:
    {"idx": <int>, "trajectory": [[tok_id, ...], ...], "nfe": <int>}

Each trajectory step is a full token-ID sequence.  Positions that are *still
masked* at that step have ``mask_token_id``.  This is the same format produced
by ``veomni.ops.trajectory_extractor.extract_and_save()``.
"""

import json
import os
from typing import Dict, List, Optional


class TrajectoryDataset:
    """In-memory trajectory store with sample-index lookup.

    Usage:
        td = TrajectoryDataset("/path/to/trajectories.jsonl")
        step = td.select_step(sample_idx=42, mask_ratio=0.5,
                              mask_token_id=151666, block_start=32, block_end=64)
        # step is a list of token IDs (the trajectory step closest to mask_ratio)
    """

    def __init__(self, path: str):
        self._data: Dict[int, dict] = {}
        self._load(path)

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._data[entry["idx"]] = entry

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, idx: int) -> bool:
        return idx in self._data

    def get(self, idx: int) -> Optional[List[List[int]]]:
        entry = self._data.get(idx)
        return entry["trajectory"] if entry is not None else None

    def nfe(self, idx: int) -> Optional[int]:
        entry = self._data.get(idx)
        return entry["nfe"] if entry is not None else None

    def select_step(
        self,
        sample_idx: int,
        mask_ratio: float,
        mask_token_id: int,
        block_start: int,
        block_end: int,
    ) -> Optional[List[int]]:
        """Return the trajectory step whose mask ratio in [block_start, block_end)
        is closest to ``mask_ratio``.

        Adapted from d3LLM's ``select_trajectory_by_ratio()``.
        """
        trajectory = self.get(sample_idx)
        if not trajectory:
            return None

        block_length = block_end - block_start
        num_unmasked = int((1 - mask_ratio) * block_length)
        target_idx = block_start + num_unmasked
        target_idx = min(target_idx, len(trajectory) - 1)
        return trajectory[target_idx]
