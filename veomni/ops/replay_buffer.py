import random
from collections import deque
from typing import Any, Dict, Optional

import torch


class ReprAlignReplayBuffer:
    """Ring buffer of past input batches for representation alignment replay.

    Stores the full micro_batch dict from past training steps. On replay,
    re-runs the student model on old data and computes a cosine alignment loss
    against the cached teacher hidden states.

    The anchor cache makes teacher hidden states free on replay: CachedTeacher
    reads precomputed .safetensors via SHA-256 hash lookup, O(1) per chunk,
    no model forward needed.

    Reference: VFM Ripple's NoiseReplayBuffer (ltx-trainer), adapted for
    frozen-teacher alignment instead of drifting adapter distributions.
    """

    def __init__(self, capacity: int = 1024):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, batch: Dict[str, Any]) -> None:
        cpu_batch = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        self._buf.append(cpu_batch)

    def sample(self, device: torch.device) -> Optional[Dict[str, Any]]:
        if not self._buf:
            return None
        batch = random.choice(self._buf)
        return {
            k: v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __len__(self) -> int:
        return len(self._buf)
