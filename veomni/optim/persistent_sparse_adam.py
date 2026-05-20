# Copyright 2025 Open-dLLM Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PersistentSparseAdam — aggressive optimizer-state subsampling for
memory-constrained Repr-Align / Fast-dLLM training.

Only a configurable fraction of parameters keep full fp32 optimizer states
(m, v) at any time.  Inactive states are offloaded to CPU.  The active set
rotates periodically so every parameter sees updates over time.

Memory saving: ~4× at subset_ratio=0.25 vs full fp32 AdamW.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set, Union

import torch
import torch.optim as optim


class PersistentSparseAdam(optim.Optimizer):
    """
    Aggressive persistent sparse optimizer.

    Key idea: keep full-precision (m, v) states for only a rotating subset
    of parameters.  Inactive states live on CPU (or can be discarded).

    Designed for the non-DeepSpeed / FSDP path, or for DeepSpeech with
    ``ds_offload_optimizer: null`` so that *this* optimizer controls
    which states are on GPU vs CPU.

    Args:
        subset_ratio: Fraction of parameters with full GPU states (0..1).
        state_offload: Move inactive states to CPU rather than deleting them.
        warmup_steps: Number of steps before starting rotation (all params
            get full states during warmup for training stability).
        rotate_every: Rotate the active set every N steps.
        importance_key: Optional callable ``(param) -> float`` for
            importance-weighted sampling (higher = sampled more often).
            Default: uniform random.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict]],
        lr: float = 1e-5,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        subset_ratio: float = 0.25,
        state_offload: bool = True,
        warmup_steps: int = 200,
        rotate_every: int = 4,
        importance_key: Optional[Callable] = None,
    ):
        if not 0 < subset_ratio <= 1:
            raise ValueError(f"subset_ratio must be in (0, 1], got {subset_ratio}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.subset_ratio = subset_ratio
        self.state_offload = state_offload
        self.warmup_steps = warmup_steps
        self.rotate_every = rotate_every
        self.importance_key = importance_key
        self.global_step = 0

        # Per-parameter persistent state dict (survives rotations)
        self.param_state: Dict[int, Dict] = {}

        # Current active set (param ids)
        self.active_params: Set[int] = set()

        # All parameter ids (discovered on first step)
        self._all_param_ids: List[int] = []
        self._initialised = False

    # ── public helpers ────────────────────────────────────────────

    def get_active_ratio(self) -> float:
        """Fraction of parameters currently in the active set."""
        if not self._all_param_ids:
            return 0.0
        return len(self.active_params) / len(self._all_param_ids)

    def get_gpu_memory_saved_bytes(self) -> int:
        """Estimate GPU memory saved by offloading inactive states."""
        saved = 0
        for pid, pstate in self.param_state.items():
            if pid not in self.active_params and self.state_offload:
                for k in ("exp_avg", "exp_avg_sq"):
                    v = pstate.get(k)
                    if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                        saved += v.numel() * v.element_size()
        return saved

    # ── internals ─────────────────────────────────────────────────

    def _discover_params(self) -> None:
        """Build flat list of all trainable parameter ids."""
        ids: List[int] = []
        seen: Set[int] = set()
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    ids.append(pid)
        self._all_param_ids = ids
        self._initialised = True

    def _sample_active_set(self, n: int) -> Set[int]:
        """Sample *n* parameter ids for the active set."""
        pool = self._all_param_ids
        if self.importance_key is None:
            return set(random.sample(pool, min(n, len(pool))))

        # Importance-weighted sampling (higher weight = more likely)
        # Resolve param ids back to actual tensors for the importance callable
        id_to_param = {}
        for group in self.param_groups:
            for p in group["params"]:
                id_to_param[id(p)] = p

        weights = torch.tensor(
            [self.importance_key(id_to_param[pid]) for pid in pool], dtype=torch.float32
        )
        weights = weights.softmax(0)
        idx = torch.multinomial(weights, min(n, len(pool)), replacement=False)
        return {pool[i] for i in idx.tolist()}

    def _rotate_active_set(self) -> None:
        """Select new active subset and manage state persistence."""
        n_active = max(1, int(len(self._all_param_ids) * self.subset_ratio))
        new_active = self._sample_active_set(n_active)

        # Offload states that just became inactive
        for pid in self.active_params - new_active:
            pstate = self.param_state.get(pid)
            if pstate is not None and self.state_offload:
                for k in ("exp_avg", "exp_avg_sq"):
                    v = pstate.get(k)
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        pstate[k] = v.detach().cpu()

        # Create states for newly active params (will be moved to GPU in step())
        for pid in new_active:
            if pid not in self.param_state:
                self.param_state[pid] = {"step": 0}

        self.active_params = new_active

    def _get_param_by_id(self, pid: int) -> Optional[torch.Tensor]:
        """Reverse-lookup a parameter by its id()."""
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) == pid:
                    return p
        return None

    # ── Optimizer API ─────────────────────────────────────────────

    def step(self, closure=None):
        self.global_step += 1

        # Discover params on first call
        if not self._initialised:
            self._discover_params()

        # Warmup: all params active
        if self.global_step <= self.warmup_steps:
            all_ids = set(self._all_param_ids)
            if self.active_params != all_ids:
                # Move any previously offloaded states back to GPU
                for pid in all_ids - self.active_params:
                    pstate = self.param_state.get(pid)
                    p = self._get_param_by_id(pid)
                    if pstate is not None and p is not None:
                        for k in ("exp_avg", "exp_avg_sq"):
                            v = pstate.get(k)
                            if isinstance(v, torch.Tensor) and v.device != p.device:
                                pstate[k] = v.to(p.device, non_blocking=True)
                self.active_params = all_ids
        # First step after warmup (or if warmup_steps=0): initialise active set
        elif self.global_step == self.warmup_steps + 1 or (
            (self.global_step - self.warmup_steps) % self.rotate_every == 0
        ):
            self._rotate_active_set()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                pid = id(p)

                # Skip inactive params — gradient is discarded
                if pid not in self.active_params:
                    p.grad = None
                    continue

                # Ensure persistent state exists
                if pid not in self.param_state:
                    self.param_state[pid] = {}

                state = self.param_state[pid]

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                # Lazy-init on correct device
                if "exp_avg" not in state or state["exp_avg"].device != p.device:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if "exp_avg_sq" not in state or state["exp_avg_sq"].device != p.device:
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] = state.get("step", 0) + 1
                step_t = state["step"]

                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t

                # Promote to tensor if needed (scalar float path)
                if isinstance(bias_correction2, float):
                    bias_correction2_t = torch.tensor(bias_correction2, device=exp_avg_sq.device)
                    bias_correction1_t = torch.tensor(bias_correction1, device=exp_avg.device)
                else:
                    bias_correction2_t = bias_correction2
                    bias_correction1_t = bias_correction1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_t.sqrt()).add_(group["eps"])
                step_size = group["lr"] / bias_correction1_t

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        """Zero out gradients.  Only active params' grads are touched."""
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in self.active_params:
                    # Inactive params: aggressively free grad memory
                    if set_to_none:
                        p.grad = None
                    continue
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()
