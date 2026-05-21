"""
DifferentialReplayLoss — efficient multi-step diffusion distillation.

Teacher runs a long trajectory (no_grad + cached velocities).
Student runs a short trajectory with a differentiable schedule.
Only the student replay path contributes gradients.

Compatible with ColaDLM, Fast-dLLM, BézierFlow, SPARTA distillation.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialReplayLoss(nn.Module):
    """
    Distill a long diffusion teacher (T steps) into a short student (S steps).

    Teacher trajectory: computed once under no_grad; velocities are cached.
    Student trajectory: differentiable replay — gradients flow through the
      student's denoising function and its learned sigma schedule.

    Args:
        teacher_steps: Number of Euler steps for the teacher trajectory.
        student_steps: Number of Euler steps for the student trajectory.
        loss_fn: Element-wise loss between final student and teacher states.
            Defaults to MSE.
        reduction: "mean" or "sum" applied to the scalar loss.
    """

    def __init__(
        self,
        teacher_steps: int = 30,
        student_steps: int = 8,
        loss_fn: Callable = F.mse_loss,
        reduction: str = "mean",
    ):
        super().__init__()
        self.teacher_steps = teacher_steps
        self.student_steps = student_steps
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(
        self,
        z_init: torch.Tensor,
        teacher_denoise_fn: Callable,
        student_denoise_fn: Callable,
        teacher_sigmas: torch.Tensor,
        student_sigmas: torch.Tensor,
        **denoise_kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            z_init: Initial noisy latent [B, ..., D].
            teacher_denoise_fn: Callable(latent, sigma, **kw) → velocity.
            student_denoise_fn: Same signature as teacher.
            teacher_sigmas: [T+1] or [B, T+1] — teacher noise schedule.
            student_sigmas: [S+1] or [B, S+1] — student schedule (differentiable).
            **denoise_kwargs: Forwarded to both denoise functions
                (prompt embeds, position ids, etc.).

        Returns:
            loss: Scalar distillation loss.
            extras: Dict with z_teacher_final, z_student_final, cached_velocities.
        """
        B = z_init.shape[0]

        if teacher_sigmas.dim() == 1:
            teacher_sigmas = teacher_sigmas.unsqueeze(0).expand(B, -1)
        if student_sigmas.dim() == 1:
            student_sigmas = student_sigmas.unsqueeze(0).expand(B, -1)

        # ── Teacher (no grad, cache velocities) ───────────────────────────
        with torch.no_grad():
            z_t = z_init.clone()
            cached_velocities = []
            for step in range(self.teacher_steps):
                sigma = teacher_sigmas[:, step]
                v = teacher_denoise_fn(z_t, sigma=sigma, **denoise_kwargs)
                cached_velocities.append(v.detach())
                if step < self.teacher_steps - 1:
                    dt = teacher_sigmas[:, step + 1] - sigma
                    z_t = z_t + v * dt.view(B, *([1] * (z_t.dim() - 1)))
            z_teacher = z_t.detach()

        # ── Student (differentiable replay) ───────────────────────────────
        z_s = z_init.clone().to(torch.float32)
        for step in range(self.student_steps):
            sigma = student_sigmas[:, step]
            v = student_denoise_fn(z_s, sigma=sigma, **denoise_kwargs)
            if step < self.student_steps - 1:
                dt = student_sigmas[:, step + 1] - sigma
                z_s = z_s + v.float() * dt.view(B, *([1] * (z_s.dim() - 1)))
        z_student = z_s.to(z_init.dtype)

        # ── Loss ──────────────────────────────────────────────────────────
        loss = self.loss_fn(z_student, z_teacher, reduction="none")
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        extras = {
            "z_teacher_final": z_teacher,
            "z_student_final": z_student,
            "cached_velocities": cached_velocities,
            "teacher_sigmas": teacher_sigmas,
            "student_sigmas": student_sigmas,
        }
        return loss, extras
