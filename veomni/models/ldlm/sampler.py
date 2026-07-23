# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adaptive timestep sampler from LDLM (arXiv:2605.07933, Section 5.3).

Dynamically adjusts the sampling distribution so that the expected
denoising loss grows linearly with t in [0, 1].
"""

from typing import Optional

import torch


class AdaptiveTimestepSampler:
    """
    Adaptive timestep sampler for latent diffusion training.

    Maintains an EMA of the denoising loss per timestep bin and
    adjusts sampling probabilities proportional to dL/du so that
    the expected loss is uniform across timesteps.
    """

    def __init__(
        self,
        num_bins: int = 100,
        ema_decay: float = 0.999,
        min_prob: float = 1e-6,
        update_interval: int = 5000,
        device: Optional[torch.device] = None,
    ):
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        self.min_prob = min_prob
        self.update_interval = update_interval
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Bin edges: u_i = i / N
        self.bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=self.device)

        # EMA of loss at each bin edge
        self.loss_ema = torch.zeros(num_bins + 1, device=self.device)

        # Probability per bin (uniform init)
        self.bin_probs = torch.ones(num_bins, device=self.device) / num_bins

        self.update_counter = 0

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, losses: torch.Tensor):
        """
        Update EMA loss estimates.

        Args:
            timesteps: (batch,) t in [0, 1]
            losses: (batch,) denoising loss L(t)
        """
        # Find which bin each timestep falls into
        bin_indices = torch.bucketize(timesteps.to(self.device), self.bin_edges[1:], right=True).clamp(0, self.num_bins - 1)

        # Update EMA per bin
        losses_flat = losses.view(-1).to(self.device)
        for i in range(self.num_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_loss = losses_flat[mask].mean()
            self.loss_ema[i] = (
                self.ema_decay * self.loss_ema[i] + (1.0 - self.ema_decay) * bin_loss
            )

        # Also update the last bin edge (t=1.0)
        self.loss_ema[-1] = (
            self.ema_decay * self.loss_ema[-1] + (1.0 - self.ema_decay) * losses.max()
        )

        self.update_counter += 1

        # Recompute probabilities periodically
        if self.update_counter % self.update_interval == 0 or self.update_counter == 1:
            self._recompute_probs()

    def _recompute_probs(self):
        """Recompute bin probabilities proportional to Delta_L / Delta_u."""
        deltas = self.loss_ema[1:] - self.loss_ema[:-1]   # dL/du approximation
        deltas = torch.clamp(deltas, min=0.0)             # monotonic

        total = deltas.sum()
        if total > 0:
            probs = deltas / total
        else:
            probs = torch.ones_like(deltas) / self.num_bins

        # Minimum probability floor
        probs = torch.clamp(probs, min=self.min_prob)
        probs = probs / probs.sum()
        self.bin_probs = probs

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps from the adaptive distribution.

        Returns:
            t: (batch_size,) tensor in [0, 1]
        """
        # Sample bin index according to current probabilities
        bin_idx = torch.multinomial(self.bin_probs, batch_size, replacement=True)

        # Sample uniformly inside the chosen bin
        u_low = self.bin_edges[bin_idx]
        u_high = self.bin_edges[bin_idx + 1]
        u = u_low + torch.rand(batch_size, device=self.device) * (u_high - u_low)

        return u.clamp(0.0, 1.0)
