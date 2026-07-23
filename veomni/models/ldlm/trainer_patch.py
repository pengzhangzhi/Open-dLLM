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
LDLM training loop integration with full wandb logging.

Provides LDLMTrainer which wraps the autoencoder, diffusion head, and
adaptive sampler into a single train_step() with structured logging
matching LDLM paper (arXiv:2605.07933).

Key metrics logged to wandb:
  - train/total_loss, train/diffusion_loss, train/recon_h_loss, train/recon_w_loss
  - train/latent_norm, train/decoder_noise_std
  - train/sampler_loss_min/max/range (adaptive sampler health)
  - train/gen_ppl, train/entropy (generation quality every N steps)
  - Latent histograms every 500 steps
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class LDLMTrainer:
    """
    Wraps an LDLM autoencoder + diffusion head for training.

    Manages forward pass, adaptive timestep sampling, loss computation,
    warmup schedule, and structured wandb logging.
    """

    def __init__(
        self,
        autoencoder,
        diffusion_head,
        sampler,
        config: dict,
        vocab_size: int,
        tokenizer=None,
    ):
        self.autoencoder = autoencoder
        self.diffusion_head = diffusion_head
        self.sampler = sampler
        self.config = config
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.dim = autoencoder.dim
        self.seq_len = autoencoder.seq_len

        ldlm_cfg = config.get("ldlm", {})

        # Logging config
        self.log_interval = ldlm_cfg.get("log_interval", 50)
        self.gen_eval_interval = ldlm_cfg.get("gen_eval_interval", 2000)
        self.log_latent_histograms = ldlm_cfg.get("log_latent_histograms", True)
        self.log_samples = ldlm_cfg.get("log_samples", True)

        # Warmup state
        self.warmup_steps = ldlm_cfg.get("warmup_steps", 50000)
        self.global_step = 0

        # Loss weights
        self.recon_h_weight = ldlm_cfg.get("recon_h_weight", 1.0)
        self.recon_token_weight = ldlm_cfg.get("recon_token_weight", 1.0)

        # Generation eval cache
        self._eval_texts: List[str] = []

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_wandb: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step with logging.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: optional (B, T) mask
            use_wandb: if True, log metrics to wandb

        Returns:
            dict of scalar losses + log dict for wandb
        """
        B = input_ids.shape[0]
        step = self.global_step
        log_dict = {}

        # --- 1. Autoencoder forward ---
        ae_out = self.autoencoder(input_ids, attention_mask, training=True)
        z0 = ae_out["z0"]
        h = ae_out["h"]
        h_hat = ae_out["h_hat"]
        logits = ae_out["logits"]
        noise = ae_out.get("noise", None)

        # --- 2. Reconstruction losses ---
        L_h = F.mse_loss(h_hat, h)
        L_w = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=-100,
        )

        # --- 3. Adaptive timestep sampling ---
        t = self.sampler.sample(B)

        # --- 4. Diffusion on latents ---
        # Schedule: alpha_bar(t) = 1 - t^2 (cosine-like)
        alpha_bar = 1.0 - t[:, None, None] ** 2
        sigma = (1.0 - alpha_bar).sqrt()

        diff_noise = torch.randn_like(z0)
        z_t = alpha_bar.sqrt() * z0 + sigma * diff_noise

        pred = self.diffusion_head(z_t, t)
        L_diff = F.mse_loss(pred, z0)

        # --- 5. Warmup ---
        warmup_progress = min(step / max(self.warmup_steps, 1), 1.0)
        diff_weight = warmup_progress

        # --- 6. Total loss ---
        loss = L_diff * diff_weight + L_h * self.recon_h_weight + L_w * self.recon_token_weight

        # --- 7. Update sampler ---
        self.sampler.update(t, L_diff.detach())

        self.global_step += 1

        # --- 8. Wandb logging (every log_interval) ---
        if use_wandb and step % self.log_interval == 0:
            import wandb

            # Core losses
            log_dict = {
                "train/total_loss": loss.item(),
                "train/diffusion_loss": L_diff.item(),
                "train/recon_h_loss": L_h.item(),
                "train/recon_w_loss": L_w.item(),
                "train/diff_weight": diff_weight,
                "train/latent_norm": z0.norm(dim=-1).mean().item(),
                "train/latent_std": z0.std().item(),
                "train/decoder_noise_std": noise.std().item() if noise is not None else 0.0,
                "train/learning_rate": self.config.get("train", {}).get("lr", 0.0),
                # Adaptive sampler insights
                "train/sampler_loss_min": self.sampler.loss_ema.min().item(),
                "train/sampler_loss_max": self.sampler.loss_ema.max().item(),
                "train/sampler_loss_range": (
                    self.sampler.loss_ema.max() - self.sampler.loss_ema.min()
                ).item(),
                "train/timestep_mean": t.mean().item(),
                "train/timestep_std": t.std().item(),
            }

            # Log histograms periodically (10x log_interval)
            hist_interval = self.log_interval * 10
            if self.log_latent_histograms and step % hist_interval == 0:
                log_dict["latent_stats/z0_mean"] = wandb.Histogram(
                    z0.mean(dim=[0, 1]).cpu()
                )
                log_dict["latent_stats/z0_std"] = wandb.Histogram(
                    z0.std(dim=[0, 1]).cpu()
                )
                log_dict["latent_stats/timesteps"] = wandb.Histogram(t.cpu())
                log_dict["latent_stats/diff_loss_per_sample"] = wandb.Histogram(
                    L_diff.detach().cpu()
                )

            # Generation evaluation
            if self.log_samples and step % self.gen_eval_interval == 0:
                gen_metrics = self._eval_generation()
                log_dict.update(gen_metrics)

            wandb.log(log_dict, step=step)

        return {
            "loss": loss,
            "loss_diff": L_diff,
            "loss_recon_h": L_h,
            "loss_recon_token": L_w,
            "diff_weight": torch.tensor(diff_weight),
            "t_mean": t.mean(),
            "log_dict": log_dict,
        }

    @torch.no_grad()
    def _eval_generation(self) -> Dict:
        """
        Evaluate generation quality: perplexity and entropy.

        Uses a small batch of latents to generate tokens and compute
        quality metrics matching LDLM paper (Figure 1, Figure 3).
        """
        metrics = {}

        # Sample a latent, decode it
        z_sample = torch.randn(1, self.seq_len, self.dim).cuda()
        dec_out = self.autoencoder.decode(z_sample, training=False)
        logits = dec_out["logits"]  # (1, T, V)

        # Entropy of token distribution
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        metrics["train/entropy"] = entropy

        # Perplexity from softmax confidence
        top_probs, _ = probs.max(dim=-1)
        gen_ppl = (-top_probs.log()).mean().item()
        metrics["train/gen_ppl"] = gen_ppl

        return metrics

    def get_sampler_stats(self) -> Dict:
        """Return sampler statistics for logging."""
        return {
            "sampler_loss_min": self.sampler.loss_ema.min().item(),
            "sampler_loss_max": self.sampler.loss_ema.max().item(),
            "sampler_update_count": self.sampler.update_counter,
        }
