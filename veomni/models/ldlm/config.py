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
LDLM configuration dataclass for integration with Open-dLLM's Hydra-style config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LDLMArguments:
    """
    LDLM-specific training arguments.

    These are loaded from the training config YAML under the 'ldlm' key.
    """
    enabled: bool = False
    """Enable LDLM autoencoder training."""

    encoder_model_name: str = "Qwen/Qwen3.6-35B-A3B"
    """Frozen encoder model name or path."""

    seq_len: int = 128
    """Number of latent tokens (compression ratio)."""

    depth: int = 6
    """Number of Perceiver resampler layers."""

    decoder_noise_std: float = 3.0
    """Gaussian noise std added to latents before decoder."""

    encoder_hidden_layer: int = -3
    """Which encoder hidden layer to use as source (LDLM uses third-to-last)."""

    decoder_num_layers: int = 3
    """Number of transformer decoder layers for token prediction."""

    perceiver_heads: int = 8
    """Number of attention heads in Perceiver modules."""

    warmup_steps: int = 50000
    """Diffusion-to-encoder warmup steps (LDLM Section 5.2)."""

    # Loss weights
    recon_h_weight: float = 1.0
    """Weight for hidden state reconstruction loss L_h."""

    recon_token_weight: float = 1.0
    """Weight for token prediction loss L_w."""

    # Adaptive timestep sampler
    adaptive_sampler_num_bins: int = 100
    """Number of bins for adaptive timestep sampling."""

    adaptive_sampler_ema_decay: float = 0.999
    """EMA decay for adaptive timestep loss tracking."""

    adaptive_sampler_update_interval: int = 5000
    """Steps between adaptive probability recomputation."""

    # Diffusion head
    diffusion_head_dim: Optional[int] = None
    """Diffusion head hidden dim (defaults to encoder hidden_size)."""

    diffusion_head_depth: int = 12
    """Number of diffusion transformer layers."""

    # Logging
    log_interval: int = 50
    """Steps between wandb loss/metric logging."""

    gen_eval_interval: int = 2000
    """Steps between generation quality evaluation (PPL, entropy)."""

    log_latent_histograms: bool = True
    """Log latent z0 mean/std histograms to wandb every 500 steps."""

    log_samples: bool = True
    """Log generation samples to wandb during gen_eval."""
