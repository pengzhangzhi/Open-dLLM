import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VFMNoiseAdapter(nn.Module):
    """Noise adapter q_φ(z|prompt) for Variational Flow Maps.

    Maps prompt token embeddings → (μ, log σ) for the initial noise distribution
    in embedding space. The adapter is a small Transformer encoder that processes
    the prompt, then pools to produce per-position Gaussian parameters.

    Architecture:
        - Small Transformer encoder (num_layers, d_model=hidden_size)
        - Pool over prompt positions → global (μ, log_σ) of shape [B, L, hidden_size]
        - L = total sequence length (prompt + generation slots)
        - For prompt positions: z = μ (deterministic copy)
        - For generation positions: z = μ + σ * ε, ε ~ N(0, I)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or 4 * hidden_size

        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.mu_head = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor, gen_length: int):
        """
        Args:
            prompt_embeds: [B, P, D] — embeddings of prompt tokens (from the LLM's embed_tokens)
            prompt_mask: [B, P] — 1 for real tokens, 0 for padding
            gen_length: number of generation positions to produce noise for

        Returns:
            mu: [B, P + G, D] — mean of noise distribution
            log_sigma: [B, P + G, D] — log std of noise distribution
        """
        B, P, D = prompt_embeds.shape
        G = gen_length

        # Expand prompt embeddings to cover generation slots
        # For generation positions, use learnable query or zero init
        gen_queries = torch.zeros(B, G, D, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        x = torch.cat([prompt_embeds, gen_queries], dim=1)

        # Add positional embeddings (clamp to max_seq_len to avoid OOB)
        positions = torch.arange(P + G, device=x.device).unsqueeze(0).clamp(max=self.pos_embed.num_embeddings - 1)
        x = x + self.pos_embed(positions)

        # Transformer encoding (bidirectional)
        full_mask = torch.cat([prompt_mask, torch.ones(B, G, device=prompt_mask.device)], dim=1)
        src_key_padding_mask = (full_mask == 0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.norm(x)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)

        # For prompt positions, set sigma=0 (deterministic)
        log_sigma[:, :P, :] = -10.0  # exp(-10) ≈ 0

        return mu, log_sigma

    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, diag(sigma^2)) with reparameterization."""
        sigma = log_sigma.exp().clamp(max=10.0)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def kl_loss(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """KL(q_φ(z|y) || p(z)) where p(z) = N(0, I). Averaged over batch."""
        # KL = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
        sigma_sq = (2 * log_sigma).exp().clamp(max=100.0)
        kl = 0.5 * (mu.pow(2) + sigma_sq - 2 * log_sigma - 1)
        return kl.sum(dim=-1).mean()


class VFMFlowMapWrapper(nn.Module):
    """Wraps a bidirectional LLM as a flow map f_θ: z → x.

    The flow map operates in embedding space:
        - Input: z ∈ R^{B x L x D} (noise / intermediate state in embedding space)
        - Output: logits ∈ R^{B x L x V} (vocab logits for token prediction)

    The LLM's embed_tokens + transformer + lm_head IS the flow map.
    We add a small projection to map arbitrary continuous embeddings back
    into the LLM's embedding space before feeding to the transformer.
    """

    def __init__(self, model: nn.Module, hidden_size: int, vocab_size: int):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Project from noise space → embedding space
        # This is needed because sampled noise z may not be in the
        # LLM's embedding distribution
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_proj[1].weight.data.zero_()
        self.input_proj[1].bias.data.copy_(
            model.get_input_embeddings().weight.mean(dim=0)
        )

    def forward(self, z: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.input_proj(z)
        outputs = self.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            is_causal=False,
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        return logits, last_hidden

    def embeddings_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def ids_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


class VariationalFlowMap(nn.Module):
    """Complete VFM model: noise adapter + flow map with joint training.

    Training objective (Eq. 19 from the paper):
        L = (1/2τ²) * L_MF + (1/2σ²) * L_obs + L_KL

    Where:
        L_MF: Mean flow / reconstruction loss — ||x - f_θ(z)||² in embedding space
        L_obs: Observation loss — ||prompt - A(f_θ(z))||² (prompt tokens match)
        L_KL: KL divergence — KL(q_φ(z|y) || p(z))
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        vocab_size: int,
        adapter_layers: int = 4,
        adapter_heads: int = 8,
        tau: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 0.5,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.flow_map = VFMFlowMapWrapper(model, hidden_size, vocab_size)
        self.adapter = VFMNoiseAdapter(
            hidden_size=hidden_size,
            num_layers=adapter_layers,
            num_heads=adapter_heads,
            max_seq_len=max_seq_len,
        )
        self.tau = tau
        self.sigma = sigma
        self.alpha = alpha
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_token_id: int):
        """
        Joint training forward pass.

        Args:
            input_ids: [B, L] — full sequence (prompt + target tokens)
            attention_mask: [B, L] — 1 for real tokens
            mask_token_id: ID of the mask token

        Returns:
            dict with loss components and metrics
        """
        B, L = input_ids.shape

        # Identify prompt (unmasked) vs generation (masked) positions
        gen_mask = (input_ids == mask_token_id)  # [B, L], True where we need to generate
        prompt_mask = (~gen_mask).float()  # [B, L], 1 for prompt positions

        # Count generation positions
        gen_length = gen_mask.sum(dim=1).max().item()
        if gen_length == 0:
            gen_length = 1

        # Get prompt embeddings
        all_embeds = self.flow_map.embeddings_from_ids(input_ids)  # [B, L, D]

        # Adapter: predict noise distribution from prompt
        # Use only prompt token embeddings as input
        prompt_lengths = prompt_mask.sum(dim=1).long()
        max_prompt = prompt_lengths.max().item()
        prompt_embeds_list = []
        prompt_mask_list = []
        for b in range(B):
            pl = prompt_lengths[b].item()
            prompt_embeds_list.append(all_embeds[b, :pl])
            prompt_mask_list.append(attention_mask[b, :pl])
        prompt_embeds_padded = torch.nn.utils.rnn.pad_sequence(
            prompt_embeds_list, batch_first=True
        )
        prompt_mask_padded = torch.stack(prompt_mask_list)

        mu, log_sigma = self.adapter(prompt_embeds_padded, prompt_mask_padded, gen_length)
        # mu, log_sigma: [B, P+G, D] — but we only need the generation positions
        # Take only the last gen_length positions for the noise
        gen_mu = mu[:, -gen_length:]  # [B, G, D]
        gen_log_sigma = log_sigma[:, -gen_length:]

        # Sample z for generation positions
        z_gen = self.adapter.sample(gen_mu, gen_log_sigma)  # [B, G, D]

        # Construct full z: prompt positions = embeddings, gen positions = sampled noise
        z_full = all_embeds.clone()
        for b in range(B):
            gen_pos = gen_mask[b].nonzero(as_tuple=True)[0]
            if len(gen_pos) > 0:
                gp_len = min(len(gen_pos), z_gen.shape[1])
                z_full[b, gen_pos[:gp_len]] = z_gen[b, :gp_len].to(z_full.dtype)

        # Mix: with probability alpha use adapter noise, else pure N(0,I)
        if self.alpha < 1.0 and self.training:
            mix_mask = (torch.rand(B, 1, 1, device=z_full.device) < self.alpha).float()
            pure_noise = torch.randn_like(z_full) * self.tau
            z_full = mix_mask * z_full + (1 - mix_mask) * pure_noise

        # Flow map: z → logits and hidden states
        logits, reconstructed = self.flow_map(z_full, attention_mask)  # [B, L, V], [B, L, D]

        # --- Losses ---

        # L_data: reconstruction in embedding space (only generation positions)
        if gen_mask.sum() > 0:
            data_loss = F.mse_loss(
                reconstructed[gen_mask],
                all_embeds.detach()[gen_mask],
            )
        else:
            data_loss = torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype)

        # L_obs: observation consistency — prompt tokens should be preserved
        # Use cross-entropy on prompt positions
        prompt_bool = prompt_mask.bool()
        if prompt_bool.sum() > 0:
            prompt_logits = logits[prompt_bool]
            prompt_targets = input_ids[prompt_bool].long()
            prompt_targets = prompt_targets.clamp(0, self.vocab_size - 1)
            obs_loss = F.cross_entropy(prompt_logits, prompt_targets)
        else:
            obs_loss = torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype)

        # L_KL: KL divergence of noise adapter
        # Only count KL for generation positions
        kl_loss = self.adapter.kl_loss(gen_mu, gen_log_sigma)

        # Total loss (Eq. 19)
        tau2 = self.tau ** 2
        sigma2 = self.sigma ** 2
        total_loss = data_loss / (2 * tau2) + obs_loss / (2 * sigma2) + kl_loss

        return {
            "loss": total_loss,
            "data_loss": data_loss.detach(),
            "obs_loss": obs_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "logits": logits,
            "reconstructed": reconstructed,
            "z_gen": z_gen,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        mask_token_id: int,
        max_new_tokens: int = 64,
        num_steps: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens using VFM: prompt → adapter → flow map → tokens.

        Args:
            prompt_ids: [B, P] — prompt token IDs
            mask_token_id: mask token ID
            max_new_tokens: number of tokens to generate
            num_steps: 1 for single-step, 2-4 for multi-step refinement
            temperature: sampling temperature

        Returns:
            generated_ids: [B, P + G] — full sequence with generated tokens
        """
        B, P = prompt_ids.shape
        G = max_new_tokens
        device = prompt_ids.device

        # Build full sequence with mask tokens
        mask_ids = torch.full((B, G), mask_token_id, device=device, dtype=prompt_ids.dtype)
        x_ids = torch.cat([prompt_ids, mask_ids], dim=1)  # [B, P+G]
        attention_mask = torch.ones(B, P + G, device=device)

        # Get prompt embeddings
        prompt_embeds = self.flow_map.embeddings_from_ids(prompt_ids)  # [B, P, D]
        prompt_attn_mask = torch.ones(B, P, device=device)

        # Step 1: Adapter produces initial noise
        mu, log_sigma = self.adapter(prompt_embeds, prompt_attn_mask, G)
        z = self.adapter.sample(mu[:, -G:], log_sigma[:, -G:])  # [B, G, D]

        # Build full z: prompt embeddings + generated noise
        z_full = torch.cat([prompt_embeds, z], dim=1)  # [B, P+G, D]

        # Steps 2+: Flow map iteration
        for step in range(num_steps):
            logits, reconstructed = self.flow_map(z_full, attention_mask)
            # For multi-step: use reconstructed embeddings as new z for gen positions
            if step < num_steps - 1:
                z_full[:, P:, :] = reconstructed[:, P:, :].detach()

        # Decode: logits → token IDs
        # Apply logit shift (same as MDM generation)
        shifted_logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        gen_logits = shifted_logits[:, P:, :]  # [B, G, V]

        if temperature > 0:
            gen_logits = gen_logits / temperature
            probs = F.softmax(gen_logits, dim=-1)
            gen_ids = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(B, G)
        else:
            gen_ids = gen_logits.argmax(dim=-1)

        # Combine prompt + generated
        x_ids[:, P:] = gen_ids
        return x_ids
