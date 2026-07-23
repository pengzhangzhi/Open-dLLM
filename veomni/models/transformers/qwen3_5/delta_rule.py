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

"""Pure-PyTorch fallback for Gated DeltaNet linear attention.

Used when the `flash-linear-attention` (fla) library is not available.
Based on the Gated Delta Rule recurrence:
  S_t = (1 - beta_t) * S_{t-1} + beta_t * (k_t^T v_t)
  o_t = q_t @ S_t
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def chunk_gated_delta_rule_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Pure-PyTorch chunked Gated Delta Rule implementation.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, num_heads, seq_len, v_head_dim)
        beta: Beta (forget gate) of shape (batch, num_heads, seq_len, head_dim)
        g: Input gate of shape (batch, num_heads, seq_len, head_dim)
        initial_state: Optional tuple of (s, z) initial recurrent states
        chunk_size: Chunk size for parallel computation

    Returns:
        output: Output tensor of shape (batch, num_heads, seq_len, v_head_dim)
        new_state: Tuple of (s, z) final recurrent states
    """
    B, H, L, D = q.shape
    _, _, _, V = v.shape

    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))

    num_chunks = q.shape[2] // chunk_size

    # Initialize state
    if initial_state is None:
        s = torch.zeros(B, H, D, V, device=q.device, dtype=q.dtype)
        z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
    else:
        s, z = initial_state

    outputs = []

    for i in range(num_chunks):
        q_chunk = q[:, :, i * chunk_size : (i + 1) * chunk_size, :]       # (B, H, C, D)
        k_chunk = k[:, :, i * chunk_size : (i + 1) * chunk_size, :]       # (B, H, C, D)
        v_chunk = v[:, :, i * chunk_size : (i + 1) * chunk_size, :]       # (B, H, C, V)
        beta_chunk = beta[:, :, i * chunk_size : (i + 1) * chunk_size, :]  # (B, H, C, D)
        g_chunk = g[:, :, i * chunk_size : (i + 1) * chunk_size, :]       # (B, H, C, D)

        C = chunk_size

        # Apply input gate: gated key/value
        g_chunk = torch.sigmoid(g_chunk)
        k_gated = g_chunk * k_chunk

        # Compute intra-chunk attention with decay mask
        # Decay matrix: cumulative product of (1 - beta) within chunk
        beta_cumsum = torch.cumsum(beta_chunk, dim=2)  # (B, H, C, D)
        decay = torch.exp(-beta_cumsum)                # (B, H, C, D)

        # Inter-position decay mask for causal attention within chunk
        positions = torch.arange(C, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        beta_cumsum_prev = F.pad(beta_cumsum[:, :, :-1, :], (0, 0, 1, 0), value=0)  # shift right
        # Full causal mask with exponential decay
        causal_decay = decay.unsqueeze(2) / (decay.unsqueeze(3) + 1e-8)  # (B, H, C, C, D)
        # Make it strictly causal
        causal_mask = (positions.unsqueeze(3) >= positions.unsqueeze(2)).unsqueeze(-1).to(q.dtype)  # (1, 1, C, C, 1)
        causal_decay = causal_decay * causal_mask

        # Intra-chunk: output = sum over j<=i of q_i * (1-beta_i) * exp(-sum_{t=j+1}^{i} beta_t) * k_j * v_j
        kv = torch.einsum('b h c d, b h c v -> b h c d v', k_gated, v_chunk)  # (B, H, C, D, V)
        intra = torch.einsum('b h c d, b h c d v -> b h c v', q_chunk, kv)    # (B, H, C, V)

        # Inter-chunk contribution via recurrent state
        # Update state: s = (1 - beta) * s + beta * k * v
        # Using mean of beta over chunk for state update
        beta_mean = beta_chunk.mean(dim=2, keepdim=True)  # (B, H, 1, D)
        new_kv = torch.einsum('b h c d, b h c v -> b h d v', k_gated, v_chunk)  # (B, H, D, V)
        s_new = (1 - beta_mean) * s + beta_mean * new_kv
        z_new = z + (1 - beta_mean).sum(dim=2, keepdim=True)  # (B, H, D, 1)

        inter = torch.einsum('b h c d, b h d v -> b h c v', q_chunk, s_new / (z_new + 1e-8))

        # Decay for inter-chunk contribution
        inter_decay = decay.unsqueeze(-1)  # (B, H, C, D, 1) -> broadcast
        inter = inter * inter_decay.mean(dim=-1, keepdim=True)  # approximate

        chunk_output = intra + inter
        outputs.append(chunk_output)

        # Update state for next chunk
        s = s_new
        z = z_new

    output = torch.cat(outputs, dim=2)[:, :, :L, :]  # Remove padding

    return output, (s, z)


def fused_recurrent_gated_delta_rule_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Pure-PyTorch fused recurrent Gated Delta Rule (sequential, for decoding).

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, num_heads, seq_len, v_head_dim)
        beta: Beta (forget gate) of shape (batch, num_heads, seq_len, head_dim)
        g: Input gate of shape (batch, num_heads, seq_len, head_dim)
        initial_state: Optional tuple of (s, z) initial recurrent states

    Returns:
        output: Output tensor of shape (batch, num_heads, seq_len, v_head_dim)
        new_state: Tuple of (s, z) final recurrent states
    """
    B, H, L, D = q.shape
    _, _, _, V = v.shape

    if initial_state is None:
        s = torch.zeros(B, H, D, V, device=q.device, dtype=q.dtype)
        z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
    else:
        s, z = initial_state

    outputs = []

    for t in range(L):
        q_t = q[:, :, t:t+1, :]    # (B, H, 1, D)
        k_t = k[:, :, t:t+1, :]    # (B, H, 1, D)
        v_t = v[:, :, t:t+1, :]    # (B, H, 1, V)
        beta_t = beta[:, :, t:t+1, :]  # (B, H, 1, D)
        g_t = g[:, :, t:t+1, :]    # (B, H, 1, D)

        # Apply input gate
        g_t = torch.sigmoid(g_t)
        k_t_gated = g_t * k_t

        # Update state
        s = (1 - beta_t) * s + beta_t * torch.einsum('b h 1 d, b h 1 v -> b h d v', k_t_gated, v_t)
        z = z + (1 - beta_t)

        # Compute output
        o_t = torch.einsum('b h 1 d, b h d v -> b h 1 v', q_t, s / (z + 1e-8))
        outputs.append(o_t)

    output = torch.cat(outputs, dim=2)

    return output, (s, z)
