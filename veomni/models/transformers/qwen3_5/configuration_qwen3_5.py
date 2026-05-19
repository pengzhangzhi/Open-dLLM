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

"""Configuration for Qwen3.5/Qwen3.6 models.

Extends HF's Qwen3Config with hybrid attention (Gated DeltaNet + full attention),
attention output gating, partial rotary embeddings, and MTP support.
"""

from typing import List, Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3_5Config(Qwen3Config):
    """Configuration class for Qwen3.5/Qwen3.6 models.

    Adds support for:
    - Hybrid linear/full attention via Gated DeltaNet
    - Attention output gating
    - Partial rotary position embeddings
    - Multi-token prediction (MTP)
    """

    model_type = "qwen3_5"

    def __init__(
        self,
        partial_rotary_factor: float = 0.25,
        head_dim: int = 256,
        attn_output_gate: bool = True,
        full_attention_interval: int = 4,
        # Linear attention (Gated DeltaNet) config
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        linear_conv_kernel_dim: int = 4,
        # MTP config
        mtp_num_hidden_layers: int = 0,
        mtp_use_dedicated_embeddings: bool = False,
        # mRoPE
        mrope_interleaved: bool = True,
        mrope_section: Optional[List[int]] = None,
        # Override Qwen3 defaults
        rope_theta: float = 10000000.0,
        vocab_size: int = 248320,
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        # Set our attributes before super().__init__() so the layer_types property
        # getter can read them when Qwen3Config sets self.layer_types = [...] internally.
        self._partial_rotary_factor = partial_rotary_factor
        self._head_dim = head_dim
        self._attn_output_gate = attn_output_gate
        self._full_attention_interval = full_attention_interval

        # Linear attention config (set before super)
        self._linear_key_head_dim = linear_key_head_dim
        self._linear_value_head_dim = linear_value_head_dim
        self._linear_num_key_heads = linear_num_key_heads
        self._linear_num_value_heads = linear_num_value_heads
        self._linear_conv_kernel_dim = linear_conv_kernel_dim

        # Must override layer_types before super().__init__() — Qwen3Config
        # sets self.layer_types = [...] in __init__ which shadows our property.
        # Qwen3Config also reads num_hidden_layers before reaching this point,
        # so we must not pass layer_types in kwargs.
        _kwargs = {k: v for k, v in kwargs.items() if k != "layer_types"}
        super().__init__(
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            layer_types=None,  # force Qwen3Config to use its default, overridden by our property
            **_kwargs,
        )

        # MTP config (set before super)
        self._mtp_num_hidden_layers = mtp_num_hidden_layers
        self._mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        # mRoPE
        self._mrope_interleaved = mrope_interleaved
        if mrope_section is None:
            self._mrope_section = [11, 11, 10]
        else:
            self._mrope_section = mrope_section

        # Qwen3Config sets self.layer_types = [...] as instance attribute, shadowing
        # our @property. Delete it so the property getter takes effect.
        if hasattr(self, "layer_types") and not isinstance(self.__class__.layer_types, property):
            del self.layer_types

        # Qwen3Config may override head_dim via setter (computes hidden_size // num_heads).
        # Restore our value so full-attention projections use the correct dimension.
        object.__setattr__(self, "_head_dim", head_dim)

    @property
    def partial_rotary_factor(self) -> float:
        return self._partial_rotary_factor

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @head_dim.setter
    def head_dim(self, value):
        object.__setattr__(self, "_head_dim", value)

    @property
    def attn_output_gate(self) -> bool:
        return self._attn_output_gate

    @property
    def full_attention_interval(self) -> int:
        return self._full_attention_interval

    @property
    def linear_key_head_dim(self) -> int:
        return self._linear_key_head_dim

    @property
    def linear_value_head_dim(self) -> int:
        return self._linear_value_head_dim

    @property
    def linear_num_key_heads(self) -> int:
        return self._linear_num_key_heads

    @property
    def linear_num_value_heads(self) -> int:
        return self._linear_num_value_heads

    @property
    def linear_conv_kernel_dim(self) -> int:
        return self._linear_conv_kernel_dim

    @property
    def mtp_num_hidden_layers(self) -> int:
        return self._mtp_num_hidden_layers

    @property
    def mtp_use_dedicated_embeddings(self) -> bool:
        return self._mtp_use_dedicated_embeddings

    @property
    def mrope_interleaved(self) -> bool:
        return self._mrope_interleaved

    @property
    def mrope_section(self) -> List[int]:
        return self._mrope_section

    @property
    def layer_types(self) -> List[str]:
        """Compute per-layer type list for hybrid attention.

        Returns a list where every `full_attention_interval`-th layer uses
        full attention and the rest use linear attention.
        """
        # Qwen3Config may have stored a computed list via setter — use it if present.
        try:
            stored = object.__getattribute__(self, "_layer_types")
        except AttributeError:
            stored = None
        if stored is not None:
            return stored
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types

    @layer_types.setter
    def layer_types(self, value):
        # Qwen3Config sets self.layer_types = [...] in __init__. Store to
        # private attr so the getter is not shadowed by instance __dict__.
        object.__setattr__(self, "_layer_types", value)
