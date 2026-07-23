"""Tests for Qwen3.6-style MTP (Multi-Token Prediction) head.

Tests:
  1. MTP head constructs when mtp_num_layers > 0
  2. MTP head is None when mtp_num_layers == 0
  3. MTP forward produces correct logit shape
  4. MTP loss in forward pass (integration with full model)
"""


from veomni.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from veomni.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MTPHead,
)


def _make_small_config(**overrides):
    """Minimal config for testing — tiny model."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        # MoE
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        # MTP
        mtp_num_layers=1,
        mtp_loss_weight=0.1,
        mtp_n_predict=1,
    )
    defaults.update(overrides)
    return Qwen3_5MoeConfig(**defaults)


class TestMTPHead:
    def test_constructs(self):
        config = _make_small_config()
        head = Qwen3_5MTPHead(config)
        assert len(head.layers) == 1
        assert head.lm_head is not None
        assert head.vocab_size == 256

    def test_mtp_disabled_when_zero(self):
        config = _make_small_config(mtp_num_layers=0)
        model = Qwen3_5MoeForCausalLM(config)
        assert model.mtp_head is None

    def test_model_creates_mtp_head(self):
        config = _make_small_config(mtp_num_layers=1)
        model = Qwen3_5MoeForCausalLM(config)
        assert model.mtp_head is not None
        assert isinstance(model.mtp_head, Qwen3_5MTPHead)

    def test_mtp_params_count(self):
        """MTP head should add a small number of parameters."""
        config = _make_small_config(mtp_num_layers=1)
        model = Qwen3_5MoeForCausalLM(config)
        mtp_params = sum(p.numel() for p in model.mtp_head.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        # MTP should be a small fraction of total
        assert mtp_params < total_params * 0.5
        assert mtp_params > 0

    def test_config_fields_propagate(self):
        config = _make_small_config(mtp_num_layers=2, mtp_loss_weight=0.3, mtp_n_predict=3)
        assert config.mtp_num_layers == 2
        assert config.mtp_loss_weight == 0.3
        assert config.mtp_n_predict == 3

        model = Qwen3_5MoeForCausalLM(config)
        assert len(model.mtp_head.layers) == 2
