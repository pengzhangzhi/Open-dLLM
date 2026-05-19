"""Tests for the three ColaDLM causal diffusion variants + factory.

Each variant is tested for:
  1. Construction (no errors)
  2. Forward pass (returns dict with 'loss' key, finite values)
  3. Mask shape and properties (variant-specific)
"""

import pytest
import torch

from veomni.models.cola_ldm import (
    CardColaDLMHead,
    ColaDLMHead,
    FastBlockColaDLMHead,
    build_cola_head,
    make_block_causal_mask,
    make_complementary_mask,
    make_soft_tail_mask,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 64
NUM_GLOBAL = 4
NUM_LOCAL = 16
BLOCK_SIZE = 4
BATCH = 2
SEQ_LEN = 32  # hidden states from the LM


@pytest.fixture
def hidden_states():
    return torch.randn(BATCH, SEQ_LEN, DIM)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_build_block_causal(self):
        head = build_cola_head(dim=DIM, variant="block_causal", num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE)
        assert isinstance(head, ColaDLMHead)

    def test_build_card(self):
        head = build_cola_head(dim=DIM, variant="card", num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE, lambda_tail=0.6)
        assert isinstance(head, CardColaDLMHead)
        assert head.lambda_tail == 0.6

    def test_build_fast_block(self):
        head = build_cola_head(dim=DIM, variant="fast_block", num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE)
        assert isinstance(head, FastBlockColaDLMHead)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown cola_variant"):
            build_cola_head(dim=DIM, variant="nonexistent")


# ---------------------------------------------------------------------------
# Block-causal (base) variant
# ---------------------------------------------------------------------------

class TestBlockCausal:
    def test_forward_pass(self, hidden_states):
        head = ColaDLMHead(dim=DIM, num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE)
        head.eval()
        with torch.no_grad():
            out = head(hidden_states)
        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert out["loss"].dim() == 0  # scalar

    def test_mask_shape(self):
        total = NUM_GLOBAL + NUM_LOCAL
        mask = make_block_causal_mask(NUM_GLOBAL, NUM_LOCAL, BLOCK_SIZE)
        assert mask.shape == (total, total)
        assert mask.dtype == torch.bool

    def test_mask_causal_across_blocks(self):
        """Positions in later blocks cannot attend to earlier blocks."""
        mask = make_block_causal_mask(NUM_GLOBAL, NUM_LOCAL, BLOCK_SIZE)
        total = NUM_GLOBAL + NUM_LOCAL
        # Position in block 1 should NOT be seen by block 0
        # (block 0 starts at NUM_GLOBAL, block 1 at NUM_GLOBAL + BLOCK_SIZE)
        if NUM_LOCAL >= 2 * BLOCK_SIZE:
            later_pos = NUM_GLOBAL + BLOCK_SIZE
            earlier_pos = NUM_GLOBAL
            assert mask[later_pos, earlier_pos]  # causal: later can see earlier
            assert not mask[earlier_pos, later_pos]  # but NOT vice versa


# ---------------------------------------------------------------------------
# CARD soft-tail variant
# ---------------------------------------------------------------------------

class TestCard:
    def test_forward_pass(self, hidden_states):
        head = CardColaDLMHead(dim=DIM, num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE, lambda_tail=0.6)
        head.eval()
        with torch.no_grad():
            out = head(hidden_states)
        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert "tail_start" in out
        assert "tail_ratio" in out

    def test_soft_tail_mask(self):
        total = NUM_GLOBAL + NUM_LOCAL
        t = torch.tensor(0.5)
        mask, tail_start = make_soft_tail_mask(total, t, lambda_tail=0.6)
        assert mask.shape == (total, total)
        assert mask.dtype == torch.bool
        assert 0 <= tail_start < total

    def test_tail_shrinks_at_low_t(self):
        """At t=0.1, the tail should be small."""
        total = NUM_GLOBAL + NUM_LOCAL
        _, tail_start_low = make_soft_tail_mask(total, torch.tensor(0.1), lambda_tail=0.6)
        _, tail_start_high = make_soft_tail_mask(total, torch.tensor(0.9), lambda_tail=0.6)
        # Higher t → more of the sequence is noisy → tail_start is earlier
        assert tail_start_low >= tail_start_high

    def test_tail_start_within_bounds(self):
        total = NUM_GLOBAL + NUM_LOCAL
        for t_val in [0.01, 0.25, 0.5, 0.75, 0.99]:
            _, tail_start = make_soft_tail_mask(total, torch.tensor(t_val), lambda_tail=0.6)
            assert 0 <= tail_start < total, f"t={t_val}: tail_start={tail_start} out of bounds"


# ---------------------------------------------------------------------------
# FastBlock variant
# ---------------------------------------------------------------------------

class TestFastBlock:
    def test_forward_pass(self, hidden_states):
        head = FastBlockColaDLMHead(dim=DIM, num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE)
        head.eval()
        with torch.no_grad():
            out = head(hidden_states)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_complementary_mask_ratio(self):
        mask, ratio = make_complementary_mask(NUM_GLOBAL, NUM_LOCAL, BLOCK_SIZE)
        assert mask.shape == (NUM_GLOBAL + NUM_LOCAL, NUM_GLOBAL + NUM_LOCAL)
        assert 0 < ratio <= 1.0

    def test_complementary_mask_shape(self):
        mask, _ = make_complementary_mask(NUM_GLOBAL, NUM_LOCAL, BLOCK_SIZE)
        assert mask.dtype == torch.bool

    def test_no_complementary_option(self, hidden_states):
        """FastBlock with use_complementary=False should still work."""
        head = FastBlockColaDLMHead(dim=DIM, num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE, use_complementary=False)
        head.eval()
        with torch.no_grad():
            out = head(hidden_states)
        assert "loss" in out
        assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    @pytest.mark.parametrize("variant", ["block_causal", "card", "fast_block"])
    def test_gradients_flow(self, variant, hidden_states):
        """Verify gradients propagate through each variant."""
        head = build_cola_head(dim=DIM, variant=variant, num_global=NUM_GLOBAL, num_local=NUM_LOCAL, block_size=BLOCK_SIZE)
        head.train()
        out = head(hidden_states)
        out["loss"].backward()
        # Check at least some parameters have gradients
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert len(grads) > 0, f"No gradients for variant={variant}"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradients for variant={variant}"
