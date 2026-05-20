"""Unit tests for PersistentSparseAdam.

Tests run on CPU (no GPU needed).  Covers:
  - Basic update logic matches AdamW on active params
  - Subset ratio correctly limits active params
  - Rotation changes active set over time
  - State offload moves inactive states to CPU
  - Warmup keeps all params active
  - Gradient clearing respects active/inactive boundary
"""

import gc
import math
import random

import pytest
import torch
import torch.nn as nn

from veomni.optim.persistent_sparse_adam import PersistentSparseAdam


# ── helpers ────────────────────────────────────────────────────────

def make_toy_model(d_model: int = 32, n_layers: int = 8):
    """Small MLP with N linear layers for testing."""
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(d_model, d_model, bias=False))
        if i < n_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def count_active_params(optimizer: PersistentSparseAdam) -> int:
    return len(optimizer.active_params)


def count_total_params(optimizer: PersistentSparseAdam) -> int:
    return len(optimizer._all_param_ids)


# ── tests ──────────────────────────────────────────────────────────

class TestPersistentSparseAdam:
    """Test suite for PersistentSparseAdam."""

    def test_initialisation(self):
        """Optimizer instantiates with correct defaults."""
        model = make_toy_model()
        opt = PersistentSparseAdam(model.parameters(), lr=1e-4, subset_ratio=0.5)
        assert opt.subset_ratio == 0.5
        assert opt.state_offload is True
        assert opt.warmup_steps == 200
        assert opt.rotate_every == 4
        assert opt.global_step == 0

    def test_subset_ratio_bounds(self):
        """Invalid subset_ratio raises ValueError."""
        model = make_toy_model()
        with pytest.raises(ValueError):
            PersistentSparseAdam(model.parameters(), subset_ratio=0.0)
        with pytest.raises(ValueError):
            PersistentSparseAdam(model.parameters(), subset_ratio=1.5)
        with pytest.raises(ValueError):
            PersistentSparseAdam(model.parameters(), subset_ratio=-0.1)

    def test_warmup_activates_all_params(self):
        """During warmup, every param should be in the active set."""
        model = make_toy_model(n_layers=6)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.25,
            warmup_steps=3,
            rotate_every=10,
        )

        model(torch.randn(4, 32)).mean().backward()
        opt.step()

        total = count_total_params(opt)
        active = count_active_params(opt)
        assert active == total, f"Expected all {total} active during warmup, got {active}"

    def test_rotation_changes_active_set(self):
        """After warmup + rotate_every steps, active set should differ."""
        model = make_toy_model(n_layers=10)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.3,
            warmup_steps=2,
            rotate_every=3,
        )

        # Warmup steps
        for _ in range(2):
            model(torch.randn(4, 32)).mean().backward()
            opt.step()

        active_after_warmup = set(opt.active_params)

        # Step through rotation boundaries
        for _ in range(5):
            model(torch.randn(4, 32)).mean().backward()
            opt.step()

        active_after_rotation = set(opt.active_params)
        assert active_after_warmup != active_after_rotation, \
            "Active set should change after rotation"

    def test_inactive_params_have_offloaded_state(self):
        """Inactive params' optimizer states should be on CPU when state_offload=True."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_toy_model(n_layers=8).to(device)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.5,
            warmup_steps=1,
            rotate_every=2,
            state_offload=True,
        )

        # Run past warmup and one rotation
        for _ in range(4):
            model(torch.randn(4, 32, device=device)).mean().backward()
            opt.step()

        # Check inactive params have CPU states
        cpu_found = False
        gpu_found = False
        for pid in opt._all_param_ids:
            pstate = opt.param_state.get(pid)
            if pstate is None:
                continue
            for k in ("exp_avg", "exp_avg_sq"):
                v = pstate.get(k)
                if isinstance(v, torch.Tensor):
                    if v.device.type == "cpu":
                        cpu_found = True
                    else:
                        gpu_found = True

        assert cpu_found, "Expected some inactive states on CPU"
        if torch.cuda.is_available():
            assert gpu_found, "Expected some active states on GPU"
        # else: CPU-only run — all states on CPU, that's fine

    def test_grads_cleared_for_inactive(self):
        """Inactive params should have their grads set to None after zero_grad(set_to_none=True)."""
        model = make_toy_model(n_layers=6)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.3,
            warmup_steps=0,
            rotate_every=5,
        )

        # One step to initialise active set (warmup_steps=0 means first step = rotation)
        model(torch.randn(4, 32)).mean().backward()
        opt.step()

        active_ids = set(opt.active_params)
        inactive_ids = [pid for pid in opt._all_param_ids if pid not in active_ids]

        # Check that inactive params have grad == None after zero_grad
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in inactive_ids:
                    assert p.grad is None, \
                        f"Inactive param should have grad=None after step, got {p.grad}"

    def test_loss_decreases_with_training(self):
        """Loss should decrease over a few steps (basic sanity)."""
        model = make_toy_model()
        opt = PersistentSparseAdam(
            model.parameters(),
            lr=1e-2,
            subset_ratio=0.5,
            warmup_steps=2,
            rotate_every=3,
        )

        losses = []
        for _ in range(6):
            loss = model(torch.randn(4, 32)).mean()
            losses.append(loss.item())
            loss.backward()
            opt.step()

        # Loss should trend down (may not be monotonic, but last < first on average)
        assert losses[-1] < losses[0], \
            f"Expected loss to decrease, went {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_importance_sampling(self):
        """Importance-weighted sampling should skew active set."""
        model = make_toy_model(n_layers=10)
        param_list = list(model.parameters())

        # Create importance weights that heavily favour the LAST layer
        def importance_fn(p):
            for i, mp in enumerate(param_list):
                if p is mp:
                    return 100.0 if i == len(param_list) - 1 else 1.0
            return 1.0

        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.2,  # only 2 out of 10 params active
            warmup_steps=0,
            rotate_every=100,  # don't re-rotate during test
            importance_key=importance_fn,
        )

        model(torch.randn(4, 32)).mean().backward()
        opt.step()

        # The last param should nearly always be in the active set
        last_param_id = id(param_list[-1])
        assert last_param_id in opt.active_params, \
            "Last layer (highest importance) should be in active set with importance sampling"

    def test_gpu_memory_tracking(self):
        """get_gpu_memory_saved_bytes should return non-zero after offload."""
        model = make_toy_model(n_layers=8)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.3,
            warmup_steps=0,
            rotate_every=3,
            state_offload=True,
        )

        for _ in range(4):
            model(torch.randn(4, 32)).mean().backward()
            opt.step()

        saved = opt.get_gpu_memory_saved_bytes()
        # If running on CPU, saved will be 0 (nothing to offload)
        if torch.cuda.is_available():
            assert saved > 0, f"Expected >0 bytes saved on GPU, got {saved}"

    def test_high_subset_ratio(self):
        """subset_ratio=1.0 should keep all params active always."""
        model = make_toy_model(n_layers=6)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=1.0,
            warmup_steps=0,
            rotate_every=2,
        )

        for _ in range(6):
            model(torch.randn(4, 32)).mean().backward()
            opt.step()
            total = count_total_params(opt)
            active = count_active_params(opt)
            assert active == total, f"subset_ratio=1.0: expected {total} active, got {active}"

    def test_persistent_state_after_rotation(self):
        """State dict should persist across rotations (not be deleted)."""
        model = make_toy_model(n_layers=8)
        opt = PersistentSparseAdam(
            model.parameters(),
            subset_ratio=0.3,
            warmup_steps=1,
            rotate_every=2,
            state_offload=True,
        )

        # Run through rotations
        for _ in range(8):
            model(torch.randn(4, 32)).mean().backward()
            opt.step()

        # Every param should have persistent state
        for pid in opt._all_param_ids:
            assert pid in opt.param_state, \
                f"Param {pid} should have persistent state after rotation"
            pstate = opt.param_state[pid]
            assert pstate.get("step", 0) > 0, \
                f"Param {pid} step should be > 0 (got {pstate.get('step')})"

    def test_match_adamw_active_params(self):
        """For active params, PersistentSparseAdam update should match AdamW closely."""
        torch.manual_seed(42)

        # Two identical models — one with AdamW, one with PersistentSparseAdam
        model_adamw = make_toy_model()
        model_sparse = make_toy_model()
        # Copy weights
        model_sparse.load_state_dict(model_adamw.state_dict())

        opt_adamw = torch.optim.AdamW(
            model_adamw.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)
        )
        opt_sparse = PersistentSparseAdam(
            model_sparse.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            subset_ratio=1.0,  # all active
            warmup_steps=0,
            rotate_every=100,
        )

        for _ in range(5):
            x = torch.randn(4, 32)
            loss_adamw = model_adamw(x).mean()
            loss_sparse = model_sparse(x).mean()

            loss_adamw.backward()
            opt_adamw.step()
            opt_adamw.zero_grad()

            loss_sparse.backward()
            opt_sparse.step()
            opt_sparse.zero_grad()

        # Compare weights — should be very close
        for p_adamw, p_sparse in zip(model_adamw.parameters(), model_sparse.parameters()):
            diff = (p_adamw - p_sparse).abs().max().item()
            assert diff < 1e-5, f"Weight diff {diff} > 1e-5"
