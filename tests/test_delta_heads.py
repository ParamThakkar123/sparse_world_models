"""Tests for the probabilistic delta heads and the sequence dataset (W1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models import SparseResidualHead, masked_delta_nll_loss, sparse_residual_loss
from models.delta_heads import (
    MAX_LOG_SCALE,
    MIN_LOG_SCALE,
    GaussianDeltaHead,
    MixtureDeltaHead,
    build_delta_head,
)
from models.sequence_datasets import TransitionSequenceDataset, contiguous_run_bounds


def test_gaussian_log_prob_matches_torch_distribution() -> None:
    torch.manual_seed(0)
    head = GaussianDeltaHead(object_feature_dim=8, hidden_dim=16)
    features = torch.randn(4, 3, 8)
    target = torch.randn(4, 3, 3) * 0.05

    distribution = head(features)
    reference = torch.distributions.Normal(distribution.mean, distribution.scale)
    torch.testing.assert_close(
        distribution.log_prob(target), reference.log_prob(target).sum(dim=-1)
    )


def test_mixture_log_prob_matches_manual_logsumexp() -> None:
    torch.manual_seed(0)
    head = MixtureDeltaHead(object_feature_dim=8, hidden_dim=16, num_components=4)
    features = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 3) * 0.05

    distribution = head(features)
    log_weights = torch.log_softmax(distribution.logits, dim=-1)
    component = torch.distributions.Normal(
        distribution.means, torch.exp(distribution.log_scales)
    ).log_prob(target.unsqueeze(-2)).sum(dim=-1)
    torch.testing.assert_close(
        distribution.log_prob(target), torch.logsumexp(log_weights + component, dim=-1)
    )


def test_component_mode_selects_highest_weight_component_not_the_average() -> None:
    """The point estimate must commit to one component.

    Using the mixture mean would reintroduce exactly the averaging that pins a squared-error
    head at the no-op floor, so this distinction is the whole reason the MDN exists.
    """
    logits = torch.tensor([[[2.0, -2.0]]])
    means = torch.tensor([[[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]])
    log_scales = torch.zeros_like(means)
    from models.delta_heads import MixtureDelta

    distribution = MixtureDelta(logits=logits, means=means, log_scales=log_scales)
    torch.testing.assert_close(distribution.component_mode, torch.tensor([[[1.0, 0.0, 0.0]]]))
    # The two components nearly cancel, so the mean sits between them -- near zero.
    assert distribution.mixture_mean.abs().max() < 1.0
    assert not torch.allclose(distribution.component_mode, distribution.mixture_mean)


def test_effective_components_flags_a_collapsed_mixture() -> None:
    from models.delta_heads import MixtureDelta

    means = torch.zeros(1, 1, 4, 3)
    log_scales = torch.zeros_like(means)
    uniform = MixtureDelta(torch.zeros(1, 1, 4), means, log_scales)
    collapsed = MixtureDelta(torch.tensor([[[50.0, 0.0, 0.0, 0.0]]]), means, log_scales)
    assert uniform.effective_components().item() == pytest.approx(4.0, abs=1e-4)
    assert collapsed.effective_components().item() == pytest.approx(1.0, abs=1e-4)


def test_log_scales_are_clamped() -> None:
    torch.manual_seed(0)
    head = GaussianDeltaHead(object_feature_dim=4, hidden_dim=8)
    # Drive the scale output to extremes; the clamp must still hold.
    with torch.no_grad():
        head.trunk.final_layer.bias[3:].fill_(1e4)
    distribution = head(torch.randn(2, 2, 4))
    assert distribution.log_scale.max().item() <= MAX_LOG_SCALE + 1e-6
    assert distribution.log_scale.min().item() >= MIN_LOG_SCALE - 1e-6


def test_mse_head_is_unchanged_so_old_checkpoints_still_load() -> None:
    """'mse' must keep producing a bare tensor from an unchanged submodule."""
    model = SparseResidualHead(object_feature_dim=12)
    assert model.delta_head_type == "mse"
    assert not model.is_probabilistic
    out = model(torch.randn(2, 3, 12), estimator="st")
    assert out.delta_dist is None
    # The parameter names are what a torch state_dict keys on.
    assert any(name.startswith("delta_head.mlp.") for name, _ in model.named_parameters())


@pytest.mark.parametrize("head_type", ["gaussian", "mdn"])
def test_probabilistic_heads_expose_a_distribution_and_train(head_type: str) -> None:
    torch.manual_seed(0)
    model = SparseResidualHead(object_feature_dim=12, delta_head_type=head_type)
    features = torch.randn(6, 3, 12)
    target_delta = torch.randn(6, 3, 3) * 0.05
    mask = (torch.rand(6, 3) > 0.5).float()

    out = model(features, estimator="st")
    assert out.delta_dist is not None
    assert out.delta.shape == target_delta.shape

    losses = sparse_residual_loss(
        out.gate.logits, out.gate.probs, out.delta, mask, target_delta,
        delta_dist=out.delta_dist,
    )
    assert losses.delta_nll is not None
    losses.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_nll_loss_ignores_unchanged_objects() -> None:
    """Only objects labelled changed may contribute, mirroring masked_delta_l2_loss."""
    torch.manual_seed(0)
    head = GaussianDeltaHead(object_feature_dim=6, hidden_dim=8)
    features = torch.randn(3, 4, 6)
    distribution = head(features)
    target = torch.randn(3, 4, 3)
    mask = torch.zeros(3, 4)
    mask[0, 0] = 1.0

    selective = masked_delta_nll_loss(distribution, target, mask)
    manual = -distribution.log_prob(target)[0, 0]
    torch.testing.assert_close(selective, manual)


def test_nll_loss_is_zero_when_nothing_changed() -> None:
    head = GaussianDeltaHead(object_feature_dim=6, hidden_dim=8)
    distribution = head(torch.randn(2, 2, 6))
    loss = masked_delta_nll_loss(distribution, torch.randn(2, 2, 3), torch.zeros(2, 2))
    assert loss.item() == 0.0


def test_build_delta_head_rejects_unknown_types() -> None:
    with pytest.raises(ValueError, match="Unsupported delta_head_type"):
        build_delta_head("laplace", object_feature_dim=4)


def test_mdn_initial_scale_is_near_the_delta_magnitude() -> None:
    """A unit-variance init would put the initial NLL far from the data's scale."""
    head = MixtureDeltaHead(object_feature_dim=6, hidden_dim=8, num_components=3)
    distribution = head(torch.zeros(1, 1, 6))
    # Bias-dominated at zero input, so the scale should sit near the configured default.
    assert distribution.log_scales.mean().item() == pytest.approx(math.log(0.02), abs=0.5)


def test_contiguous_runs_split_on_a_break() -> None:
    state = np.array([[0.0], [1.0], [2.0], [9.0], [10.0]], dtype=np.float32)
    next_state = np.array([[1.0], [2.0], [3.0], [10.0], [11.0]], dtype=np.float32)
    # Rows 0-2 chain (next[i] == state[i+1]); row 2 -> row 3 breaks; rows 3-4 chain.
    assert contiguous_run_bounds(state, next_state) == [(0, 3), (3, 5)]


def test_contiguity_uses_values_not_done_flags(tmp_path) -> None:
    """A filtered dataset keeps stale done flags, so windows must not trust them."""
    path = tmp_path / "gapped.npz"
    # Two disjoint 2-step runs; done is True only on the very last row, as a filter leaves it.
    state = np.array([[0.0], [1.0], [50.0], [51.0]], dtype=np.float32)
    next_state = np.array([[1.0], [2.0], [51.0], [52.0]], dtype=np.float32)
    state = np.repeat(state, 31, axis=1)
    next_state = np.repeat(next_state, 31, axis=1)
    np.savez(
        path,
        s_t=state, a_t=np.zeros((4, 2), np.float32), s_t1=next_state,
        done=np.array([False, False, False, True]),
        object_change_mask=np.zeros((4, 3), np.float32),
        object_delta=np.zeros((4, 3, 3), np.float32),
    )
    dataset = TransitionSequenceDataset(path, horizon=2)
    # A done-based reader would see one 4-step episode and offer 3 windows; only 2 are real.
    assert len(dataset) == 2
    assert dataset.window_statistics()["num_runs"] == 2


def test_sequence_dataset_rejects_a_horizon_longer_than_any_run(tmp_path) -> None:
    path = tmp_path / "short.npz"
    state = np.repeat(np.array([[0.0], [9.0]], dtype=np.float32), 31, axis=1)
    next_state = np.repeat(np.array([[5.0], [12.0]], dtype=np.float32), 31, axis=1)
    np.savez(
        path,
        s_t=state, a_t=np.zeros((2, 2), np.float32), s_t1=next_state,
        done=np.array([False, True]),
        object_change_mask=np.zeros((2, 3), np.float32),
        object_delta=np.zeros((2, 3, 3), np.float32),
    )
    with pytest.raises(ValueError, match="No contiguous windows"):
        TransitionSequenceDataset(path, horizon=3)


def test_sequence_items_carry_a_leading_horizon_axis(tmp_path) -> None:
    path = tmp_path / "runs.npz"
    base = np.arange(5, dtype=np.float32).reshape(5, 1)
    state = np.repeat(base, 31, axis=1)
    next_state = np.repeat(base + 1.0, 31, axis=1)
    np.savez(
        path,
        s_t=state, a_t=np.zeros((5, 2), np.float32), s_t1=next_state,
        done=np.array([False, False, False, False, True]),
        object_change_mask=np.zeros((5, 3), np.float32),
        object_delta=np.zeros((5, 3, 3), np.float32),
    )
    dataset = TransitionSequenceDataset(path, horizon=3)
    item = dataset[0]
    assert item["state"].shape == (3, 31)
    assert item["object_change_mask"].shape == (3, 3)
    assert item["object_delta"].shape == (3, 3, 3)
    assert item["current_object_pose"].shape == (3, 9)
