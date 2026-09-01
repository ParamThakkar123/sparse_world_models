"""Tests for the latent dynamics control baseline (W5).

This model exists to answer "would *any* learned model do?", so if it is quietly broken the
comparison it anchors is worthless in a direction that flatters our own method. The
detached-consistency detail matters most: without it the latent can collapse to a constant,
which minimises the consistency term perfectly while destroying the model.
"""

from __future__ import annotations

import pytest
import torch

from models.latent_dynamics import LatentDynamicsModel

STATE_DIM, ACTION_DIM, NUM_OBJECTS = 31, 2, 3


def _model(**kwargs) -> LatentDynamicsModel:
    torch.manual_seed(0)
    return LatentDynamicsModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, num_objects=NUM_OBJECTS,
        latent_dim=kwargs.pop("latent_dim", 16), hidden_dim=kwargs.pop("hidden_dim", 32),
        num_layers=kwargs.pop("num_layers", 2), **kwargs,
    )


def _batch(size: int = 8):
    torch.manual_seed(1)
    return (
        torch.randn(size, STATE_DIM),
        torch.randn(size, ACTION_DIM),
        torch.randn(size, STATE_DIM),
        torch.randn(size, NUM_OBJECTS, 3),
        torch.randn(size, NUM_OBJECTS, 3),
    )


def test_forward_returns_object_poses() -> None:
    model = _model()
    state, action, *_ = _batch()
    out = model(state, action)
    assert out.shape == (8, NUM_OBJECTS, 3)


def test_encode_decode_shapes() -> None:
    model = _model(latent_dim=16)
    state, action, *_ = _batch()
    latent = model.encode(state)
    assert latent.shape == (8, 16)
    assert model.step_latent(latent, action).shape == (8, 16)
    assert model.decode(latent).shape == (8, NUM_OBJECTS, 3)


def test_consistency_target_is_detached() -> None:
    """The encoder must not be trained *through* the consistency target.

    If it were, the cheapest way to satisfy consistency is to make ``encode`` constant --
    a perfect score on that term and a useless model. Verified by checking the encoder
    gradient equals the one produced when the target is an explicit constant.
    """
    model = _model()
    state, action, next_state, current_pose, next_pose = _batch()

    model.zero_grad()
    model.losses(state, action, next_state, current_pose, next_pose,
                 consistency_weight=1.0, reconstruction_weight=0.0).consistency.backward()
    from_model = [p.grad.clone() for p in model.encoder.parameters()]

    model.zero_grad()
    target = model.encode(next_state).detach()
    manual = torch.nn.functional.mse_loss(model.step_latent(model.encode(state), action), target)
    manual.backward()
    from_manual = [p.grad.clone() for p in model.encoder.parameters()]

    for a, b in zip(from_model, from_manual):
        torch.testing.assert_close(a, b)


def test_losses_are_finite_and_trainable() -> None:
    model = _model()
    batch = _batch()
    losses = model.losses(*batch)
    for value in (losses.total, losses.reconstruction, losses.consistency, losses.prediction):
        assert torch.isfinite(value)
    losses.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_reduces_prediction_error() -> None:
    """A short fit on one fixed batch must drive the prediction term down."""
    model = _model()
    state, action, next_state, current_pose, next_pose = _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = model.losses(state, action, next_state, current_pose, next_pose).prediction.item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = model.losses(state, action, next_state, current_pose, next_pose)
        loss.total.backward()
        optimizer.step()
    last = model.losses(state, action, next_state, current_pose, next_pose).prediction.item()
    assert last < first * 0.5, f"prediction loss barely moved: {first:.4f} -> {last:.4f}"


def test_loss_weights_are_respected() -> None:
    model = _model()
    batch = _batch()
    only_prediction = model.losses(*batch, consistency_weight=0.0, reconstruction_weight=0.0)
    torch.testing.assert_close(only_prediction.total, only_prediction.prediction)


def test_latent_forward_wrapper_matches_the_planner_interface() -> None:
    """planning_mpc's ModelForward contract: (states, actions) -> (B, N, 3)."""
    from experiments.planning_mpc import LatentForward
    from models import StateLayout

    model = _model()
    layout = StateLayout(num_objects=NUM_OBJECTS)
    forward = LatentForward(model, NUM_OBJECTS, layout, torch.device("cpu"))
    state, action, *_ = _batch()
    with torch.no_grad():
        out = forward(state, action)
    assert out.shape == (8, NUM_OBJECTS, 3)


def test_latent_is_registered_as_a_model_condition() -> None:
    """It must be dispatched as a learned model, not fall through to a policy branch."""
    from experiments.planning_mpc import CONDITIONS, MODEL_CONDITIONS

    assert "latent" in CONDITIONS
    assert "latent" in MODEL_CONDITIONS
    assert set(MODEL_CONDITIONS) <= set(CONDITIONS)


def test_decoder_output_dimension_follows_object_count() -> None:
    for num_objects in (1, 5, 12):
        model = LatentDynamicsModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, num_objects=num_objects,
            latent_dim=8, hidden_dim=16, num_layers=2,
        )
        out = model.decode(torch.randn(4, 8))
        assert out.shape == (4, num_objects, 3)


def test_rejects_mismatched_action_dim() -> None:
    model = _model()
    state, _, *_ = _batch()
    with pytest.raises(RuntimeError):
        model(state, torch.randn(8, ACTION_DIM + 3))
