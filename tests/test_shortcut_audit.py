"""Tests for the interaction filter and the trivial-rule battery.

These two pieces now carry the project's central negative result -- that both existing
benchmarks are solvable without learning -- so their correctness matters more than most. The
failure modes are quiet ones: a filter that selects the wrong population still produces a
plausible-looking benchmark, and a trivial rule with an indexing bug still produces a
plausible-looking F1. Each test below pins a property that would be invisible in the outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.create_hard_subset import (
    PUSHER_ACTION_SCALE,
    compute_interaction_keep_mask,
    compute_onset_keep_mask,
)
from experiments.onset_shortcut_audit import object_distance_matrix, trivial_rules
from models.layout import StateLayout

COUNT = 3
LAYOUT = StateLayout(num_objects=COUNT)


def _dataset(
    poses: np.ndarray,
    velocities: np.ndarray,
    deltas: np.ndarray,
    pusher: np.ndarray,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the minimal dataset dict the filters and rules consume."""
    rows = poses.shape[0]
    state = np.zeros((rows, LAYOUT.state_dim))
    state[:, 0:2] = pusher
    state[:, LAYOUT.object_pose_slice] = poses.reshape(rows, -1)
    velocity_block = np.zeros((rows, COUNT, 6))
    velocity_block[:, :, 3:5] = velocities
    state[:, LAYOUT.object_velocity_slice] = velocity_block.reshape(rows, -1)
    changed = (np.linalg.norm(deltas[:, :, :2], axis=2) > 1e-9).astype(np.float32)
    return {
        "s_t": state, "state": state, "a_t": action, "action": action,
        "object_delta": deltas, "object_change_mask": changed,
        "target_mask": changed,
    }


# ------------------------------------------------------------------ interaction filter

def test_interaction_filter_rejects_direct_pusher_contact() -> None:
    """The nearest object starting to move is a DIRECT contact and must not qualify."""
    poses = np.array([[[0.0, 0.0, 0.0], [0.2, 0.2, 0.0], [-0.2, -0.2, 0.0]]])
    velocities = np.zeros((1, COUNT, 2))
    deltas = np.zeros((1, COUNT, 3))
    deltas[0, 0, 0] = 0.05  # object 0 moves
    pusher = np.array([[0.02, 0.0]])  # ...and object 0 is the nearest to the pusher
    action = np.zeros((1, 2))
    keep = compute_interaction_keep_mask(_dataset(poses, velocities, deltas, pusher, action), 0.02)
    assert keep.tolist() == [False]


def test_interaction_filter_accepts_indirect_change() -> None:
    """A non-nearest object starting to move could only have been reached through another."""
    poses = np.array([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [-0.25, -0.25, 0.0]]])
    velocities = np.zeros((1, COUNT, 2))
    deltas = np.zeros((1, COUNT, 3))
    deltas[0, 1, 0] = 0.05  # object 1 moves, but object 0 is nearer the pusher
    pusher = np.array([[-0.02, 0.0]])
    action = np.zeros((1, 2))
    keep = compute_interaction_keep_mask(_dataset(poses, velocities, deltas, pusher, action), 0.02)
    assert keep.tolist() == [True]


def test_interaction_filter_requires_the_object_to_have_been_at_rest() -> None:
    """An already-moving object continuing is not an onset, however far from the pusher."""
    poses = np.array([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [-0.25, -0.25, 0.0]]])
    velocities = np.zeros((1, COUNT, 2))
    velocities[0, 1] = (0.5, 0.0)  # object 1 is already moving
    deltas = np.zeros((1, COUNT, 3))
    deltas[0, 1, 0] = 0.05
    pusher = np.array([[-0.02, 0.0]])
    keep = compute_interaction_keep_mask(
        _dataset(poses, velocities, deltas, pusher, np.zeros((1, 2))), 0.02
    )
    assert keep.tolist() == [False]


def test_interaction_filter_respects_the_displacement_threshold() -> None:
    poses = np.array([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [-0.25, -0.25, 0.0]]])
    deltas = np.zeros((1, COUNT, 3))
    deltas[0, 1, 0] = 0.001  # real but far below the threshold
    keep = compute_interaction_keep_mask(
        _dataset(poses, np.zeros((1, COUNT, 2)), deltas, np.array([[-0.02, 0.0]]), np.zeros((1, 2))),
        0.02,
    )
    assert keep.tolist() == [False]


def test_interaction_filter_uses_the_POST_action_pusher_position() -> None:
    """The pusher moves before the contact resolves; using its old position mislabels events."""
    poses = np.array([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [-0.25, -0.25, 0.0]]])
    deltas = np.zeros((1, COUNT, 3))
    deltas[0, 1, 0] = 0.05
    # Pre-action the pusher is nearest object 0; a full-throttle +x action carries it past the
    # midpoint so object 1 becomes nearest, which flips this from indirect to direct.
    pusher = np.array([[0.045, 0.0]])
    still = compute_interaction_keep_mask(
        _dataset(poses, np.zeros((1, COUNT, 2)), deltas, pusher, np.zeros((1, 2))), 0.02
    )
    moved = compute_interaction_keep_mask(
        _dataset(poses, np.zeros((1, COUNT, 2)), deltas, pusher, np.ones((1, 2)) * np.array([1.0, 0.0])),
        0.02,
    )
    assert still.tolist() == [True]
    assert moved.tolist() == [False]
    # Guard the fixture itself: the action must actually have moved the pusher past the midpoint.
    assert 0.045 + PUSHER_ACTION_SCALE > 0.05


def test_interaction_is_a_strict_subset_of_onset() -> None:
    """Every interaction event is an onset event; the converse is what the filter removes."""
    rng = np.random.default_rng(0)
    rows = 400
    poses = np.zeros((rows, COUNT, 3))
    poses[:, :, :2] = rng.uniform(-0.25, 0.25, size=(rows, COUNT, 2))
    velocities = np.where(rng.random((rows, COUNT, 1)) < 0.3, rng.normal(0, 0.4, (rows, COUNT, 2)), 0.0)
    deltas = np.zeros((rows, COUNT, 3))
    mask = rng.random((rows, COUNT)) < 0.3
    deltas[:, :, 0] = np.where(mask, 0.05, 0.0)
    data = _dataset(poses, velocities, deltas, rng.uniform(-0.2, 0.2, (rows, 2)),
                    rng.uniform(-1, 1, (rows, 2)))
    onset = compute_onset_keep_mask(data, 0.02)
    interaction = compute_interaction_keep_mask(data, 0.02)
    assert np.all(interaction <= onset), "an interaction event that is not an onset event"
    assert interaction.sum() < onset.sum(), "the filter should be strictly more selective"


# ---------------------------------------------------------------------- trivial rules

def _battery_dataset() -> dict[str, np.ndarray]:
    poses = np.array([[[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.30, 0.30, 0.0]]])
    velocities = np.zeros((1, COUNT, 2))
    velocities[0, 0] = (0.5, 0.0)  # object 0 is moving
    deltas = np.zeros((1, COUNT, 3))
    return _dataset(poses, velocities, deltas, np.array([[-0.05, 0.0]]), np.zeros((1, 2)))


def test_nearest_and_second_nearest_are_disjoint_single_selections() -> None:
    rules = trivial_rules(_battery_dataset(), COUNT, radius=0.05)
    nearest, second = rules["nearest_to_pusher"], rules["second_nearest_to_pusher"]
    assert nearest.sum() == 1 and second.sum() == 1
    assert not np.any(nearest * second), "nearest and second-nearest must not coincide"
    assert np.array_equal(rules["two_nearest_to_pusher"], np.maximum(nearest, second))


def test_near_a_mover_fires_next_to_a_moving_object_only() -> None:
    """The chain shortcut: adjacency to something already in motion."""
    rules = trivial_rules(_battery_dataset(), COUNT, radius=0.08)
    near_mover = rules["near_a_mover"][0]
    # Object 0 moves; object 1 is 0.06 away from it, object 2 is far.
    assert near_mover[1] == 1.0
    assert near_mover[2] == 0.0


def test_near_a_mover_is_silent_when_nothing_moves() -> None:
    """With no movers the min over an empty set is inf; it must predict nothing, not everything."""
    data = _battery_dataset()
    data["state"][:, LAYOUT.object_velocity_slice] = 0.0
    rules = trivial_rules(data, COUNT, radius=0.5)
    assert rules["near_a_mover"].sum() == 0.0


def test_object_distance_matrix_excludes_self_pairs() -> None:
    """A zero self-distance would make every object trivially 'near a mover'."""
    matrix = object_distance_matrix(_battery_dataset(), COUNT)
    assert np.all(np.isinf(np.diagonal(matrix[0])))
    assert matrix[0, 0, 1] == pytest.approx(0.06)


def test_every_rule_returns_one_prediction_per_object() -> None:
    data = _battery_dataset()
    for name, prediction in trivial_rules(data, COUNT, radius=0.05).items():
        assert prediction.shape == (1, COUNT), f"{name} returned {prediction.shape}"
        assert set(np.unique(prediction)) <= {0.0, 1.0}, f"{name} is not binary"
