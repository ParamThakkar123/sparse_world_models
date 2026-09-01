"""Tests for counterfactual splicing (W3).

These generate data that gets trained on, so a silent correctness bug here would show up as
a fabricated result rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from models import POSE_DIM, StateLayout
from models.counterfactual import (
    CONTACT_RADIUS,
    INTERACTION_MARGIN,
    OBJECT_CLEARANCE,
    generate_counterfactuals,
    placement_is_clear,
    recompute_labels,
)

NUM_OBJECTS = 3
LAYOUT = StateLayout(num_objects=NUM_OBJECTS)


def _transition(poses: np.ndarray, next_poses: np.ndarray, pusher=(0.0, -0.22), action=(0.0, 0.0)):
    state = np.zeros(LAYOUT.state_dim, dtype=np.float32)
    next_state = np.zeros(LAYOUT.state_dim, dtype=np.float32)
    state[0:2] = pusher
    next_state[0:2] = pusher
    state[LAYOUT.object_pose_slice] = poses.reshape(-1)
    next_state[LAYOUT.object_pose_slice] = next_poses.reshape(-1)
    return state[None, :], np.asarray([action], dtype=np.float32), next_state[None, :]


def test_placement_rejects_overlap_with_another_object() -> None:
    other = np.array([[0.0, 0.0]])
    empty = np.empty((0, 2))
    far = np.array([10.0, 10.0])
    assert not placement_is_clear(
        np.array([OBJECT_CLEARANCE * 0.5, 0.0]),
        other_xy=other, moved_xy=empty, pusher_xy=far, pusher_next_xy=far,
    )
    assert placement_is_clear(
        np.array([OBJECT_CLEARANCE * 2.0, 0.0]),
        other_xy=other, moved_xy=empty, pusher_xy=far, pusher_next_xy=far,
    )


def test_placement_rejects_proximity_to_something_that_moved() -> None:
    """A pose near a moving object might have been struck, which the label would deny."""
    empty = np.empty((0, 2))
    far = np.array([10.0, 10.0])
    moved = np.array([[0.0, 0.0]])
    assert not placement_is_clear(
        np.array([INTERACTION_MARGIN * 0.5, 0.0]),
        other_xy=empty, moved_xy=moved, pusher_xy=far, pusher_next_xy=far,
    )


def test_placement_rejects_points_on_the_pusher_sweep_not_just_its_endpoints() -> None:
    """Testing only start and end would miss an object the pusher passed straight through."""
    empty = np.empty((0, 2))
    start, end = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    midpoint = np.array([0.0, 0.0])
    assert np.linalg.norm(midpoint - start) > CONTACT_RADIUS + INTERACTION_MARGIN
    assert not placement_is_clear(
        midpoint, other_xy=empty, moved_xy=empty, pusher_xy=start, pusher_next_xy=end
    )


def test_masked_splicing_only_relocates_unchanged_objects() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
    next_poses = poses.copy()
    next_poses[0, 0] += 0.05  # object 0 moved
    state, action, next_state = _transition(poses, next_poses)
    mask = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    cf_state, _, cf_next, stats = generate_counterfactuals(
        state, action, next_state, mask,
        rng=np.random.default_rng(0), num_samples=25,
        pose_pool=np.array([[0.9, 0.9, 0.0]], dtype=np.float32),
    )
    assert stats.accepted > 0
    cf_poses = cf_state[:, LAYOUT.object_pose_slice].reshape(-1, NUM_OBJECTS, POSE_DIM)
    # The changed object must never be touched.
    np.testing.assert_allclose(cf_poses[:, 0], np.tile(poses[0], (cf_poses.shape[0], 1)), atol=1e-6)
    # Its recorded motion must survive into the synthetic next state.
    cf_next_poses = cf_next[:, LAYOUT.object_pose_slice].reshape(-1, NUM_OBJECTS, POSE_DIM)
    np.testing.assert_allclose(
        cf_next_poses[:, 0], np.tile(next_poses[0], (cf_next_poses.shape[0], 1)), atol=1e-6
    )


def test_relocated_objects_are_labelled_stationary() -> None:
    """The defining assumption: a causally inert object's next pose is its new pose."""
    poses = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
    next_poses = poses.copy()
    next_poses[0, 0] += 0.05
    state, action, next_state = _transition(poses, next_poses)
    mask = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    cf_state, _, cf_next, _ = generate_counterfactuals(
        state, action, next_state, mask,
        rng=np.random.default_rng(1), num_samples=20,
        pose_pool=np.array([[0.9, 0.9, 0.5]], dtype=np.float32),
    )
    current = cf_state[:, LAYOUT.object_pose_slice].reshape(-1, NUM_OBJECTS, POSE_DIM)
    following = cf_next[:, LAYOUT.object_pose_slice].reshape(-1, NUM_OBJECTS, POSE_DIM)
    np.testing.assert_allclose(current[:, 1:], following[:, 1:], atol=1e-6)


def test_no_free_object_is_rejected_not_fabricated() -> None:
    poses = np.zeros((NUM_OBJECTS, POSE_DIM), dtype=np.float32)
    state, action, next_state = _transition(poses, poses.copy())
    everything_moved = np.ones((1, NUM_OBJECTS), dtype=np.float32)

    cf_state, _, _, stats = generate_counterfactuals(
        state, action, next_state, everything_moved,
        rng=np.random.default_rng(0), num_samples=10,
    )
    assert cf_state.shape[0] == 0
    assert stats.accepted == 0
    assert stats.rejected_no_free_object == 10


def test_unmasked_mode_ignores_causal_structure() -> None:
    """The ablation must actually relocate objects the mask says are changing."""
    poses = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
    next_poses = poses.copy()
    next_poses[0, 0] += 0.05
    state, action, next_state = _transition(poses, next_poses)
    mask = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    cf_state, _, _, stats = generate_counterfactuals(
        state, action, next_state, mask,
        rng=np.random.default_rng(0), num_samples=20, respect_mask=False,
        pose_pool=np.array([[0.9, 0.9, 0.0]], dtype=np.float32),
    )
    cf_poses = cf_state[:, LAYOUT.object_pose_slice].reshape(-1, NUM_OBJECTS, POSE_DIM)
    moved_object_relocated = np.any(np.abs(cf_poses[:, 0] - poses[0]).max(axis=-1) > 1e-6)
    assert moved_object_relocated, "unmasked mode must be free to relocate a changed object"
    assert stats.accepted > 0


def test_recompute_labels_derives_from_synthetic_poses_not_the_source() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
    next_poses = poses.copy()
    next_poses[1, 0] += 0.05  # only object 1 moves
    state, action, next_state = _transition(poses, next_poses)

    mask, delta = recompute_labels(state, next_state, position_eps=0.002, yaw_eps=0.05)
    np.testing.assert_allclose(mask[0], np.array([0.0, 1.0, 0.0]))
    assert delta[0, 1, 0] == pytest.approx(0.05, abs=1e-6)


def test_recompute_labels_wraps_yaw() -> None:
    """A yaw crossing pi must not register as a near-2pi rotation."""
    poses = np.zeros((NUM_OBJECTS, POSE_DIM), dtype=np.float32)
    poses[0, 2] = np.pi - 0.01
    next_poses = poses.copy()
    next_poses[0, 2] = -np.pi + 0.01
    state, _, next_state = _transition(poses, next_poses)

    _, delta = recompute_labels(state, next_state, position_eps=0.002, yaw_eps=0.05)
    assert abs(delta[0, 0, 2]) == pytest.approx(0.02, abs=1e-5)


def test_acceptance_rate_reported() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]], dtype=np.float32)
    state, action, next_state = _transition(poses, poses.copy())
    mask = np.zeros((1, NUM_OBJECTS), dtype=np.float32)
    _, _, _, stats = generate_counterfactuals(
        state, action, next_state, mask,
        rng=np.random.default_rng(0), num_samples=10,
        pose_pool=np.array([[0.9, 0.9, 0.0]], dtype=np.float32),
    )
    summary = stats.as_dict()
    assert summary["proposed"] == 10
    assert 0.0 <= summary["acceptance_rate"] <= 1.0
