"""Tests for the 2D planar pushing environment (W4 breadth).

The point of a second environment is that everything downstream runs against it unchanged,
so most of these check *interface parity* with the MuJoCo tabletop rather than physics
realism. The one physics property that must hold is the structural one the whole project is
about: only objects that are actually contacted may move.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.generate_transitions import compute_diff_labels, flatten_state
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim
from models.envs.planar_push import OBJECT_RADIUS, PUSHER_RADIUS, PlanarPushConfig, PlanarPushEnv
from models.policies import ScriptedPushPolicy


def _env(num_objects: int = 3, seed: int = 0, **kwargs) -> PlanarPushEnv:
    return PlanarPushEnv(PlanarPushConfig(num_objects=num_objects, seed=seed, **kwargs))


def test_state_layout_matches_the_tabletop() -> None:
    """The shared StateLayout must infer the same object count from the flat state."""
    for num_objects in (1, 3, 8, 20):
        env = _env(num_objects)
        state = flatten_state(env.reset())
        assert infer_num_objects_from_state_dim(state.shape[0]) == num_objects
        assert state.shape[0] == StateLayout(num_objects=num_objects).state_dim


def test_observation_keys_and_shapes_match_the_tabletop() -> None:
    env = _env(5)
    obs = env.reset()
    assert set(obs) == {"pusher_xy", "object_poses", "object_velocities", "goal_xy"}
    assert obs["pusher_xy"].shape == (2,)
    # 7 = xyz + quaternion, so extract_planar_object_state works verbatim.
    assert obs["object_poses"].shape == (5, 7)
    assert obs["object_velocities"].shape == (5, 6)
    assert obs["goal_xy"].shape == (2,)


def test_quaternion_encodes_yaw_recoverably() -> None:
    env = _env(2)
    env.reset()
    env.object_yaw[:] = np.array([0.7, -1.3])
    poses = env.get_observation()["object_poses"]
    recovered = np.array([2.0 * np.arctan2(q[6], q[3]) for q in poses])
    np.testing.assert_allclose(recovered, np.array([0.7, -1.3]), atol=1e-9)


def test_untouched_objects_do_not_move() -> None:
    """The structural property the whole study rests on."""
    env = _env(4, seed=3)
    obs = env.reset()
    # Park every object far from the pusher, then act; nothing may move.
    env.set_planar_state(
        np.array([0.0, -0.25]),
        np.array([[0.2, 0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
    )
    before = env.get_state()["object_pose"].copy()
    obs = env.get_observation()
    next_obs, _, _, _ = env.step(np.array([0.0, 1.0]))
    mask, _ = compute_diff_labels(obs, next_obs)
    assert mask.sum() == 0
    np.testing.assert_allclose(env.get_state()["object_pose"], before, atol=1e-9)


def test_a_contacted_object_moves_away_from_the_pusher() -> None:
    env = _env(1)
    env.reset()
    env.set_planar_state(np.array([0.0, 0.0]), np.array([[PUSHER_RADIUS + OBJECT_RADIUS, 0.0, 0.0]]))
    before = env.object_xy[0].copy()
    env.step(np.array([1.0, 0.0]))
    after = env.object_xy[0]
    assert after[0] > before[0], "object should be driven along +x"


def test_change_is_sparse_under_the_scripted_policy() -> None:
    """Most objects must be stationary most of the time, or the domain is the wrong shape."""
    env = _env(5, seed=1)
    fractions = []
    obs = env.reset()
    policy = ScriptedPushPolicy()
    for _ in range(120):
        previous = obs
        obs, _, done, _ = env.step(policy.act(obs))
        mask, _ = compute_diff_labels(previous, obs)
        fractions.append(mask.mean())
        if done:
            break
    mean_changed = float(np.mean(fractions))
    assert 0.0 < mean_changed < 0.5, f"changed fraction {mean_changed:.3f} is not sparse"


def test_snapshot_restore_is_exact_and_replays_identically() -> None:
    env = _env(4, seed=7)
    obs = env.reset()
    policy = ScriptedPushPolicy()
    for _ in range(15):
        obs, _, _, _ = env.step(policy.act(obs))

    snapshot = env.snapshot()
    action = np.array([0.4, -0.6])
    first, _, _, _ = env.step(action)
    env.restore(snapshot)
    second, _, _, _ = env.step(action)
    np.testing.assert_allclose(first["object_poses"], second["object_poses"], atol=1e-12)
    np.testing.assert_allclose(first["pusher_xy"], second["pusher_xy"], atol=1e-12)


def test_relocate_object_touches_only_that_object() -> None:
    env = _env(4, seed=2)
    env.reset()
    before = env.get_state()["object_pose"].copy()
    env.relocate_object(2, np.array([0.24, -0.24]), yaw=0.3)
    after = env.get_state()["object_pose"]
    np.testing.assert_allclose(np.delete(after, 2, axis=0), np.delete(before, 2, axis=0), atol=1e-12)
    np.testing.assert_allclose(after[2], np.array([0.24, -0.24, 0.3]), atol=1e-12)
    assert np.all(env.object_vel[2] == 0.0)


def test_relocate_rejects_a_bad_index() -> None:
    env = _env(2)
    env.reset()
    with pytest.raises(IndexError):
        env.relocate_object(5, np.array([0.0, 0.0]))


def test_set_planar_state_round_trips_exactly() -> None:
    """Lossless here, unlike the tabletop, because the planar state IS the full state."""
    env = _env(3, seed=5)
    env.reset()
    pose = np.array([[0.1, 0.1, 0.5], [-0.15, 0.05, -1.0], [0.0, -0.15, 2.0]])
    pusher = np.array([0.05, -0.1])
    env.set_planar_state(pusher, pose)

    state = flatten_state(env.get_observation())
    layout = StateLayout(num_objects=3)
    np.testing.assert_allclose(state[0:2], pusher, atol=1e-12)
    np.testing.assert_allclose(
        state[layout.object_pose_slice].reshape(3, POSE_DIM), pose, atol=1e-9
    )


def test_set_planar_state_rejects_a_mismatched_object_count() -> None:
    env = _env(3)
    env.reset()
    with pytest.raises(ValueError, match="rows"):
        env.set_planar_state(np.zeros(2), np.zeros((2, 3)))


def test_objects_stay_inside_the_configured_bounds() -> None:
    env = _env(3, seed=4, object_bounds=(-0.2, 0.2))
    obs = env.reset()
    policy = ScriptedPushPolicy()
    for _ in range(80):
        obs, _, done, _ = env.step(policy.act(obs))
        assert np.all(np.abs(env.object_xy) <= 0.2 + 1e-9)
        if done:
            break


def test_placement_respects_minimum_separation() -> None:
    env = _env(6, seed=11, min_object_separation=0.09)
    env.reset()
    for i in range(6):
        for j in range(i + 1, 6):
            assert np.linalg.norm(env.object_xy[i] - env.object_xy[j]) > 0.09 - 1e-9


def test_high_object_counts_are_supported() -> None:
    """A key reason this env exists: it must scale past what the tabletop affords."""
    env = _env(30, seed=0, min_object_separation=0.06)
    obs = env.reset()
    assert obs["object_poses"].shape == (30, 7)
    env.step(np.array([0.3, 0.3]))
