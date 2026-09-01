"""Interface conformance for the two third-party-engine domains.

The whole point of adding Box2D and Chipmunk domains is that the existing pipeline runs
against them unchanged. That only holds if they honour the same contract as the MuJoCo and
planar environments, and the ways it can silently fail to hold are exactly the ways that
would corrupt results rather than raise: an observation array with the wrong shape gets
reshaped into nonsense, a restore that does not restore turns the counterfactual machinery
into noise, and an object that escapes the table produces a delta the change filter reads as
real motion.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.generate_transitions import (
    build_env,
    compute_diff_labels,
    extract_planar_object_state,
    flatten_state,
)
from models.layout import StateLayout
from models.policies import ScriptedPushPolicy

DOMAINS = ("billiards", "clutter")


def _env(domain: str, count: int = 3, seed: int = 0):
    return build_env(domain, count, 60, seed, {})


@pytest.mark.parametrize("domain", DOMAINS)
def test_observation_matches_shared_contract(domain: str) -> None:
    env = _env(domain)
    obs = env.reset()
    assert set(obs) == {"pusher_xy", "object_poses", "object_velocities", "goal_xy"}
    assert obs["pusher_xy"].shape == (2,)
    # 7 = xyz + quaternion; the shared planar extractor reads columns 3: as (w, x, y, z).
    assert obs["object_poses"].shape == (3, 7)
    # 6 = the MuJoCo cvel layout (angular, linear); only the planar entries are meaningful.
    assert obs["object_velocities"].shape == (3, 6)
    assert obs["goal_xy"].shape == (2,)


@pytest.mark.parametrize("domain", DOMAINS)
def test_flat_state_width_matches_layout(domain: str) -> None:
    """A width mismatch here would be reshaped downstream rather than raising."""
    for count in (3, 5):
        env = _env(domain, count=count)
        state = flatten_state(env.reset())
        assert state.shape == (StateLayout(num_objects=count).state_dim,)


@pytest.mark.parametrize("domain", DOMAINS)
def test_quaternion_round_trips_through_yaw(domain: str) -> None:
    """Yaw must survive the quaternion encoding, or a third of the pose target is lost."""
    env = _env(domain)
    env.reset()
    yaw = extract_planar_object_state(env.get_observation())[:, 2]
    assert np.all(np.abs(yaw) <= np.pi + 1e-9)
    stored = env.get_state()["object_pose"][:, 2]
    assert np.allclose(yaw, stored, atol=1e-6)


@pytest.mark.parametrize("domain", DOMAINS)
def test_objects_stay_within_bounds(domain: str) -> None:
    """An escaped object registers as a huge delta and pollutes the change statistics."""
    env = _env(domain)
    policy = ScriptedPushPolicy()
    obs = env.reset()
    for _ in range(120):
        obs, _, done, _ = env.step(policy.act(obs))
        assert np.all(np.abs(obs["object_poses"][:, :2]) < 0.35)
        if done:
            break


# Neither engine exposes its warm-start contact cache, so a restored rollout is not
# bit-identical to an uninterrupted one. These are the MEASURED residuals (see each env's
# snapshot docstring), set just above the observed value so a real regression trips the test
# while the known solver residual does not. Both are orders of magnitude below the
# corresponding domain's motion threshold (0.031 and 0.029 m), so neither can flip a change
# label.
RESTORE_TOLERANCE = {"billiards": 1e-5, "clutter": 5e-4}


@pytest.mark.parametrize("domain", DOMAINS)
def test_restore_reproduces_the_same_transition(domain: str) -> None:
    """snapshot/restore is the primitive the counterfactual validity check is built on."""
    env = _env(domain)
    obs = env.reset()
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0.4, 0.6]))
    snapshot = env.snapshot()
    first, _, _, _ = env.step(np.array([0.5, 0.3]))
    env.restore(snapshot)
    second, _, _, _ = env.step(np.array([0.5, 0.3]))
    difference = np.abs(first["object_poses"] - second["object_poses"]).max()
    assert difference < RESTORE_TOLERANCE[domain]
    # The residual must stay far below the threshold that defines "this object moved",
    # otherwise restoring could manufacture or erase a change label.
    assert difference < 0.029 / 50.0


@pytest.mark.parametrize("domain", DOMAINS)
def test_relocate_moves_only_the_named_object(domain: str) -> None:
    """The counterfactual splice depends on relocation perturbing nothing else."""
    env = _env(domain)
    env.reset()
    before = env.get_state()["object_pose"].copy()
    env.relocate_object(1, np.array([0.12, -0.05]), yaw=0.3)
    after = env.get_state()["object_pose"]
    assert np.allclose(after[0], before[0], atol=1e-9)
    assert np.allclose(after[2], before[2], atol=1e-9)
    assert np.allclose(after[1, :2], [0.12, -0.05], atol=1e-6)


@pytest.mark.parametrize("domain", DOMAINS)
def test_set_planar_state_round_trips(domain: str) -> None:
    env = _env(domain)
    env.reset()
    pose = np.array([[0.10, 0.05, 0.2], [-0.08, 0.14, -1.0], [0.0, -0.10, 2.5]])
    env.set_planar_state(np.array([0.01, -0.2]), pose)
    recovered = env.get_state()["object_pose"]
    assert np.allclose(recovered, pose, atol=1e-5)


@pytest.mark.parametrize("domain", DOMAINS)
def test_change_is_sparse_and_the_shortcut_condition_holds(domain: str) -> None:
    """Both domains must reproduce the structural property the whole study is about.

    Not a tuning check -- the bounds are wide. It asserts that most objects are unchanged at
    a typical step (so "sparse change" is true here at all), and that already-moving objects
    are far more likely to change than resting ones, which is the momentum shortcut's
    existence condition and the thing the cross-domain claim rests on.
    """
    env = _env(domain)
    policy = ScriptedPushPolicy()
    at_rest, changed = [], []
    obs = env.reset()
    for _ in range(25):
        obs = env.reset()
        for _ in range(60):
            next_obs, _, done, _ = env.step(policy.act(obs))
            mask, _ = compute_diff_labels(obs, next_obs)
            speed = np.linalg.norm(obs["object_velocities"][:, 3:5], axis=1)
            at_rest.append(speed <= 2.55e-05)
            changed.append(mask.astype(bool))
            obs = next_obs
            if done:
                break
    at_rest = np.concatenate(at_rest)
    changed = np.concatenate(changed)

    assert 0.0 < changed.mean() < 0.6, "change should be sparse in these scenes"
    assert at_rest.any() and (~at_rest).any(), "both populations must be present"
    assert changed[~at_rest].mean() > 5.0 * changed[at_rest].mean()
