from __future__ import annotations

import numpy as np
import torch

from experiments.rollout_horizon_error import (
    build_ground_truth_arrays,
    enumerate_rollout_samples,
    per_object_pose_l2,
    reconstruct_episode_lengths,
    wrap_angle,
)


def test_reconstruct_episode_lengths_splits_on_done() -> None:
    done = np.array([False, True, False, False, True], dtype=bool)
    assert reconstruct_episode_lengths(done) == [2, 3]


def test_reconstruct_episode_lengths_requires_terminal_done() -> None:
    done = np.array([False, True, False], dtype=bool)
    try:
        reconstruct_episode_lengths(done)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for dataset not ending with done=True.")


def _toy_dataset() -> dict[str, np.ndarray]:
    # One 2-object episode of 2 transitions (states t0, t1, t2). State layout for
    # 2 objects: pusher(2) + pose(6) + velocity(12) + goal(2) = 22 dims.
    state_dim = 22
    s_t = np.zeros((2, state_dim), dtype=np.float32)
    s_t1 = np.zeros((2, state_dim), dtype=np.float32)
    # Encode a recognizable value in the first pose coordinate at each time.
    s_t[0, 2] = 10.0  # state at t0
    s_t[1, 2] = 11.0  # state at t1
    s_t1[0, 2] = 11.0  # t1 (chains with s_t[1])
    s_t1[1, 2] = 12.0  # t2
    return {
        "s_t": s_t,
        "s_t1": s_t1,
        "a_t": np.array([[0.5, -0.5], [0.25, 0.75]], dtype=np.float32),
        "done": np.array([False, True], dtype=bool),
    }


def test_build_ground_truth_arrays_reconstructs_state_sequence() -> None:
    gt_states, gt_actions, lengths = build_ground_truth_arrays(_toy_dataset())

    assert lengths == [2]
    # T + 1 = 3 states, and the pose coordinate traces the full trajectory.
    assert gt_states.shape[0] == 3
    np.testing.assert_allclose(gt_states[:, 2], [10.0, 11.0, 12.0])
    # Terminal-state action row is zero-padded and never read.
    np.testing.assert_allclose(gt_actions[2], [0.0, 0.0])
    np.testing.assert_allclose(gt_actions[0], [0.5, -0.5])


def test_enumerate_rollout_samples_covers_every_start() -> None:
    samples = enumerate_rollout_samples([2, 3], max_horizon=20)

    # 2 + 3 = 5 sliding starts.
    assert samples.base.shape[0] == 5
    # First episode: starts at global offset 0 (t0=0,1); second episode at 0+3=3.
    np.testing.assert_array_equal(samples.base.numpy(), [0, 0, 3, 3, 3])
    np.testing.assert_array_equal(samples.t0.numpy(), [0, 1, 0, 1, 2])
    # Reachable horizon = length - t0 (capped by max_horizon).
    np.testing.assert_array_equal(samples.max_h.numpy(), [2, 1, 3, 2, 1])


def test_enumerate_rollout_samples_caps_at_max_horizon() -> None:
    samples = enumerate_rollout_samples([10], max_horizon=3)
    assert int(samples.max_h.max()) == 3


def test_per_object_pose_l2_wraps_angle_and_splits_translation() -> None:
    pred = torch.tensor([[[0.0, 0.0, 3.10]]])
    gt = torch.tensor([[[3.0, 4.0, -3.10]]])

    pose_l2, xy_l2 = per_object_pose_l2(pred, gt)

    # Translation error is a clean 3-4-5 triangle.
    np.testing.assert_allclose(xy_l2.numpy(), [[5.0]], atol=1e-5)
    # Angle error wraps: 3.10 - (-3.10) = 6.20 -> ~-0.0832 rad, not 6.20.
    expected_theta = float(wrap_angle(torch.tensor(6.20)))
    expected = np.sqrt(25.0 + expected_theta**2)
    np.testing.assert_allclose(pose_l2.numpy(), [[expected]], atol=1e-5)


def test_noop_rollout_error_equals_displacement_from_start() -> None:
    # A no-op rollout holds the start pose, so its horizon-h error must equal the
    # ground-truth displacement between the start pose and the pose h steps later.
    dataset = _toy_dataset()
    gt_states, _, lengths = build_ground_truth_arrays(dataset)
    from models import POSE_DIM, StateLayout

    layout = StateLayout(num_objects=2)
    gt_pose = gt_states[:, layout.object_pose_slice].reshape(-1, 2, POSE_DIM)
    # Start at t0=0, horizon 2: pose coord goes 10 -> 12, displacement 2.0 on object 0.
    pred = torch.from_numpy(gt_pose[0:1])  # held at start
    target = torch.from_numpy(gt_pose[2:3])
    pose_l2, _ = per_object_pose_l2(pred, target)
    np.testing.assert_allclose(pose_l2.numpy()[0, 0], 2.0, atol=1e-5)
