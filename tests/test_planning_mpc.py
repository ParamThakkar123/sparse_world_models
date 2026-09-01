from __future__ import annotations

import numpy as np
import torch

from experiments.planning_mpc import (
    CEMConfig,
    ModelForward,
    advance_pusher,
    cem_plan,
    imagined_target_trajectory,
    reconstruct_state,
    sequence_cost,
)
from models import POSE_DIM, StateLayout


def test_advance_pusher_clips_action_scales_and_bounds() -> None:
    pusher = torch.tensor([[0.0, 0.25]])
    # Action beyond [-1, 1] is clipped, then scaled, then bounded.
    action = torch.tensor([[2.0, 1.0]])  # clips to [1, 1]
    out = advance_pusher(pusher, action, action_scale=0.04, bounds=(-0.26, 0.26))
    # x: 0.0 + 1*0.04 = 0.04 ; y: 0.25 + 1*0.04 = 0.29 -> bounded to 0.26
    np.testing.assert_allclose(out.numpy(), [[0.04, 0.26]], atol=1e-6)


def test_reconstruct_state_overwrites_pose_and_pusher_preserves_rest() -> None:
    layout = StateLayout(num_objects=2)
    dim = layout.state_dim  # 2 + 6 + 12 + 2 = 22
    state = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    next_pose = torch.full((1, 2, POSE_DIM), 99.0)
    next_pusher = torch.tensor([[-1.0, -2.0]])

    new_state = reconstruct_state(state, next_pose, next_pusher, layout)

    # Pusher slice overwritten.
    np.testing.assert_allclose(new_state[0, 0:2].numpy(), [-1.0, -2.0])
    # Object-pose slice overwritten with the predicted poses.
    np.testing.assert_allclose(new_state[0, layout.object_pose_slice].numpy(), np.full(6, 99.0))
    # Velocity + goal slices untouched.
    np.testing.assert_allclose(
        new_state[0, layout.object_velocity_slice].numpy(),
        state[0, layout.object_velocity_slice].numpy(),
    )
    np.testing.assert_allclose(
        new_state[0, layout.goal_slice].numpy(), state[0, layout.goal_slice].numpy()
    )


class _ConstantDeltaForward(ModelForward):
    """Fake model: every object advances by a fixed planar delta each step."""

    def __init__(self, layout: StateLayout, delta: torch.Tensor):
        super().__init__(layout.num_objects, layout, torch.device("cpu"))
        self.delta = delta

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        current = states[:, self.layout.object_pose_slice].reshape(-1, self.num_objects, POSE_DIM)
        return current + self.delta


def test_imagined_target_trajectory_tracks_known_dynamics() -> None:
    layout = StateLayout(num_objects=1)
    state = torch.zeros(1, layout.state_dim)  # target starts at origin
    forward = _ConstantDeltaForward(layout, torch.tensor([[[0.1, 0.0, 0.0]]]))
    # Single 3-step sequence; actions do not affect the fake object dynamics.
    actions = torch.zeros(1, 3, 2)

    target_xy, pusher_xy = imagined_target_trajectory(
        forward, state[0], actions, layout, target_object=0,
        action_scale=0.04, pusher_bounds=(-0.26, 0.26),
    )
    # Object marches +0.1 in x each of 3 steps.
    np.testing.assert_allclose(target_xy[0, :, 0].numpy(), [0.1, 0.2, 0.3], atol=1e-6)
    # Pusher stays put under zero action.
    np.testing.assert_allclose(pusher_xy.numpy(), np.zeros((1, 3, 2)), atol=1e-6)


def test_sequence_cost_prefers_trajectory_ending_at_goal() -> None:
    goal = torch.tensor([1.0, 1.0])
    config = CEMConfig(terminal_weight=3.0, proximity_weight=0.0)
    # Candidate A ends at the goal; candidate B stays at the origin.
    at_goal = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    at_origin = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    pusher = torch.zeros(1, 2, 2)
    cost_goal = sequence_cost(at_goal, pusher, goal, config)
    cost_origin = sequence_cost(at_origin, pusher, goal, config)
    assert float(cost_goal.item()) < float(cost_origin.item())
    np.testing.assert_allclose(cost_goal.numpy(), [0.0], atol=1e-6)


def test_cem_plan_converges_to_quadratic_optimum() -> None:
    # A model-free objective with a known optimum inside the action box: the optimal
    # sequence is a constant 0.5 everywhere. CEM should recover it closely.
    horizon = 4
    device = torch.device("cpu")
    optimum = torch.full((horizon, 2), 0.5)

    def evaluate(samples: torch.Tensor) -> torch.Tensor:
        return ((samples - optimum) ** 2).sum(dim=(1, 2))

    config = CEMConfig(horizon=horizon, num_samples=512, cem_iters=8, elite_frac=0.1, init_std=0.6)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    best_sequence, best_cost = cem_plan(evaluate, horizon, device, config, torch.zeros(horizon, 2), generator)

    np.testing.assert_allclose(best_sequence.numpy(), optimum.numpy(), atol=0.1)
    assert best_cost < 0.05


def test_cem_plan_respects_action_bounds() -> None:
    # Optimum lies outside the [-1, 1] box; CEM must clamp and return a boundary action.
    horizon = 3
    device = torch.device("cpu")
    unreachable = torch.full((horizon, 2), 5.0)

    def evaluate(samples: torch.Tensor) -> torch.Tensor:
        return ((samples - unreachable) ** 2).sum(dim=(1, 2))

    config = CEMConfig(horizon=horizon, num_samples=1024, cem_iters=8, elite_frac=0.1)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    best_sequence, _ = cem_plan(evaluate, horizon, device, config, torch.zeros(horizon, 2), generator)

    # Every action stays inside the box, and the best solution saturates near the
    # upper bound (the closest reachable point to the unreachable optimum).
    assert float(best_sequence.max().item()) <= 1.0 + 1e-6
    assert float(best_sequence.min().item()) >= 0.85


def test_env_snapshot_restore_roundtrip() -> None:
    from models.envs import TabletopPushConfig, TabletopPushEnv

    env = TabletopPushEnv(TabletopPushConfig(num_objects=3, seed=7))
    env.reset()
    snap = env.snapshot()
    qpos_before = snap["qpos"].copy()

    # Perturb the simulator. The pusher is directly actuated, so qpos always changes
    # even if no object is contacted.
    for _ in range(10):
        env.step(np.array([1.0, 1.0]))
    assert not np.allclose(env.snapshot()["qpos"], qpos_before)

    # Restoring must return the exact full dynamical state (qpos and qvel).
    env.restore(snap)
    np.testing.assert_allclose(env.snapshot()["qpos"], qpos_before, atol=1e-12)
    np.testing.assert_allclose(env.snapshot()["qvel"], snap["qvel"], atol=1e-12)
