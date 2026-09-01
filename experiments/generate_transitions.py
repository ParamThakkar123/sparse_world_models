from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import ExperimentLogger
from models import StateLayout
from models.envs import TabletopPushConfig, TabletopPushEnv
from models.policies import RandomPolicy, ScriptedPushPolicy


POSITION_EPS = 1e-3
YAW_EPS = 1e-2


def extract_planar_object_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    yaws = np.array([quat_to_yaw(quat) for quat in obs["object_poses"][:, 3:]], dtype=np.float64)
    return np.column_stack((obs["object_poses"][:, 0], obs["object_poses"][:, 1], yaws))


def flatten_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    planar_pose = extract_planar_object_state(obs)
    return np.concatenate(
        [
            obs["pusher_xy"].reshape(-1),
            planar_pose.reshape(-1),
            obs["object_velocities"].reshape(-1),
            obs["goal_xy"].reshape(-1),
        ]
    )


def compute_diff_labels(
    obs: dict[str, np.ndarray], next_obs: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    current_pose = extract_planar_object_state(obs)
    next_pose = extract_planar_object_state(next_obs)
    delta = next_pose - current_pose
    delta[:, 2] = wrap_angle(delta[:, 2])

    position_change = np.linalg.norm(delta[:, :2], axis=1) > POSITION_EPS
    yaw_change = np.abs(delta[:, 2]) > YAW_EPS
    changed_mask = np.logical_or(position_change, yaw_change).astype(np.float32)
    return changed_mask, delta.astype(np.float32)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def build_policy(name: str, seed: int | None):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "scripted":
        return ScriptedPushPolicy()
    raise ValueError(f"Unsupported policy '{name}'.")


def build_env(
    env_name: str,
    num_objects: int,
    max_steps: int,
    seed: int | None,
    config_overrides: dict[str, object],
):
    """Construct the requested simulator.

    Every environment exposes the same observation dictionary and the same
    snapshot/restore/relocate API, so everything downstream -- training, the ablations, the
    counterfactual machinery -- is env-agnostic and selected purely by this flag.

    The four domains span two axes deliberately. **Engine**: ``tabletop`` is MuJoCo,
    ``billiards`` is Box2D, ``clutter`` is Chipmunk2D, and ``planar`` is ours -- so a result
    that holds across them cannot be a shared-implementation artefact. **Post-contact
    motion**, which is what the momentum shortcut feeds on: ``billiards`` (near-elastic,
    very long), ``tabletop`` (impulsive, short slide), ``clutter`` (high-friction, chained),
    ``planar`` (quasi-static, stops immediately). All four are calibrated to a comparable
    changed-object fraction (~0.35) so cross-domain comparisons vary the physics rather than
    the sparsity of the prediction target.
    """
    if env_name == "tabletop":
        return TabletopPushEnv(
            TabletopPushConfig(
                num_objects=num_objects, max_steps=max_steps, seed=seed, **config_overrides
            )
        )
    if env_name == "planar":
        from models.envs.planar_push import PlanarPushConfig, PlanarPushEnv

        return PlanarPushEnv(
            PlanarPushConfig(
                num_objects=num_objects, max_steps=max_steps, seed=seed, **config_overrides
            )
        )
    if env_name == "billiards":
        from models.envs.box2d_billiards import Box2DBilliardsConfig, Box2DBilliardsEnv

        return Box2DBilliardsEnv(
            Box2DBilliardsConfig(
                num_objects=num_objects, max_steps=max_steps, seed=seed, **config_overrides
            )
        )
    if env_name == "clutter":
        from models.envs.pymunk_clutter import PymunkClutterConfig, PymunkClutterEnv

        return PymunkClutterEnv(
            PymunkClutterConfig(
                num_objects=num_objects, max_steps=max_steps, seed=seed, **config_overrides
            )
        )
    raise ValueError(
        f"Unsupported env '{env_name}'. Expected one of "
        "'tabletop', 'planar', 'billiards', 'clutter'."
    )


def generate_dataset(
    output_path: Path,
    policy_name: str,
    num_episodes: int,
    max_steps: int,
    seed: int | None,
    num_objects: int,
    object_bound: float | None = None,
    min_object_separation: float | None = None,
    env_name: str = "tabletop",
) -> dict[str, float]:
    config_overrides: dict[str, object] = {}
    if object_bound is not None:
        config_overrides["object_bounds"] = (-abs(object_bound), abs(object_bound))
    if min_object_separation is not None:
        config_overrides["min_object_separation"] = min_object_separation
    env = build_env(env_name, num_objects, max_steps, seed, config_overrides)
    policy = build_policy(policy_name, seed)
    layout = StateLayout(num_objects=num_objects)

    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    object_change_masks = []
    object_delta_vectors = []
    episode_lengths = []
    episode_returns = []
    successes = 0

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        steps = 0

        while True:
            state = flatten_state(obs)
            if state.shape[0] != layout.state_dim:
                raise ValueError(f"State dim {state.shape[0]} does not match expected layout dim {layout.state_dim}.")
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)

            change_mask, delta_vector = compute_diff_labels(obs, next_obs)

            states.append(state)
            actions.append(action.astype(np.float32))
            next_states.append(flatten_state(next_obs))
            rewards.append(reward)
            dones.append(done)
            object_change_masks.append(change_mask)
            object_delta_vectors.append(delta_vector)

            episode_return += reward
            steps += 1
            obs = next_obs

            if done:
                successes += int(info["success"])
                episode_lengths.append(steps)
                episode_returns.append(episode_return)
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        s_t=np.asarray(states, dtype=np.float32),
        a_t=np.asarray(actions, dtype=np.float32),
        s_t1=np.asarray(next_states, dtype=np.float32),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        next_state=np.asarray(next_states, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=bool),
        object_change_mask=np.asarray(object_change_masks, dtype=np.float32),
        object_delta=np.asarray(object_delta_vectors, dtype=np.float32),
        changed_mask=np.asarray(object_change_masks, dtype=np.float32),
        delta_vector=np.asarray(object_delta_vectors, dtype=np.float32),
    )

    return {
        "num_episodes": float(num_episodes),
        "num_transitions": float(len(states)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_episode_return": float(np.mean(episode_returns)),
        "success_rate": float(successes / num_episodes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate action-transition pairs from the tabletop MuJoCo env.")
    parser.add_argument("--policy", choices=["random", "scripted"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument(
        "--object-bound",
        type=float,
        default=None,
        help="Symmetric half-extent for object placement (overrides env default of 0.18).",
    )
    parser.add_argument(
        "--min-object-separation",
        type=float,
        default=None,
        help="Minimum center-to-center object separation at reset (overrides env default of 0.12).",
    )
    parser.add_argument(
        "--env",
        choices=["tabletop", "planar", "billiards", "clutter"],
        default="tabletop",
        help=(
            "Simulator. 'tabletop' is MuJoCo, 'planar' our dependency-free 2D domain, "
            "'billiards' Box2D (near-elastic), 'clutter' Chipmunk2D (high-friction). "
            "See build_env for why the suite spans four engines and four contact regimes."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/transitions/tabletop_push_random.npz"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = ExperimentLogger(run_name=args.run_name or f"{args.policy}_transition_gen")
    logger.log_config(
        {
            "task": "tabletop_pushing",
            "simulator": args.env,
            "num_objects": args.num_objects,
            "policy": args.policy,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "object_bound": args.object_bound,
            "min_object_separation": args.min_object_separation,
            "output": str(args.output),
        }
    )

    summary = generate_dataset(
        output_path=args.output,
        policy_name=args.policy,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        num_objects=args.num_objects,
        object_bound=args.object_bound,
        min_object_separation=args.min_object_separation,
        env_name=args.env,
    )
    logger.log_summary(summary)
    logger.log_metrics(
        0,
        num_transitions=summary["num_transitions"],
        avg_episode_length=summary["avg_episode_length"],
        avg_episode_return=summary["avg_episode_return"],
        success_rate=summary["success_rate"],
    )

    print(json.dumps({"output": str(args.output), **summary}, indent=2))


if __name__ == "__main__":
    main()
