"""Demo 3 -- MPC episodes with each world model as the forward simulator.

``experiments/planning_mpc.py`` reports aggregate success rates; it does not keep
per-step trajectories, so nothing there can be animated. This module re-runs the
*same* planner (same ``CEMConfig``, ``cem_plan``, ``model_evaluator``,
``oracle_evaluator``, same per-episode env seeds) and records what happened,
without touching the experiment script that produced the paper numbers.

What the animation adds over the table is the *imagined* rollout: at every real
step CEM commits to an action sequence, and we replay what the model believed
that sequence would do to the target object. Drawn as a faint line from the
target, it shows the dense monolith planning against a fantasy -- its imagined
target wanders even when the real one is untouched -- while the contact-aware
sparse model imagines motion only when the pusher is actually placed to push.

Episode selection is stated rather than hidden: by default the animated episode
is the first one the sparse model *solves*, which is explicitly a minority
outcome (RESULTS.md: 0.23 +- 0.06 success). The accompanying bar chart shows
every episode run, solved or not, so the GIF cannot be mistaken for the average.

Example
-------
python -m experiments.demos.demo_planning --num-episodes 8 --oracle-episodes 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.demos.render2d import (
    ScenePainter,
    save_gif,
    write_scene_json,
)
from experiments.generate_transitions import flatten_state
from experiments.planning_mpc import (
    CEMConfig,
    build_forward,
    cem_plan,
    imagined_target_trajectory,
    model_evaluator,
    oracle_evaluator,
)
from models import StateLayout
from models.envs import TabletopPushConfig, TabletopPushEnv

CONDITION_TITLES = {
    "sparse": "Sparse (contact features)",
    "dense": "Dense (monolithic MLP)",
    "oracle": "Oracle (true simulator)",
    "scripted": "Scripted expert",
}
CONDITION_COLORS = {
    "sparse": "#1b9e77",
    "dense": "#d95f02",
    "oracle": "#7570b3",
    "scripted": "#666666",
}


def make_env(num_objects: int, max_steps: int, seed: int) -> TabletopPushEnv:
    return TabletopPushEnv(
        TabletopPushConfig(num_objects=num_objects, max_steps=max_steps, seed=seed)
    )


def record_model_episode(
    env: TabletopPushEnv,
    forward,
    layout: StateLayout,
    config: CEMConfig,
    max_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict:
    """``run_mpc_episode`` with per-step recording, including the imagined rollout."""
    obs = env.reset()
    target_object = env.config.target_object
    goal_xy = torch.tensor(env.config.goal_xy, dtype=torch.float32, device=device)
    mean = torch.zeros(config.horizon, 2, device=device)

    poses = [env.get_state()["object_pose"].copy()]
    pushers = [obs["pusher_xy"].copy()]
    imagined: list[np.ndarray] = []
    distances = [float(np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy)))]

    success, steps, total_plan_time = False, 0, 0.0
    for step in range(max_steps):
        state = torch.from_numpy(flatten_state(obs).astype(np.float32)).to(device)
        evaluate = model_evaluator(
            forward, state, goal_xy, layout, target_object,
            env.config.action_scale, env.config.pusher_bounds, config,
        )
        start = time.perf_counter()
        best_sequence, _ = cem_plan(evaluate, config.horizon, device, config, mean, generator)
        total_plan_time += time.perf_counter() - start

        # Replay the *chosen* plan through the model to recover what it believed
        # would happen -- this is the model's imagination, not the real outcome.
        with torch.no_grad():
            imagined_xy, _ = imagined_target_trajectory(
                forward, state, best_sequence.unsqueeze(0), layout, target_object,
                env.config.action_scale, env.config.pusher_bounds,
            )
        # Anchor the line at the target's *real* current position, so the reader
        # sees the imagined future as this object's future. Without the anchor a
        # hallucinating model's line floats in empty space, disconnected from the
        # box it is supposedly about.
        imagined.append(
            np.concatenate(
                [obs["object_poses"][target_object, :2][None, :], imagined_xy[0].cpu().numpy()]
            )
        )

        obs, _, terminated, info = env.step(best_sequence[0].cpu().numpy())
        steps = step + 1
        poses.append(env.get_state()["object_pose"].copy())
        pushers.append(obs["pusher_xy"].copy())
        distances.append(float(info["target_distance"]))

        mean = torch.cat([best_sequence[1:], torch.zeros(1, 2, device=device)], dim=0)
        if info["success"]:
            success = True
            break
        if terminated:
            break

    return _pack(poses, pushers, distances, imagined, success, steps, total_plan_time)


def record_oracle_episode(
    env: TabletopPushEnv,
    config: CEMConfig,
    max_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict:
    """``run_oracle_mpc_episode`` with recording. No imagined line: the oracle's
    'imagination' is the simulator itself, so plotting it would just retrace truth."""
    obs = env.reset()
    target_object = env.config.target_object
    goal_xy = torch.tensor(env.config.goal_xy, dtype=torch.float32, device=device)
    mean = torch.zeros(config.horizon, 2, device=device)

    sim = TabletopPushEnv(env.config)
    sim.reset()

    poses = [env.get_state()["object_pose"].copy()]
    pushers = [obs["pusher_xy"].copy()]
    distances = [float(np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy)))]

    success, steps, total_plan_time = False, 0, 0.0
    for step in range(max_steps):
        sim.restore(env.snapshot())
        evaluate = oracle_evaluator(sim, env.snapshot(), goal_xy, target_object, config)
        start = time.perf_counter()
        best_sequence, _ = cem_plan(evaluate, config.horizon, device, config, mean, generator)
        total_plan_time += time.perf_counter() - start

        obs, _, terminated, info = env.step(best_sequence[0].cpu().numpy())
        steps = step + 1
        poses.append(env.get_state()["object_pose"].copy())
        pushers.append(obs["pusher_xy"].copy())
        distances.append(float(info["target_distance"]))

        mean = torch.cat([best_sequence[1:], torch.zeros(1, 2, device=device)], dim=0)
        if info["success"]:
            success = True
            break
        if terminated:
            break

    return _pack(poses, pushers, distances, [], success, steps, total_plan_time)


def _pack(poses, pushers, distances, imagined, success, steps, total_plan_time) -> dict:
    return {
        "poses": np.stack(poses),  # (T, N, 3) planar (x, y, yaw)
        "pusher": np.stack(pushers),
        "distance": np.asarray(distances),
        "imagined": imagined,
        "success": bool(success),
        "steps": int(steps),
        "plan_ms_per_step": (total_plan_time * 1000.0 / max(1, steps)),
    }


def pad_to_length(episode: dict, length: int) -> dict:
    """Freeze a finished episode on its last frame so panels stay in lock-step.

    Conditions end at different times (the oracle solves in ~11 steps, a failing
    model runs the full budget). Holding the solved scene is the honest way to
    show them side by side -- the alternative, looping, would imply continued
    activity that never happened.
    """
    pad = length - episode["poses"].shape[0]
    if pad <= 0:
        return episode
    padded = dict(episode)
    padded["poses"] = np.concatenate([episode["poses"], np.repeat(episode["poses"][-1:], pad, axis=0)])
    padded["pusher"] = np.concatenate([episode["pusher"], np.repeat(episode["pusher"][-1:], pad, axis=0)])
    padded["distance"] = np.concatenate([episode["distance"], np.repeat(episode["distance"][-1:], pad)])
    return padded


def save_episode_chart(records: dict[str, list[dict]], success_radius: float, path: Path) -> Path:
    """Final target-goal distance for *every* episode run, per condition."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    conditions = list(records)
    width = 0.8 / max(1, len(conditions))
    for offset, condition in enumerate(conditions):
        episodes = records[condition]
        positions = np.arange(len(episodes)) + offset * width - 0.4 + width / 2
        finals = [float(ep["distance"][-1]) for ep in episodes]
        solved = [ep["success"] for ep in episodes]
        ax.bar(
            positions, finals, width=width, color=CONDITION_COLORS[condition],
            edgecolor="white", linewidth=0.6,
            label=f"{CONDITION_TITLES[condition]} - {sum(solved)}/{len(episodes)} solved",
        )
        # A condition that lands *on* the goal draws a zero-height bar, which reads
        # as missing data rather than as a perfect result. Mark every bar top.
        ax.scatter(
            positions, finals, s=14, color=CONDITION_COLORS[condition],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
    ax.axhline(success_radius, color="#c0392b", linestyle="--", linewidth=1.4)
    ax.text(
        ax.get_xlim()[1], success_radius, f"  success radius {success_radius:g} m",
        color="#c0392b", fontsize=8.5, va="center",
    )
    ax.set_xlabel("Episode (identical object configuration across conditions)")
    ax.set_ylabel("Final target-goal distance (m)")
    ax.set_title("Every planning episode run for this demo, solved or not", fontsize=11.5, fontweight="bold")
    ax.set_xticks(np.arange(max(len(v) for v in records.values())))
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    layout = StateLayout(num_objects=args.num_objects)
    cem_config = CEMConfig(
        horizon=args.horizon, num_samples=args.num_samples, cem_iters=args.cem_iters
    )
    checkpoints = {"sparse": args.sparse_checkpoint, "dense": args.dense_checkpoint}

    conditions = list(args.conditions)
    records: dict[str, list[dict]] = {condition: [] for condition in conditions}
    for condition in conditions:
        forward = (
            build_forward(condition, checkpoints, args.num_objects, layout, device)
            if condition in ("sparse", "dense")
            else None
        )
        episode_count = args.oracle_episodes if condition == "oracle" else args.num_episodes
        for i in range(episode_count):
            env = make_env(args.num_objects, args.max_steps, args.base_seed + i)
            generator = torch.Generator(device=device)
            generator.manual_seed(args.plan_seed + i)
            if condition == "oracle":
                episode = record_oracle_episode(env, cem_config, args.max_steps, generator, device)
            else:
                episode = record_model_episode(
                    env, forward, layout, cem_config, args.max_steps, generator, device
                )
            records[condition].append(episode)
            print(
                f"[{condition}] ep {i + 1}/{episode_count} success={episode['success']} "
                f"steps={episode['steps']} final={episode['distance'][-1]:.4f}"
            )

    # Animate an episode every condition ran, preferring one the sparse model solves.
    shared = min(len(records[c]) for c in conditions)
    solved_by_sparse = [
        i for i in range(shared) if "sparse" in records and records["sparse"][i]["success"]
    ]
    if args.episode is not None:
        episode_index, why = args.episode, "explicitly requested"
    elif solved_by_sparse:
        episode_index, why = solved_by_sparse[0], "first episode the sparse model solves"
    else:
        episode_index, why = 0, "no sparse success in this run; showing the first episode"
    print(f"animating episode {episode_index} ({why})")

    chosen = {c: records[c][episode_index] for c in conditions}
    length = max(ep["poses"].shape[0] for ep in chosen.values())
    chosen = {c: pad_to_length(ep, length) for c, ep in chosen.items()}

    goal_xy = np.asarray(TabletopPushConfig(num_objects=args.num_objects).goal_xy)
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.3 * len(conditions), 5.0))
    if len(conditions) == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")
    painters = {
        condition: ScenePainter(
            ax, num_objects=args.num_objects, goal_xy=goal_xy,
            title=CONDITION_TITLES[condition], target_object=0, view_half=0.30,
        )
        for ax, condition in zip(axes, conditions)
    }
    imagined_lines = {
        condition: ax.plot([], [], color="#444444", linewidth=1.3, linestyle=":", alpha=0.85, zorder=6)[0]
        for ax, condition in zip(axes, conditions)
    }

    fig.suptitle(
        f"Planning with each world model as the forward simulator (episode {episode_index})",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.045,
        "Dotted line: where the planner's chosen action sequence *imagines* the target will go.  "
        f"Animated episode is the {why}.",
        ha="center", fontsize=8.5, color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    def update(frame: int):
        for condition in conditions:
            episode = chosen[condition]
            painters[condition].update(
                episode["poses"][frame],
                pusher_xy=episode["pusher"][frame],
                subtitle=(
                    f"step {min(frame, episode['steps'])}  |  "
                    f"target-goal {episode['distance'][frame]:.3f} m"
                    + ("  |  SOLVED" if episode["success"] and frame >= episode["steps"] else "")
                ),
            )
            plan = episode["imagined"]
            if plan and frame < len(plan):
                imagined_lines[condition].set_data(plan[frame][:, 0], plan[frame][:, 1])
        return []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"planning_{args.num_objects}obj"
    gif_path = save_gif(fig, update, length, args.out_dir / f"{stem}.gif", fps=args.fps)
    chart_path = save_episode_chart(records, args.success_radius, args.out_dir / f"{stem}_episodes.png")

    json_path = write_scene_json(
        args.out_dir / f"{stem}.json",
        meta={
            "demo": "planning",
            "num_objects": args.num_objects,
            "goal_xy": goal_xy.round(5).tolist(),
            "target_object": 0,
            "episode_index": int(episode_index),
            "episode_choice_reason": why,
            "conditions": conditions,
            "sparse_checkpoint": str(args.sparse_checkpoint),
            "dense_checkpoint": str(args.dense_checkpoint),
            "cem": {"horizon": args.horizon, "num_samples": args.num_samples, "iters": args.cem_iters},
        },
        panels={
            condition: {
                "poses": chosen[condition]["poses"],
                "pusher": chosen[condition]["pusher"],
                "distance": chosen[condition]["distance"],
            }
            for condition in conditions
        },
    )

    summary = {
        "gif": str(gif_path),
        "episode_chart": str(chart_path),
        "json": str(json_path),
        "animated_episode": int(episode_index),
        "episode_choice_reason": why,
        "note": (
            "Demo-scale run, not the paper numbers. RESULTS.md reports 3-seed means over "
            "20 episodes: sparse 0.23 +- 0.06, dense 0.00, oracle 1.00, scripted 0.95."
        ),
        "per_condition": {
            condition: {
                "episodes": len(episodes),
                "success_rate": round(float(np.mean([ep["success"] for ep in episodes])), 3),
                "mean_final_distance": round(float(np.mean([ep["distance"][-1] for ep in episodes])), 4),
                "mean_plan_ms_per_step": round(float(np.mean([ep["plan_ms_per_step"] for ep in episodes])), 1),
            }
            for condition, episodes in records.items()
        },
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animated MPC planning comparison (blog demo).")
    parser.add_argument("--conditions", nargs="+", default=["sparse", "dense", "oracle"])
    parser.add_argument("--sparse-checkpoint", type=Path, default=Path("models/checkpoints/sparse_contact_3obj_v1.pt"))
    parser.add_argument("--dense-checkpoint", type=Path, default=Path("models/checkpoints/dense_mixed_3obj_v1.pt"))
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--oracle-episodes", type=int, default=3, help="Oracle plans ~30x slower; kept small by default.")
    parser.add_argument("--episode", type=int, default=None, help="Force which episode is animated.")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1000, help="Env seed for episode 0 -- planning_mpc's default, so episodes match the paper run.")
    parser.add_argument("--plan-seed", type=int, default=0)
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/runs/demos"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
