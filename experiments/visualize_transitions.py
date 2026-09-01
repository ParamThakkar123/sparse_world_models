from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from models import POSE_DIM, reshape_object_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sampled tabletop transition pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "changed", "topk"],
        default="changed",
        help="How to choose transitions for plotting.",
    )
    parser.add_argument(
        "--min-xy-delta",
        type=float,
        default=0.005,
        help="Threshold used in the printed quality summary for meaningful planar motion.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional path to save the computed quality summary as JSON.",
    )
    return parser.parse_args()


def extract_object_pose(state_batch: np.ndarray) -> np.ndarray:
    return reshape_object_pose(state_batch)


def compute_quality_summary(
    states: np.ndarray,
    next_states: np.ndarray,
    change_mask: np.ndarray,
    min_xy_delta: float,
) -> dict[str, object]:
    object_pose = extract_object_pose(states)
    next_object_pose = extract_object_pose(next_states)
    object_delta_xy = next_object_pose[:, :, :2] - object_pose[:, :, :2]
    object_delta_norm = np.linalg.norm(object_delta_xy, axis=2)

    pusher_xy = states[:, :2]
    nearest_object_dist = np.min(
        np.linalg.norm(object_pose[:, :, :2] - pusher_xy[:, None, :], axis=2),
        axis=1,
    )
    per_step_changed = change_mask.sum(axis=1)
    target_delta = object_delta_norm[:, 0]

    summary = {
        "num_transitions": int(states.shape[0]),
        "mean_changed_fraction": float(change_mask.mean()),
        "steps_with_any_change": float(np.mean(per_step_changed > 0)),
        "steps_with_exactly_one_changed_object": float(np.mean(per_step_changed == 1)),
        "steps_with_multiple_changed_objects": float(np.mean(per_step_changed >= 2)),
        "target_object_mean_xy_delta": float(target_delta.mean()),
        "target_object_fraction_above_threshold": float(np.mean(target_delta >= min_xy_delta)),
        "any_object_fraction_above_threshold": float(np.mean(np.max(object_delta_norm, axis=1) >= min_xy_delta)),
        "median_nearest_object_distance_from_pusher": float(np.median(nearest_object_dist)),
        "fraction_pusher_within_5cm_of_any_object": float(np.mean(nearest_object_dist <= 0.05)),
    }
    return summary


def choose_indices(
    object_pose: np.ndarray,
    next_object_pose: np.ndarray,
    change_mask: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    sample_mode: str,
) -> np.ndarray:
    if object_pose.shape[0] == 0:
        return np.array([], dtype=np.int64)

    object_delta_xy = next_object_pose[:, :, :2] - object_pose[:, :, :2]
    object_delta_norm = np.linalg.norm(object_delta_xy, axis=2)
    max_delta = np.max(object_delta_norm, axis=1)
    any_change = np.any(change_mask > 0, axis=1)

    if sample_mode == "random":
        pool = np.arange(object_pose.shape[0])
        count = min(num_samples, pool.size)
        return np.sort(rng.choice(pool, size=count, replace=False))

    if sample_mode == "changed":
        pool = np.flatnonzero(any_change)
        if pool.size == 0:
            pool = np.arange(object_pose.shape[0])
        count = min(num_samples, pool.size)
        return np.sort(rng.choice(pool, size=count, replace=False))

    ranked = np.argsort(max_delta)[::-1]
    count = min(num_samples, ranked.size)
    return np.sort(ranked[:count])


def add_legend(fig: plt.Figure) -> None:
    handles = [
        mlines.Line2D([], [], color="tab:red", marker="o", linestyle="None", markersize=7, label="before"),
        mlines.Line2D([], [], color="tab:red", marker="x", linestyle="None", markersize=8, label="after"),
        mlines.Line2D([], [], color="black", marker="s", linestyle="None", markersize=6, label="pusher"),
        mlines.Line2D([], [], color="goldenrod", marker="*", linestyle="None", markersize=10, label="goal"),
        mlines.Line2D([], [], color="0.35", linestyle="-", linewidth=2, label="changed object"),
        mlines.Line2D([], [], color="0.5", linestyle="--", linewidth=1.2, label="unchanged object"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.98))


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    states = data["s_t"]
    next_states = data["s_t1"]
    change_mask = data["object_change_mask"]

    summary = compute_quality_summary(states, next_states, change_mask, args.min_xy_delta)
    print(json.dumps(summary, indent=2))
    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    object_pose = extract_object_pose(states)
    next_object_pose = extract_object_pose(next_states)
    indices = choose_indices(
        object_pose=object_pose,
        next_object_pose=next_object_pose,
        change_mask=change_mask,
        num_samples=args.num_samples,
        rng=rng,
        sample_mode=args.sample_mode,
    )

    if indices.size == 0:
        raise ValueError("No transitions available for visualization.")

    object_pose = object_pose[indices]
    next_object_pose = next_object_pose[indices]
    pusher_xy = states[indices, :2]
    goal_xy = states[indices, -2:]
    num_objects = int(object_pose.shape[1])

    cols = min(3, indices.size)
    rows = int(np.ceil(indices.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5 * rows), squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(num_objects)]
    labels = [f"obj{idx}" for idx in range(num_objects)]

    for plot_idx, ax in enumerate(axes.flat):
        if plot_idx >= indices.size:
            ax.axis("off")
            continue

        original_idx = int(indices[plot_idx])
        sample_mask = change_mask[original_idx]
        for obj_idx, (color, label) in enumerate(zip(colors, labels)):
            before_xy = object_pose[plot_idx, obj_idx, :2]
            after_xy = next_object_pose[plot_idx, obj_idx, :2]
            changed = bool(sample_mask[obj_idx])

            ax.scatter(before_xy[0], before_xy[1], color=color, marker="o", s=80, edgecolors="white", linewidths=0.6, zorder=3)
            ax.scatter(after_xy[0], after_xy[1], color=color, marker="x", s=95, linewidths=2.0, zorder=4)
            ax.annotate(
                "",
                xy=(after_xy[0], after_xy[1]),
                xytext=(before_xy[0], before_xy[1]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 2.0 if changed else 1.0,
                    "linestyle": "-" if changed else "--",
                    "alpha": 0.95 if changed else 0.55,
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=2,
            )
            ax.text(before_xy[0] + 0.008, before_xy[1] + 0.008, label, color=color, fontsize=9)

        ax.scatter(pusher_xy[plot_idx, 0], pusher_xy[plot_idx, 1], color="black", marker="s", s=70, zorder=5)
        ax.scatter(goal_xy[plot_idx, 0], goal_xy[plot_idx, 1], color="goldenrod", marker="*", s=180, edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_xlim(-0.26, 0.26)
        ax.set_ylim(-0.26, 0.26)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"Transition {original_idx} | changed={int(sample_mask.sum())}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    add_legend(fig)
    fig.suptitle(
        f"Sampled Tabletop Transitions ({args.sample_mode}): before (o) vs after (x)",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
