"""Demo 1 -- autoregressive rollout drift, sparse vs dense vs no-op, side by side.

This is the animated form of the horizon-error table in ``RESULTS.md``: close the
loop on each world model and watch where the predicted scene ends up after ``H``
steps. The point that a static table understates is *how* dense fails -- it
perturbs every object every step, so the boxes that never moved slowly wander off
their true footprints, while the sparse model's gate copies them verbatim and
they stay locked in place.

Rollout mechanics are identical to ``experiments/rollout_horizon_error.py``
(same ``_step_model``, same exogenous-state handling: pusher, velocities and goal
come from the ground truth at every step so all models are driven identically),
just for a single trajectory instead of aggregated over all of them.

Panels: ground truth | sparse | dense, plus a growing per-horizon error chart
carrying the no-op reference -- the "how much does the world move anyway" floor
that dense ends up *above*.

Example
-------
python -m experiments.demos.demo_rollout_drift \
    --data data/transitions/scale_3obj_heldout.npz \
    --sparse-checkpoint models/checkpoints/sparse_3obj_s0.pt \
    --dense-checkpoint models/checkpoints/dense_3obj_s0.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.compare_phase4_models import load_dense_model, load_sparse_model
from experiments.demos.render2d import (
    GOAL_RADIUS,
    TABLE_HALF,
    ScenePainter,
    save_gif,
    truth_overlay_legend,
    write_scene_json,
)
from experiments.rollout_horizon_error import (
    MODEL_STYLE,
    _step_model,
    build_ground_truth_arrays,
    per_object_pose_l2,
)
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim

PANEL_MODELS = ("truth", "sparse", "dense", "no_op")
PANEL_TITLES = {
    "truth": "Ground truth (MuJoCo)",
    "sparse": "Sparse: gate + residual",
    "dense": "Dense: monolithic MLP",
    "no_op": "No-op: 'nothing moves'",
}
ERROR_MODELS = ("sparse", "dense", "no_op")
ERROR_LABELS = {
    "sparse": "sparse (gate + residual)",
    "dense": "dense (monolithic MLP)",
    "no_op": "no-op reference",
}


def rollout_single(
    model_name: str,
    model,
    sparse_cfg: dict | None,
    gt_states: torch.Tensor,
    gt_actions: torch.Tensor,
    start_index: int,
    horizon: int,
    layout: StateLayout,
    num_objects: int,
) -> np.ndarray:
    """Roll one model forward from ``start_index`` -> poses ``(horizon + 1, N, 3)``.

    Frame 0 is the shared ground-truth start pose, so every panel begins identical
    and the divergence the reader sees is entirely accumulated by the model.
    """
    pose_slice = layout.object_pose_slice
    pred_pose = gt_states[start_index, pose_slice].reshape(1, num_objects, POSE_DIM).clone()
    trajectory = [pred_pose[0].clone()]

    with torch.no_grad():
        for step in range(horizon):
            hybrid = gt_states[start_index + step].clone().unsqueeze(0)
            hybrid[:, pose_slice] = pred_pose.reshape(1, -1)
            action = gt_actions[start_index + step].unsqueeze(0)
            pred_pose = _step_model(
                model_name, model, sparse_cfg, hybrid, action, pred_pose, layout, num_objects
            )
            trajectory.append(pred_pose[0].clone())

    return torch.stack(trajectory).cpu().numpy()


def rollout_batch(
    model_name: str,
    model,
    sparse_cfg: dict | None,
    gt_states: torch.Tensor,
    gt_actions: torch.Tensor,
    start_indices: torch.Tensor,
    horizon: int,
    layout: StateLayout,
    num_objects: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll many launch points at once.

    Returns final-horizon mean per-object pose error and, per sample, the furthest
    ``|x|``/``|y|`` any predicted object reached during the rollout. The second
    quantity separates two very different failure modes that the error alone
    conflates: bounded jitter on every object versus one object drifting off the
    table entirely.
    """
    pose_slice = layout.object_pose_slice
    pose_all = gt_states[:, pose_slice].reshape(-1, num_objects, POSE_DIM)
    pred_pose = pose_all[start_indices].clone()
    max_extent = torch.zeros(start_indices.shape[0], device=gt_states.device)

    with torch.no_grad():
        for step in range(horizon):
            index = start_indices + step
            hybrid = gt_states[index].clone()
            hybrid[:, pose_slice] = pred_pose.reshape(pred_pose.shape[0], -1)
            pred_pose = _step_model(
                model_name, model, sparse_cfg, hybrid, gt_actions[index], pred_pose, layout, num_objects
            )
            max_extent = torch.maximum(max_extent, pred_pose[..., :2].abs().amax(dim=(1, 2)))

    pose_l2, _ = per_object_pose_l2(pred_pose, pose_all[start_indices + horizon])
    return pose_l2.mean(dim=1), max_extent


def pick_start_index(
    gt_states: torch.Tensor,
    gt_actions: torch.Tensor,
    lengths: list[int],
    horizon: int,
    layout: StateLayout,
    num_objects: int,
    scan_episodes: int | None,
    motion_percentile: float,
    max_candidates: int,
    sparse_model,
    sparse_cfg: dict,
    dense_model,
) -> tuple[int, dict]:
    """Choose a *representative* active rollout window, not an extreme one.

    Most launch points are near-static (that is the project's premise), so a random
    one would demonstrate nothing -- but the single highest-motion window is an
    outlier that flatters or damns whichever model happens to break on it. So:
    keep windows whose ground-truth motion is above ``motion_percentile``, then
    among those take the one whose sparse final-horizon error is the **median**.
    The reader sees a scene where something happens, at typical model error.
    """
    pose_all = gt_states[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
    indices: list[int] = []
    offset = 0
    for length in lengths if scan_episodes is None else lengths[:scan_episodes]:
        indices.extend(range(offset, offset + max(0, length - horizon)))
        offset += length + 1
    if not indices:
        raise SystemExit("No rollout window long enough for the requested horizon.")

    index_tensor = torch.tensor(indices, dtype=torch.long, device=gt_states.device)
    windows = torch.stack([pose_all[i : i + horizon + 1, :, :2] for i in indices])
    motion = torch.linalg.norm(windows[:, 1:] - windows[:, :-1], dim=-1).sum(dim=(1, 2))

    threshold = torch.quantile(motion, motion_percentile / 100.0)
    keep = torch.nonzero(motion >= threshold, as_tuple=False).squeeze(1)
    if keep.numel() > max_candidates:  # even subsample keeps the motion spread
        keep = keep[torch.linspace(0, keep.numel() - 1, max_candidates).long()]

    candidates = index_tensor[keep]
    sparse_error, sparse_extent = rollout_batch(
        "sparse", sparse_model, sparse_cfg, gt_states, gt_actions, candidates, horizon, layout, num_objects
    )
    dense_error, dense_extent = rollout_batch(
        "dense", dense_model, None, gt_states, gt_actions, candidates, horizon, layout, num_objects
    )

    order = torch.argsort(sparse_error)
    chosen = int(order[order.numel() // 2].item())
    return int(candidates[chosen].item()), {
        "num_candidates": int(candidates.numel()),
        "motion_percentile": motion_percentile,
        "motion_threshold_m": round(float(threshold), 4),
        "chosen_motion_m": round(float(motion[keep][chosen]), 4),
        # Distribution over the candidate pool, so a caption can say how typical the
        # rendered window is instead of implying it is the whole story.
        "candidate_error": {
            "sparse_median": round(float(sparse_error.median()), 4),
            "sparse_mean": round(float(sparse_error.mean()), 4),
            "dense_median": round(float(dense_error.median()), 4),
            "dense_mean": round(float(dense_error.mean()), 4),
        },
        "candidate_off_table_rate": {
            "sparse": round(float((sparse_extent > TABLE_HALF).float().mean()), 3),
            "dense": round(float((dense_extent > TABLE_HALF).float().mean()), 3),
        },
        "candidate_max_extent_median_m": {
            "sparse": round(float(sparse_extent.median()), 4),
            "dense": round(float(dense_extent.median()), 4),
        },
    }


def auto_view_half(trajectories: dict[str, np.ndarray], goal_xy: np.ndarray, pad: float = 0.045) -> float:
    """Crop tight enough that the scene fills the panel, wide enough to hide nothing.

    A fixed crop is dangerous here: a drifting prediction can leave the frame, which
    reads as "the model lost the object" when the honest picture is "it put the
    object *there*". So the view is fitted to every panel's actual extent -- if a
    model flings a box outward, the reader sees exactly how far.
    """
    extent = max(
        float(np.abs(np.concatenate([traj[..., :2].reshape(-1, 2) for traj in trajectories.values()])).max()),
        float(np.abs(goal_xy).max()) + GOAL_RADIUS,
    )
    return float(np.clip(extent + pad, 0.22, 0.60))


def build_scene_figure(num_objects: int, goal_xy: np.ndarray, target_object: int, view_half: float):
    """Four scene panels in a row -- the animated asset."""
    fig, axes = plt.subplots(1, len(PANEL_MODELS), figsize=(4.0 * len(PANEL_MODELS), 4.8))
    fig.patch.set_facecolor("white")
    painters = {}
    for ax, name in zip(axes, PANEL_MODELS):
        painters[name] = ScenePainter(
            ax,
            num_objects=num_objects,
            goal_xy=goal_xy,
            title=PANEL_TITLES[name],
            show_truth_overlay=(name != "truth"),
            target_object=target_object,
            view_half=view_half,
        )
    return fig, painters


def save_error_chart(errors: dict[str, np.ndarray], horizon: int, num_objects: int, path: Path) -> Path:
    """Static per-horizon error curve for this single rollout (companion to the GIF)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for name in ERROR_MODELS:
        ax.plot(
            np.arange(horizon + 1),
            errors[name],
            color=MODEL_STYLE[name]["color"],
            marker=MODEL_STYLE[name]["marker"],
            markersize=4,
            linewidth=2.0,
            label=ERROR_LABELS[name],
        )
    ax.axhline(errors["no_op"][-1], color=MODEL_STYLE["no_op"]["color"], linestyle=":", alpha=0.5)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Mean per-object pose L2")
    ax.set_title(
        f"Error compounding over a {horizon}-step rollout ({num_objects} objects)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    raw = np.load(args.data)
    dataset = {key: raw[key] for key in raw.files}
    num_objects = infer_num_objects_from_state_dim(int(dataset["s_t"].shape[1]))
    layout = StateLayout(num_objects=num_objects)

    gt_states_np, gt_actions_np, lengths = build_ground_truth_arrays(dataset)
    gt_states = torch.from_numpy(gt_states_np).to(device)
    gt_actions = torch.from_numpy(gt_actions_np).to(device)

    sparse_model, sparse_cfg = load_sparse_model(args.sparse_checkpoint, device)
    dense_model, _ = load_dense_model(args.dense_checkpoint, device)
    if int(sparse_cfg["num_objects"]) != num_objects:  # type: ignore[call-overload]
        raise SystemExit(
            f"Sparse checkpoint is for {sparse_cfg['num_objects']} objects but data has {num_objects}."
        )

    if args.start_index is not None:
        start_index, selection = args.start_index, {"mode": "explicit"}
    else:
        start_index, selection = pick_start_index(
            gt_states,
            gt_actions,
            lengths,
            args.horizon,
            layout,
            num_objects,
            args.scan_episodes,
            args.motion_percentile,
            args.max_candidates,
            sparse_model,
            sparse_cfg,
            dense_model,
        )
        selection["mode"] = "median-error window above motion percentile"
    print(f"start_index={start_index} selection={json.dumps(selection)}")

    pose_slice = layout.object_pose_slice
    truth = (
        gt_states[start_index : start_index + args.horizon + 1, pose_slice]
        .reshape(-1, num_objects, POSE_DIM)
        .cpu()
        .numpy()
    )
    pusher = gt_states[start_index : start_index + args.horizon + 1, 0:2].cpu().numpy()
    goal_xy = gt_states[start_index, layout.goal_slice].cpu().numpy()

    handles = {"sparse": sparse_model, "dense": dense_model, "no_op": None}
    trajectories = {"truth": truth}
    for name in ("sparse", "dense", "no_op"):
        trajectories[name] = rollout_single(
            name,
            handles[name],
            sparse_cfg if name == "sparse" else None,
            gt_states,
            gt_actions,
            start_index,
            args.horizon,
            layout,
            num_objects,
        )

    # Per-frame mean per-object full-pose L2 against the ground truth.
    truth_t = torch.from_numpy(truth)
    errors = {
        name: per_object_pose_l2(torch.from_numpy(trajectories[name]), truth_t)[0]
        .mean(dim=1)
        .numpy()
        for name in ("sparse", "dense", "no_op")
    }

    target_object = int(args.target_object)
    view_half = args.view_half if args.view_half is not None else auto_view_half(trajectories, goal_xy)
    fig, painters = build_scene_figure(num_objects, goal_xy, target_object, view_half)
    fig.suptitle(
        f"Closing the loop: {num_objects}-object push, {args.horizon}-step autoregressive rollout",
        fontsize=13,
        fontweight="bold",
    )
    truth_overlay_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))

    def update(frame: int):
        for name in PANEL_MODELS:
            painters[name].update(
                trajectories[name][frame],
                pusher_xy=pusher[frame],
                truth_poses=truth[frame] if name != "truth" else None,
                subtitle=(
                    f"step {frame}/{args.horizon}"
                    if name == "truth"
                    else f"step {frame}  |  error {errors[name][frame]:.3f}"
                ),
            )
        return []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rollout_drift_{num_objects}obj{args.name_suffix}"
    gif_path = save_gif(fig, update, args.horizon + 1, args.out_dir / f"{stem}.gif", fps=args.fps)
    chart_path = save_error_chart(
        errors, args.horizon, num_objects, args.out_dir / f"{stem}_error_curve.png"
    )

    json_path = write_scene_json(
        args.out_dir / f"{stem}.json",
        meta={
            "demo": "rollout_drift",
            "num_objects": num_objects,
            "horizon": args.horizon,
            "target_object": target_object,
            "goal_xy": goal_xy.round(5).tolist(),
            "data": str(args.data),
            "sparse_checkpoint": str(args.sparse_checkpoint),
            "dense_checkpoint": str(args.dense_checkpoint),
            "start_index": int(start_index),
            "selection": selection,
            "feature_mode": str(sparse_cfg.get("feature_mode", "global")),
        },
        panels={
            "truth": {"poses": truth, "pusher": pusher},
            "sparse": {"poses": trajectories["sparse"], "pusher": pusher, "error": errors["sparse"]},
            "dense": {"poses": trajectories["dense"], "pusher": pusher, "error": errors["dense"]},
            "no_op": {"poses": trajectories["no_op"], "pusher": pusher, "error": errors["no_op"]},
        },
    )

    summary = {
        "gif": str(gif_path),
        "error_curve": str(chart_path),
        "json": str(json_path),
        "start_index": int(start_index),
        "final_error": {name: round(float(errors[name][-1]), 4) for name in errors},
        "dense_over_no_op": round(float(errors["dense"][-1] / max(errors["no_op"][-1], 1e-9)), 2),
        "dense_over_sparse": round(float(errors["dense"][-1] / max(errors["sparse"][-1], 1e-9)), 2),
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animated rollout-drift comparison (blog demo).")
    parser.add_argument("--data", type=Path, default=Path("data/transitions/scale_3obj_heldout.npz"))
    parser.add_argument("--sparse-checkpoint", type=Path, default=Path("models/checkpoints/sparse_3obj_s0.pt"))
    parser.add_argument("--dense-checkpoint", type=Path, default=Path("models/checkpoints/dense_3obj_s0.pt"))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=None, help="Explicit launch row; default auto-picks a representative active window.")
    parser.add_argument("--scan-episodes", type=int, default=40, help="Episodes to scan when auto-picking (None = all).")
    parser.add_argument("--motion-percentile", type=float, default=90.0, help="Keep only windows this active before picking the median-error one.")
    parser.add_argument("--max-candidates", type=int, default=400, help="Cap on candidate windows rolled out during selection.")
    parser.add_argument("--target-object", type=int, default=0, help="Object outlined as the push target.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--view-half", type=float, default=None, help="Half-width of the drawn region; default fits every panel.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/runs/demos"))
    parser.add_argument("--name-suffix", type=str, default="", help="Appended to output filenames so variants coexist.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
