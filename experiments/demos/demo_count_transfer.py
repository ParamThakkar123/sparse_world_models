"""Demo 4 -- one set of weights, three scene sizes, zero retraining.

The sharpest object-centric claim in ``RESULTS.md``: because the sparse model is
a per-object gate + residual head with weights shared across objects, and the
count-invariant featurizer gives a fixed input width for any object count, a
model trained on ``N``-object scenes runs unchanged on ``M``-object scenes. The
dense monolith cannot even be *executed* off-count -- its input and output layers
are sized to one specific count, so an off-diagonal evaluation raises a
``RuntimeError``.

That asymmetry is hard to feel from a transfer matrix. Here a *single* sparse
checkpoint steps 3-, 5- and 8-object scenes side by side, and the demo actually
*attempts* the dense model on each scene, printing the real exception text it
raises rather than asserting that it would fail.

Requires the count-invariant checkpoints (the ones behind the transfer matrix are
trained in-process and not saved, so train them once):

    for N in 3 5 8; do
      python -m experiments.train_sparse_model \\
        --train data/transitions/splits_${N}obj_s0/scale_${N}obj_s0_hard_train.npz \\
        --val   data/transitions/splits_${N}obj_s0/scale_${N}obj_s0_hard_val.npz \\
        --run-name sparse_invariant_${N}obj_s0 --feature-mode invariant \\
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed 0
    done

Example
-------
python -m experiments.demos.demo_count_transfer --train-count 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.compare_phase4_models import load_dense_model, load_sparse_model
from experiments.demos.render2d import ScenePainter, save_gif, write_scene_json
from experiments.rollout_horizon_error import (
    _step_model,
    build_ground_truth_arrays,
    per_object_pose_l2,
)
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim


def episode_starts(lengths: list[int], horizon: int) -> list[int]:
    """Row index of every episode long enough to roll ``horizon`` steps."""
    starts: list[int] = []
    offset = 0
    for length in lengths:
        if length > horizon:
            starts.append(offset)
        offset += length + 1
    if not starts:
        raise SystemExit(f"No episode longer than {horizon} steps.")
    return starts


def pick_episode_by_motion(
    starts: list[int],
    pose_all: torch.Tensor,
    horizon: int,
    percentile: float,
    skip: int,
) -> int:
    """Pick the episode whose ground-truth motion sits at ``percentile``.

    Panels here compare *different datasets*, so taking "the first long enough
    episode" in each would vary scene activity across panels and the reader would
    read that difference as a transfer effect. Selecting at the same motion
    percentile in every count makes the panels comparable by construction.
    """
    motions = torch.tensor(
        [
            float(
                torch.linalg.norm(
                    pose_all[start + 1 : start + horizon + 1, :, :2]
                    - pose_all[start : start + horizon, :, :2],
                    dim=-1,
                ).sum()
            )
            for start in starts
        ]
    )
    order = torch.argsort(motions)
    position = min(len(starts) - 1, int(round((percentile / 100.0) * (len(starts) - 1))) + skip)
    return starts[int(order[position].item())]


def transfer_error(
    model,
    config: dict,
    gt_states: torch.Tensor,
    gt_actions: torch.Tensor,
    starts: list[int],
    horizon: int,
    layout: StateLayout,
    num_objects: int,
    max_episodes: int,
) -> float:
    """Mean final-horizon error over many episodes, so the panel is not the claim."""
    index = torch.tensor(starts[:max_episodes], dtype=torch.long, device=gt_states.device)
    pose_slice = layout.object_pose_slice
    pose_all = gt_states[:, pose_slice].reshape(-1, num_objects, POSE_DIM)
    pred_pose = pose_all[index].clone()
    with torch.no_grad():
        for step in range(horizon):
            hybrid = gt_states[index + step].clone()
            hybrid[:, pose_slice] = pred_pose.reshape(pred_pose.shape[0], -1)
            pred_pose = _step_model(
                "sparse", model, config, hybrid, gt_actions[index + step], pred_pose, layout, num_objects
            )
    error = per_object_pose_l2(pred_pose, pose_all[index + horizon])[0].mean(dim=1)
    return float(error.mean())


def roll_scene(
    checkpoint: Path,
    data_path: Path,
    horizon: int,
    episode_skip: int,
    motion_percentile: float,
    max_episodes: int,
    device: torch.device,
) -> dict:
    """Roll the sparse checkpoint over one episode of a possibly different count."""
    raw = np.load(data_path)
    dataset = {key: raw[key] for key in raw.files}
    num_objects = infer_num_objects_from_state_dim(int(dataset["s_t"].shape[1]))
    layout = StateLayout(num_objects=num_objects)

    gt_states_np, gt_actions_np, lengths = build_ground_truth_arrays(dataset)
    gt_states = torch.from_numpy(gt_states_np).to(device)
    gt_actions = torch.from_numpy(gt_actions_np).to(device)
    pose_slice = layout.object_pose_slice
    pose_all = gt_states[:, pose_slice].reshape(-1, num_objects, POSE_DIM)

    starts = episode_starts(lengths, horizon)
    start = pick_episode_by_motion(starts, pose_all, horizon, motion_percentile, episode_skip)

    model, config = load_sparse_model(checkpoint, device)
    pred_pose = pose_all[start].unsqueeze(0).clone()
    predicted = [pred_pose[0].cpu().numpy().copy()]

    with torch.no_grad():
        for step in range(horizon):
            hybrid = gt_states[start + step].clone().unsqueeze(0)
            hybrid[:, pose_slice] = pred_pose.reshape(1, -1)
            pred_pose = _step_model(
                "sparse", model, config, hybrid, gt_actions[start + step].unsqueeze(0),
                pred_pose, layout, num_objects,
            )
            predicted.append(pred_pose[0].cpu().numpy().copy())

    truth = pose_all[start : start + horizon + 1].cpu().numpy()
    predicted_arr = np.stack(predicted)
    error = per_object_pose_l2(torch.from_numpy(predicted_arr), torch.from_numpy(truth))[0].mean(dim=1).numpy()

    return {
        "num_objects": num_objects,
        "poses": predicted_arr,
        "truth": truth,
        "pusher": gt_states[start : start + horizon + 1, 0:2].cpu().numpy(),
        "goal_xy": gt_states[start, layout.goal_slice].cpu().numpy(),
        "error": error,
        "mean_error_over_episodes": transfer_error(
            model, config, gt_states, gt_actions, starts, horizon, layout, num_objects, max_episodes
        ),
        "num_episodes_scored": min(len(starts), max_episodes),
        "feature_mode": str(config.get("feature_mode", "global")),
        "trained_on": int(config["num_objects"]),  # type: ignore[call-overload]
    }


def probe_dense(dense_checkpoint: Path, data_path: Path, device: torch.device) -> dict:
    """Actually run the off-count dense model and capture whatever it does.

    Attempting the call is the point: the failure is a real shape mismatch, not a
    claim. If a future dense variant *did* accept the input, this would report that
    honestly instead of asserting a failure that no longer happens.
    """
    raw = np.load(data_path)
    states = torch.from_numpy(raw["s_t"][:1].astype(np.float32)).to(device)
    actions = torch.from_numpy(raw["a_t"][:1].astype(np.float32)).to(device)
    try:
        model, _ = load_dense_model(dense_checkpoint, device)
        with torch.no_grad():
            model(states, actions)
        return {"ran": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - reporting the real failure is the demo
        text = str(exc).split("\n")[0]
        return {"ran": False, "error": f"{type(exc).__name__}: {text[:150]}"}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = args.sparse_checkpoint or Path(
        f"models/checkpoints/sparse_invariant_{args.train_count}obj_s0.pt"
    )
    if not checkpoint.exists():
        raise SystemExit(
            f"Missing {checkpoint}. Train the count-invariant checkpoints first "
            "(see this module's docstring)."
        )
    dense_checkpoint = args.dense_checkpoint or Path(
        f"models/checkpoints/dense_{args.train_count}obj_s0.pt"
    )

    scenes = []
    dense_probe = {}
    for count in args.test_counts:
        data_path = args.data_template.format(count=count)
        scene = roll_scene(
            checkpoint, Path(data_path), args.horizon, args.episode_skip,
            args.motion_percentile, args.max_episodes, device,
        )
        scenes.append(scene)
        dense_probe[str(count)] = probe_dense(dense_checkpoint, Path(data_path), device) | {
            "data": data_path
        }

    fig, axes = plt.subplots(1, len(scenes), figsize=(4.3 * len(scenes), 5.2))
    if len(scenes) == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")
    painters = []
    for ax, scene in zip(axes, scenes):
        in_distribution = scene["num_objects"] == args.train_count
        painters.append(
            ScenePainter(
                ax,
                num_objects=scene["num_objects"],
                goal_xy=scene["goal_xy"],
                title=f"{scene['num_objects']}-object scene"
                + ("  (trained here)" if in_distribution else "  (never seen)"),
                show_truth_overlay=True,
                target_object=0,
                view_half=0.30,
            )
        )
        probe = dense_probe[str(scene["num_objects"])]
        ax.text(
            0.5,
            -0.055,
            "dense: runs" if probe["ran"] else f"dense: {probe['error'][:52]}...",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#2f7d32" if probe["ran"] else "#b3261e",
            family="monospace",
        )

    fig.suptitle(
        f"One sparse checkpoint trained on {args.train_count} objects, run on "
        + ", ".join(f"{s['num_objects']}" for s in scenes)
        + " objects with no retraining",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Weights are shared across objects and the count-invariant featurizer has a fixed width, "
        "so the same file steps any scene size.\n"
        "The dense monolith is *attempted* on each scene below each panel -- its input and output "
        "layers are sized to one object count.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4a4a4a",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    def update(frame: int):
        for painter, scene in zip(painters, scenes):
            painter.update(
                scene["poses"][frame],
                pusher_xy=scene["pusher"][frame],
                truth_poses=scene["truth"][frame],
                subtitle=(
                    f"step {frame}/{args.horizon}  |  err {scene['error'][frame]:.3f}"
                    f"  |  {scene['num_episodes_scored']}-ep mean {scene['mean_error_over_episodes']:.3f}"
                ),
            )
        return []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"count_transfer_from{args.train_count}obj"
    gif_path = save_gif(fig, update, args.horizon + 1, args.out_dir / f"{stem}.gif", fps=args.fps)

    json_path = write_scene_json(
        args.out_dir / f"{stem}.json",
        meta={
            "demo": "count_transfer",
            "num_objects": max(s["num_objects"] for s in scenes),
            "train_count": args.train_count,
            "target_object": 0,
            "goal_xy": scenes[0]["goal_xy"].round(5).tolist(),
            "test_counts": list(args.test_counts),
            "sparse_checkpoint": str(checkpoint),
            "dense_checkpoint": str(dense_checkpoint),
            "feature_mode": scenes[0]["feature_mode"],
            "horizon": args.horizon,
            "dense_probe": dense_probe,
        },
        panels={
            f"{scene['num_objects']}obj": {
                "poses": scene["poses"],
                "truth": scene["truth"],
                "pusher": scene["pusher"],
                "error": scene["error"],
            }
            for scene in scenes
        },
    )

    summary = {
        "gif": str(gif_path),
        "json": str(json_path),
        "sparse_checkpoint": str(checkpoint),
        "shown_episode_final_error": {
            f"{scene['num_objects']}obj": round(float(scene["error"][-1]), 4) for scene in scenes
        },
        "mean_final_error_over_episodes": {
            f"{scene['num_objects']}obj": round(scene["mean_error_over_episodes"], 4) for scene in scenes
        },
        "dense_probe": dense_probe,
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-object-count transfer (blog demo).")
    parser.add_argument("--train-count", type=int, default=3, help="Object count the shown checkpoint was trained on.")
    parser.add_argument("--test-counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--sparse-checkpoint", type=Path, default=None)
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    parser.add_argument("--data-template", type=str, default="data/transitions/scale_{count}obj_heldout.npz")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--episode-skip", type=int, default=0, help="Step this far past the selected percentile.")
    parser.add_argument("--motion-percentile", type=float, default=90.0, help="Scene activity percentile shown in every panel.")
    parser.add_argument("--max-episodes", type=int, default=60, help="Episodes scored for the mean-error figure.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/runs/demos"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
