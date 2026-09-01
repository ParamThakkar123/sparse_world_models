"""Multi-step (autoregressive) rollout evaluation of the world models.

This is the *world-model* experiment: instead of scoring a single one-step
prediction, we close the loop. Starting from a ground-truth state we predict the
next object poses, write them back into the state, feed that reconstructed state
in again, and repeat for ``H`` steps. Prediction error is measured per object as
an L2 pose error at every horizon ``1..H`` and plotted against the horizon.

The hypothesis is that the sparse residual model's structural prior -- only the
objects the gate flags as "changed" receive a delta, everything else is copied
verbatim -- should *compound* far less error over a rollout than the dense
baseline, which perturbs every object's pose every step. The no-op baseline
(poses never change) is the reference for "how much does the world move over H
steps".

Rollout mechanics
-----------------
The models only predict planar object poses ``(x, y, theta)``. The remaining
state components needed to rebuild the network input -- pusher position, object
velocities and the goal -- are *exogenous* (control / static / unmodeled), so at
each step we take them from the ground-truth trajectory and overwrite only the
object-pose slice with the rolled-out poses. This isolates the compounding error
to the quantity the models actually predict, and it treats all three models
identically so the comparison is fair.

  * sparse : next_pose = pred_pose + gate . delta   (masked residual)
  * dense  : next_pose = f(hybrid_state, action)    (absolute pose)
  * no-op  : next_pose = pred_pose                  (unchanged)

Rollouts are launched from *every* timestep of every episode (sliding starts),
each rolling forward up to ``min(H, steps_remaining)``. Errors are aggregated per
horizon over all rollouts that reached that horizon; the contributing sample
count is reported per horizon since longer horizons draw on fewer starts.

Data note
---------
The checkpoints in this repo are trained on the *hard* subset, whose trajectory
continuity is broken by filtering, so rollouts need a trajectory-preserving
source. The reported numbers use the ``scale_{N}obj_heldout.npz`` sets: 250
fresh scripted episodes per count generated with seed 100, unseen at training
(which used seeds 0/1/2) and verified to share no state with any training set,
so no rollout start is an in-distribution training configuration. Run one
manifest per training seed (``rollout_manifest_heldout_s{0,1,2}.json``) to get
the mean +- std reported in the paper.

Passing the full ``scale_{N}obj_s{seed}.npz`` datasets still works, but those
are the superset the hard *training* subset was drawn from, so absolute error
levels would include in-distribution configurations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import load_dense_model, load_sparse_model
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim

MODELS = ("sparse", "dense", "no_op")
MODEL_STYLE = {
    "sparse": {"color": "#1b9e77", "marker": "o"},
    "dense": {"color": "#d95f02", "marker": "s"},
    "no_op": {"color": "#7570b3", "marker": "^"},
}


@dataclass
class RolloutSamples:
    """Flattened sliding-window rollout starts over all episodes.

    ``base + t0 + t`` indexes into the concatenated ground-truth arrays for the
    state at local time ``t`` of the sample's episode.
    """

    base: torch.Tensor  # (K,) global offset of each sample's episode start
    t0: torch.Tensor  # (K,) local start time within the episode
    max_h: torch.Tensor  # (K,) furthest horizon this sample can reach


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to ``[-pi, pi]`` (torch counterpart of the generator's helper)."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def reconstruct_episode_lengths(done: np.ndarray) -> list[int]:
    """Return the transition count of each episode delimited by ``done``."""
    done_indices = np.flatnonzero(done)
    if done_indices.size == 0 or done_indices[-1] != len(done) - 1:
        raise ValueError("Dataset must contain complete episodes ending with done=True.")
    lengths: list[int] = []
    start = 0
    for end in done_indices:
        lengths.append(int(end) + 1 - start)
        start = int(end) + 1
    return lengths


def build_ground_truth_arrays(
    dataset: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Reconstruct per-episode ground-truth full-state and action sequences.

    Within an episode row ``i``: ``s_t[i]`` is the state at time ``i`` and
    ``s_t1[i]`` the state at time ``i + 1``, so an episode of ``T`` transitions
    yields ``T + 1`` states. Returns concatenated ``(M, D)`` states, ``(M, A)``
    actions (the action taken *at* each state; terminal-state rows are zero and
    never read), and the per-episode transition counts.
    """
    s_t = dataset["s_t"].astype(np.float32)
    s_t1 = dataset["s_t1"].astype(np.float32)
    actions = dataset["a_t"].astype(np.float32)
    lengths = reconstruct_episode_lengths(dataset["done"])

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    start = 0
    action_dim = actions.shape[1]
    for length in lengths:
        stop = start + length
        # T states from s_t plus the final s_t1 gives the full T+1 state sequence.
        episode_states = np.concatenate([s_t[start:stop], s_t1[stop - 1 : stop]], axis=0)
        episode_actions = np.concatenate(
            [actions[start:stop], np.zeros((1, action_dim), dtype=np.float32)], axis=0
        )
        state_chunks.append(episode_states)
        action_chunks.append(episode_actions)
        start = stop

    gt_states = np.concatenate(state_chunks, axis=0)
    gt_actions = np.concatenate(action_chunks, axis=0)
    return gt_states, gt_actions, lengths


def enumerate_rollout_samples(lengths: list[int], max_horizon: int) -> RolloutSamples:
    """Every (episode, start-time) launch point, with its reachable horizon."""
    base_list: list[int] = []
    t0_list: list[int] = []
    max_h_list: list[int] = []
    offset = 0
    for length in lengths:
        for t0 in range(length):  # need >= 1 future step; t0 in 0..T-1
            base_list.append(offset)
            t0_list.append(t0)
            max_h_list.append(min(max_horizon, length - t0))
        offset += length + 1  # T+1 states per episode
    return RolloutSamples(
        base=torch.tensor(base_list, dtype=torch.long),
        t0=torch.tensor(t0_list, dtype=torch.long),
        max_h=torch.tensor(max_h_list, dtype=torch.long),
    )


def per_object_pose_l2(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-object errors: full-pose L2 (angle wrapped) and translation-only L2.

    Shapes ``(K, num_objects, 3)`` -> two ``(K, num_objects)`` tensors.
    """
    err_xy = pred_pose[..., :2] - gt_pose[..., :2]
    err_theta = wrap_angle(pred_pose[..., 2] - gt_pose[..., 2])
    xy_l2 = torch.linalg.norm(err_xy, dim=-1)
    pose_l2 = torch.sqrt((err_xy**2).sum(dim=-1) + err_theta**2)
    return pose_l2, xy_l2


def _step_model(
    model_name: str,
    model,
    sparse_cfg: dict | None,
    hybrid_state: torch.Tensor,
    action: torch.Tensor,
    pred_pose: torch.Tensor,
    layout: StateLayout,
    num_objects: int,
) -> torch.Tensor:
    """Advance predicted poses by one step for the given model on active samples."""
    if model_name == "no_op":
        return pred_pose
    if model_name == "dense":
        pred_next = model(hybrid_state, action)
        return pred_next.reshape(-1, num_objects, POSE_DIM)
    # sparse
    estimator = str(sparse_cfg.get("eval_estimator", resolve_gate_estimator(False, str(sparse_cfg["estimator"]))))
    temperature = float(sparse_cfg["temperature"])
    feature_mode = str(sparse_cfg.get("feature_mode", "global"))
    features = build_object_features_by_mode(hybrid_state, action, feature_mode)
    out = model(features, estimator=estimator, temperature=temperature, hard=True)
    return pred_pose + out.masked_delta


def rollout_model(
    model_name: str,
    model,
    sparse_cfg: dict | None,
    gt_states: torch.Tensor,
    gt_actions: torch.Tensor,
    samples: RolloutSamples,
    max_horizon: int,
    num_objects: int,
    layout: StateLayout,
    step_batch_size: int,
) -> dict[str, np.ndarray]:
    """Roll one model forward over all samples, accumulating per-horizon error.

    Returns arrays indexed by horizon (length ``max_horizon``):
    ``pose_l2`` / ``xy_l2`` (mean over objects and samples), ``pose_l2_per_object``
    (``max_horizon x num_objects``) and ``count`` (samples reaching the horizon).
    """
    pose_slice = layout.object_pose_slice
    gt_pose_all = gt_states[:, pose_slice].reshape(-1, num_objects, POSE_DIM)

    num_samples = samples.base.shape[0]
    pose_sum = torch.zeros(max_horizon, dtype=torch.float64)
    xy_sum = torch.zeros(max_horizon, dtype=torch.float64)
    pose_sum_obj = torch.zeros(max_horizon, num_objects, dtype=torch.float64)
    count = torch.zeros(max_horizon, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, num_samples, step_batch_size):
            stop = min(start + step_batch_size, num_samples)
            base = samples.base[start:stop]
            t0 = samples.t0[start:stop]
            max_h = samples.max_h[start:stop]

            # Initialize rolled-out poses at each sample's ground-truth start pose.
            pred_pose = gt_pose_all[base + t0].clone()

            for horizon in range(1, max_horizon + 1):
                active = max_h >= horizon
                if not bool(active.any()):
                    break
                active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                in_idx = base[active_idx] + t0[active_idx] + (horizon - 1)
                tgt_idx = base[active_idx] + t0[active_idx] + horizon

                hybrid = gt_states[in_idx].clone()
                active_pred = pred_pose[active_idx]
                hybrid[:, pose_slice] = active_pred.reshape(active_idx.shape[0], -1)
                action = gt_actions[in_idx]

                new_pose = _step_model(
                    model_name, model, sparse_cfg, hybrid, action, active_pred, layout, num_objects
                )
                pred_pose[active_idx] = new_pose

                pose_l2, xy_l2 = per_object_pose_l2(new_pose, gt_pose_all[tgt_idx])
                pose_sum[horizon - 1] += pose_l2.mean(dim=1).sum().double()
                xy_sum[horizon - 1] += xy_l2.mean(dim=1).sum().double()
                pose_sum_obj[horizon - 1] += pose_l2.sum(dim=0).double()
                count[horizon - 1] += active_idx.shape[0]

    safe_count = count.clamp(min=1).double()
    return {
        "horizon": np.arange(1, max_horizon + 1, dtype=np.int64),
        "pose_l2": (pose_sum / safe_count).numpy(),
        "xy_l2": (xy_sum / safe_count).numpy(),
        "pose_l2_per_object": (pose_sum_obj / safe_count.unsqueeze(1)).numpy(),
        "count": count.numpy(),
    }


def evaluate_object_count(
    data_path: Path,
    sparse_checkpoint: Path,
    dense_checkpoint: Path,
    max_horizon: int,
    device: torch.device,
    step_batch_size: int,
    max_episodes: int | None,
) -> dict:
    """Run the sparse/dense/no-op rollout comparison for one dataset."""
    raw = np.load(data_path)
    dataset = {key: raw[key] for key in raw.files}
    if max_episodes is not None:
        dataset = _truncate_episodes(dataset, max_episodes)

    num_objects = infer_num_objects_from_state_dim(int(dataset["s_t"].shape[1]))
    layout = StateLayout(num_objects=num_objects)

    gt_states_np, gt_actions_np, lengths = build_ground_truth_arrays(dataset)
    gt_states = torch.from_numpy(gt_states_np).to(device)
    gt_actions = torch.from_numpy(gt_actions_np).to(device)
    samples = enumerate_rollout_samples(lengths, max_horizon)
    samples = RolloutSamples(
        base=samples.base.to(device), t0=samples.t0.to(device), max_h=samples.max_h.to(device)
    )

    sparse_model, sparse_cfg = load_sparse_model(sparse_checkpoint, device)
    dense_model, _ = load_dense_model(dense_checkpoint, device)
    if int(sparse_cfg["num_objects"]) != num_objects:
        raise ValueError(
            f"Sparse checkpoint is for {sparse_cfg['num_objects']} objects but data has {num_objects}."
        )

    handles = {"sparse": sparse_model, "dense": dense_model, "no_op": None}
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name in MODELS:
        curves[name] = rollout_model(
            name,
            handles[name],
            sparse_cfg if name == "sparse" else None,
            gt_states,
            gt_actions,
            samples,
            max_horizon,
            num_objects,
            layout,
            step_batch_size,
        )

    return {
        "num_objects": num_objects,
        "num_episodes": len(lengths),
        "num_rollout_starts": int(samples.base.shape[0]),
        "curves": curves,
    }


def _truncate_episodes(dataset: dict[str, np.ndarray], max_episodes: int) -> dict[str, np.ndarray]:
    """Keep only the first ``max_episodes`` complete episodes (row-aligned keys)."""
    done_indices = np.flatnonzero(dataset["done"])
    if done_indices.size <= max_episodes:
        return dataset
    cutoff = int(done_indices[max_episodes - 1]) + 1
    return {key: value[:cutoff] for key, value in dataset.items()}


def write_curves_csv(results: list[dict], path: Path) -> None:
    lines = ["object_count,model,horizon,pose_l2,xy_l2,num_samples"]
    for result in results:
        for model_name in MODELS:
            curve = result["curves"][model_name]
            for i, horizon in enumerate(curve["horizon"]):
                lines.append(
                    f"{result['num_objects']},{model_name},{int(horizon)},"
                    f"{curve['pose_l2'][i]:.6f},{curve['xy_l2'][i]:.6f},{int(curve['count'][i])}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_object_csv(results: list[dict], path: Path) -> None:
    lines = ["object_count,model,horizon,object_index,pose_l2"]
    for result in results:
        for model_name in MODELS:
            curve = result["curves"][model_name]
            per_object = curve["pose_l2_per_object"]
            for i, horizon in enumerate(curve["horizon"]):
                for obj_idx in range(per_object.shape[1]):
                    lines.append(
                        f"{result['num_objects']},{model_name},{int(horizon)},{obj_idx},"
                        f"{per_object[i, obj_idx]:.6f}"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_horizon_curves(results: list[dict], output_dir: Path) -> list[str]:
    """One two-panel figure (full-pose L2 and translation L2) per object count."""
    paths: list[str] = []
    for result in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for metric, ax, title in (
            ("pose_l2", axes[0], "Full-pose L2 (x, y, wrapped theta)"),
            ("xy_l2", axes[1], "Translation-only L2 (x, y)"),
        ):
            for model_name in MODELS:
                curve = result["curves"][model_name]
                style = MODEL_STYLE[model_name]
                ax.plot(
                    curve["horizon"],
                    curve[metric],
                    label=model_name,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4,
                    linewidth=1.8,
                )
            ax.set_xlabel("Rollout horizon (steps)")
            ax.set_ylabel("Mean per-object L2 error")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(
            f"Autoregressive rollout error vs horizon | {result['num_objects']} objects "
            f"| {result['num_rollout_starts']} rollout starts"
        )
        output_path = output_dir / f"rollout_horizon_{result['num_objects']}obj.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        paths.append(str(output_path))
    return paths


def plot_combined(results: list[dict], output_dir: Path) -> str | None:
    """Overlay per-object-count curves (sparse vs dense vs no-op) in one figure."""
    if len(results) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    linestyles = ["-", "--", ":", "-."]
    for result_idx, result in enumerate(results):
        linestyle = linestyles[result_idx % len(linestyles)]
        for model_name in MODELS:
            curve = result["curves"][model_name]
            ax.plot(
                curve["horizon"],
                curve["pose_l2"],
                label=f"{model_name} ({result['num_objects']}obj)",
                color=MODEL_STYLE[model_name]["color"],
                linestyle=linestyle,
                linewidth=1.6,
            )
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Mean per-object full-pose L2 error")
    ax.set_title("Rollout error vs horizon across object counts")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=len(results))
    output_path = output_dir / "rollout_horizon_combined.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def resolve_runs(args: argparse.Namespace) -> list[dict]:
    """Build the list of (object-count) runs from a manifest or single-run flags."""
    if args.manifest is not None:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        runs = manifest["runs"]
        for run in runs:
            run["data"] = Path(run["data"])
            run["sparse_checkpoint"] = Path(run["sparse_checkpoint"])
            run["dense_checkpoint"] = Path(run["dense_checkpoint"])
        return runs
    if args.data is None or args.sparse_checkpoint is None or args.dense_checkpoint is None:
        raise SystemExit("Provide --manifest, or all of --data, --sparse-checkpoint, --dense-checkpoint.")
    return [
        {
            "data": args.data,
            "sparse_checkpoint": args.sparse_checkpoint,
            "dense_checkpoint": args.dense_checkpoint,
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-step rollout / horizon-error evaluation (world-model experiment)."
    )
    parser.add_argument("--data", type=Path, default=None, help="Trajectory-preserving .npz (has done).")
    parser.add_argument("--sparse-checkpoint", type=Path, default=None)
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON with {\"runs\": [{data, sparse_checkpoint, dense_checkpoint}, ...]} for multi-count sweeps.",
    )
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap episodes per dataset (speed).")
    parser.add_argument("--step-batch-size", type=int, default=4096)
    parser.add_argument("--run-name", type=str, default="rollout_horizon_error")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    runs = resolve_runs(args)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config(
        {
            "task": "rollout_horizon_error",
            "max_horizon": args.max_horizon,
            "max_episodes": args.max_episodes,
            "device": args.device,
            "runs": [
                {
                    "data": str(run["data"]),
                    "sparse_checkpoint": str(run["sparse_checkpoint"]),
                    "dense_checkpoint": str(run["dense_checkpoint"]),
                }
                for run in runs
            ],
        }
    )

    results: list[dict] = []
    for run in runs:
        result = evaluate_object_count(
            data_path=run["data"],
            sparse_checkpoint=run["sparse_checkpoint"],
            dense_checkpoint=run["dense_checkpoint"],
            max_horizon=args.max_horizon,
            device=device,
            step_batch_size=args.step_batch_size,
            max_episodes=args.max_episodes,
        )
        results.append(result)
        for horizon_idx in range(args.max_horizon):
            logger.log_metrics(
                horizon_idx + 1,
                object_count=result["num_objects"],
                sparse_pose_l2=float(result["curves"]["sparse"]["pose_l2"][horizon_idx]),
                dense_pose_l2=float(result["curves"]["dense"]["pose_l2"][horizon_idx]),
                no_op_pose_l2=float(result["curves"]["no_op"]["pose_l2"][horizon_idx]),
                num_samples=int(result["curves"]["sparse"]["count"][horizon_idx]),
            )

    results.sort(key=lambda result: result["num_objects"])
    curves_csv = output_dir / "rollout_curves.csv"
    per_object_csv = output_dir / "rollout_per_object.csv"
    write_curves_csv(results, curves_csv)
    write_per_object_csv(results, per_object_csv)
    figure_paths = plot_horizon_curves(results, output_dir)
    combined_path = plot_combined(results, output_dir)

    summary = {
        "curves_csv": str(curves_csv),
        "per_object_csv": str(per_object_csv),
        "figures": figure_paths,
        "combined_figure": combined_path,
        "object_counts": [result["num_objects"] for result in results],
        "final_horizon": args.max_horizon,
        "final_horizon_pose_l2": {
            str(result["num_objects"]): {
                model_name: float(result["curves"][model_name]["pose_l2"][-1]) for model_name in MODELS
            }
            for result in results
        },
        "sparse_beats_dense_at_final_horizon": {
            str(result["num_objects"]): bool(
                result["curves"]["sparse"]["pose_l2"][-1] < result["curves"]["dense"]["pose_l2"][-1]
            )
            for result in results
        },
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
