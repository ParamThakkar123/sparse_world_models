"""Oracle-gate diagnostic: is the bottleneck detection or regression?

The sparse model's changed-object L2 sits close to the no-op reference, which is
ambiguous: it could mean the *gate* misses changed objects (a detection failure, whose
masked delta collapses to zero and predicts "no change" exactly like no-op), or it could
mean the *delta head* regresses poorly even when it fires (a regression failure). This
diagnostic separates the two by feeding the **ground-truth changed mask** to the delta
head instead of the predicted gate:

  * predicted-gate : next = current + (gate >= 0.5) . delta   (the deployed model)
  * oracle-gate    : next = current + target_mask . delta     (perfect detection)
  * no-op          : next = current

Comparing changed-object L2 across the three, restricted to the objects that truly
changed:

  * oracle-gate L2                      = the pure delta-regression error (a floor).
  * predicted-gate L2 - oracle-gate L2  = the error the gate's misses add on top.

If oracle-gate is far below predicted-gate, detection is the bottleneck; if they are
close, the delta regression is the limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import compute_pose_metrics, load_dataset, load_sparse_model
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-gate vs predicted-gate diagnostic.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--split-template",
        type=str,
        default=None,
        help=(
            "Where to read splits. Point at the episode-disjoint clean splits for new work: "
            "data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz"
        ),
    )
    parser.add_argument("--run-name", type=str, default="oracle_gate_diagnostic")
    parser.add_argument("--sparse-template", type=str, default="models/checkpoints/sparse_{n}obj_s{seed}.pt")
    return parser.parse_args()


def split_path(count: int, seed: int, split: str, template: str | None = None) -> Path:
    """Locate a split file.

    The default points at the ORIGINAL directories, which leak (25% of source episodes have
    chunks in both train and test -- see experiments/build_clean_splits.py). Pass the clean
    template for any new measurement.
    """
    if template is None:
        template = "data/transitions/splits_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz"
    return Path(template.format(n=count, seed=seed, split=split))


def sparse_deltas(model, config, dataset, device, batch_size):
    """Return the raw per-object delta and predicted gate mask for all samples."""
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    temperature = float(config["temperature"])
    feature_mode = str(config.get("feature_mode", "global"))

    delta_chunks: list[np.ndarray] = []
    gate_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(start + batch_size, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            out = model(features, estimator=estimator, temperature=temperature, hard=True)
            delta_chunks.append(out.delta.cpu().numpy())
            gate_chunks.append((out.gate.probs >= 0.5).float().cpu().numpy())
    return np.concatenate(delta_chunks, axis=0), np.concatenate(gate_chunks, axis=0)


def evaluate_count(count: int, args: argparse.Namespace, device: torch.device) -> dict:
    dataset = load_dataset(split_path(count, args.seed, "test", args.split_template))
    model, config = load_sparse_model(Path(args.sparse_template.format(n=count, seed=args.seed)), device)
    delta, pred_gate = sparse_deltas(model, config, dataset, device, args.batch_size)

    current_pose = dataset["current_pose"]
    next_pose = dataset["next_pose"]
    target_mask = dataset["target_mask"]

    predicted_pred = current_pose + pred_gate[:, :, None] * delta
    oracle_pred = current_pose + target_mask[:, :, None] * delta
    noop_pred = current_pose.copy()

    predicted_metrics = compute_pose_metrics(predicted_pred, current_pose, next_pose, target_mask)
    oracle_metrics = compute_pose_metrics(oracle_pred, current_pose, next_pose, target_mask)
    noop_metrics = compute_pose_metrics(noop_pred, current_pose, next_pose, target_mask)

    detection_gap = predicted_metrics["changed_object_l2"] - oracle_metrics["changed_object_l2"]
    return {
        "object_count": count,
        "predicted_gate_changed_l2": predicted_metrics["changed_object_l2"],
        "oracle_gate_changed_l2": oracle_metrics["changed_object_l2"],
        "no_op_changed_l2": noop_metrics["changed_object_l2"],
        "detection_gap": detection_gap,
        "regression_floor": oracle_metrics["changed_object_l2"],
        # Fraction of the predicted-gate error attributable to detection (gate misses).
        "detection_share": detection_gap / predicted_metrics["changed_object_l2"]
        if predicted_metrics["changed_object_l2"] > 0
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "oracle_gate_diagnostic", "counts": args.counts, "seed": args.seed})

    results = [evaluate_count(count, args, device) for count in args.counts]

    lines = [
        "object_count,predicted_gate_changed_l2,oracle_gate_changed_l2,no_op_changed_l2,"
        "detection_gap,regression_floor,detection_share"
    ]
    for row in results:
        lines.append(
            f"{row['object_count']},{row['predicted_gate_changed_l2']:.6f},"
            f"{row['oracle_gate_changed_l2']:.6f},{row['no_op_changed_l2']:.6f},"
            f"{row['detection_gap']:.6f},{row['regression_floor']:.6f},{row['detection_share']:.6f}"
        )
    (output_dir / "oracle_gate_diagnostic.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "counts": list(args.counts),
        "csv": str(output_dir / "oracle_gate_diagnostic.csv"),
        "results": results,
        "detection_is_dominant_bottleneck": {
            str(row["object_count"]): bool(row["detection_share"] > 0.5) for row in results
        },
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
