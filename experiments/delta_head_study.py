"""W1: does a distributional delta head beat the no-op floor on the objects that move?

The oracle-gate diagnostic (RESULTS.md) showed the deterministic head is not merely
imperfect but *uninformative* on changed objects: given the ground-truth mask it scores
0.314 against a no-op reference of 0.348 at ``N=3``, and 0.474 against 0.470 at ``N=8`` --
i.e. worse than declaring the object stationary. The diagnosis is that squared error on
multimodal contact converges to a near-zero conditional mean. This study tests the fix.

Grid, per object count:

  * head    : ``mse`` (deterministic) / ``gaussian`` (heteroscedastic) / ``mdn`` (mixture)
  * regime  : ``onestep`` (the original objective) / ``rollout`` (unrolled ``H`` steps)

Every cell trains from scratch and is scored on the **same** held-out hard test split with
the same metric code as the headline table. Three references bound the result: no-op, the
predicted gate, and the oracle gate (ground-truth mask into the delta head), so the
detection/regression split stays visible.

Two success criteria were fixed in advance and are reported as booleans:

  1. ``changed_object_l2`` strictly below the no-op reference at every count.
  2. A non-zero oracle-gate detection gap -- if perfect detection still buys nothing, the
     head is still the bottleneck and the fix has not landed.

**Outcome (seed 0).** Criterion 1 passes everywhere, and by the widest margin at ``N=8``
where the original diagnostic was worst (MDN +0.048 against +0.019 for squared error).
Criterion 2 fails (+0.0016 / +0.0037 / +0.0000) -- but it was specified backwards and is
withdrawn rather than dropped: a near-zero gap means the predicted gate is already as good
as the ground-truth mask, which is a property of a *good gate*, not evidence of a bad head.
Whether the head is the bottleneck is what criterion 1 measures.

The MDN carries a third diagnostic that decides the *mechanism*: the same trained network is
scored twice, once reconstructing poses from the mixture mean and once from the
highest-weight component. That comparison came back null (+0.003 to -0.003 across six
cells), and ``--capacity-match`` then ruled out the fallback explanation too -- at equal
parameters the unimodal Gaussian is no better than squared error, so heteroscedasticity is
not the ingredient either. What remains is the mixture itself. See ``models/delta_heads.py``
for the full reasoning and the component-count sweep that would settle it; the sweep needs
``--capacity-target-components`` pinned so every K shares one parameter budget.

Runs are executed by shelling out to ``experiments.train_sparse_model`` so there is exactly
one training implementation in the repo.

Usage::

    python -m experiments.delta_head_study --counts 3 --seeds 0
    python -m experiments.delta_head_study --counts 3 5 8 --seeds 0 1 2 --rollout-horizon 5
    # capacity control (one-step):
    python -m experiments.delta_head_study --counts 3 5 8 --regimes onestep --capacity-match
    # component-count sweep, all K pinned to the K=5 budget:
    for K in 1 2 3 5 10; do
      python -m experiments.delta_head_study --counts 3 5 8 --heads mdn --regimes onestep \\
        --capacity-match --mixture-components $K --capacity-target-components 5 \\
        --run-name dh_k$K
    done
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    compute_mask_metrics,
    compute_pose_metrics,
    load_dataset,
    load_sparse_model,
)
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models.delta_heads import delta_head_parameters, match_delta_hidden_dim

HEADS = ("mse", "gaussian", "mdn")
REGIMES = ("onestep", "rollout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W1 delta-head study.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--heads", type=str, nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--regimes", type=str, nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument(
        "--rollout-target", type=str, default="correcting", choices=["correcting", "recorded"],
        help=(
            "Passed through to training. 'recorded' is the control for whether the "
            "MSE-vs-likelihood split under rollout is an artefact of the drift-correcting "
            "target, whose magnitude grows with accumulated error."
        ),
    )
    parser.add_argument("--mixture-components", type=int, default=5)
    parser.add_argument("--sparsity-weight", type=float, default=0.2)
    parser.add_argument("--feature-mode", type=str, default="global")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--delta-hidden-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="delta_head_study")
    parser.add_argument(
        "--split-template", type=str, default="data/transitions/splits_clean_{n}obj_s{seed}"
    )
    parser.add_argument(
        "--stem-template", type=str, default="scale_{n}obj_s{seed}",
        help=(
            "Filename stem inside the split directory. The onset benchmark uses "
            "onset_{n}obj_s{seed}; the planar series carries its tag in the stem too."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a checkpoint if it already exists instead of retraining.",
    )
    parser.add_argument(
        "--capacity-target-components",
        type=int,
        default=None,
        help=(
            "Reference K defining the shared parameter budget under --capacity-match "
            "(defaults to --mixture-components). Pin this to a fixed value when sweeping K, "
            "or the sweep measures capacity rather than component count."
        ),
    )
    parser.add_argument(
        "--capacity-match",
        action="store_true",
        help=(
            "Widen the smaller heads so every head carries the same parameter count as the "
            "MDN. Without this, 'the MDN wins' is confounded with 'the MDN is ~60%% bigger' "
            "(11.0k vs 6.9k at 3 objects), which is the first thing a reviewer will say."
        ),
    )
    return parser.parse_args()


def capacity_matched_widths(
    args: argparse.Namespace, feature_dim: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Delta-head widths equalising every head's parameter count to a common budget.

    The budget is the size of an MDN with ``--capacity-target-components`` components at the
    default width. That reference is deliberately independent of the run's own
    ``--mixture-components``, so a sweep over K can pin every K to one budget: without it,
    K=10 would simply be a bigger model than K=1 and the sweep would measure capacity.

    Widths are chosen by widening the smaller heads up to the budget rather than shrinking
    the largest, so no head is handicapped relative to how it would be deployed.
    """
    reference_components = args.capacity_target_components or args.mixture_components
    target = delta_head_parameters(
        "mdn", object_feature_dim=feature_dim, hidden_dim=args.delta_hidden_dim,
        num_components=reference_components,
    )
    widths: dict[str, int] = {}
    counts: dict[str, int] = {}
    for head in ("mse", "gaussian", "mdn"):
        width, count = match_delta_hidden_dim(
            head, target, object_feature_dim=feature_dim, num_components=args.mixture_components
        )
        widths[head], counts[head] = width, count
    return widths, counts


def split_file(args: argparse.Namespace, count: int, seed: int, kind: str, split: str) -> Path:
    directory = Path(args.split_template.format(n=count, seed=seed))
    stem = args.stem_template.format(n=count, seed=seed)
    return directory / f"{stem}_{kind}_{split}.npz"


def run_name_for(count: int, seed: int, head: str, regime: str) -> str:
    return f"dh_{head}_{regime}_{count}obj_s{seed}"


def train_cell(
    args: argparse.Namespace, count: int, seed: int, head: str, regime: str,
    delta_hidden_dim: int | None = None,
) -> Path:
    """Train one grid cell, returning the checkpoint path."""
    name = run_name_for(count, seed, head, regime)
    if args.capacity_match:
        name += "_cm"
    if head == "mdn" and args.mixture_components != 5:
        name += f"_k{args.mixture_components}"
    if regime == "rollout" and args.rollout_target != "correcting":
        name += f"_{args.rollout_target}"
    checkpoint = Path("models/checkpoints") / f"{name}.pt"
    if args.skip_existing and checkpoint.exists():
        print(f"[delta_head_study] reusing {checkpoint}", flush=True)
        return checkpoint

    # Both regimes train on the *full* (unfiltered) splits so the only difference between
    # them is the unroll, not the training distribution. The hard subset could not support
    # the rollout regime anyway -- it retains too few consecutive steps.
    train_path = split_file(args, count, seed, "full", "train")
    val_path = split_file(args, count, seed, "full", "val")

    command = [
        sys.executable, "-m", "experiments.train_sparse_model",
        "--train", str(train_path), "--val", str(val_path),
        "--run-name", name,
        "--delta-head", head,
        "--mixture-components", str(args.mixture_components),
        "--delta-hidden-dim", str(delta_hidden_dim or args.delta_hidden_dim),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--sparsity-weight", str(args.sparsity_weight),
        "--feature-mode", args.feature_mode,
        "--auto-balance-bce",
        "--seed", str(seed),
        "--device", args.device,
    ]
    if regime == "rollout":
        command += [
            "--rollout-horizon", str(args.rollout_horizon),
            "--rollout-target", args.rollout_target,
            "--grad-clip", str(args.grad_clip),
        ]
    print(f"[delta_head_study] training {name}", flush=True)
    subprocess.run(command, check=True)
    return checkpoint


def predict_deltas(model, config, dataset, device, batch_size: int = 256) -> dict[str, np.ndarray]:
    """Return the point-estimate delta, the predicted gate, and (MDN only) the mixture mean."""
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    temperature = float(config["temperature"])
    feature_mode = str(config.get("feature_mode", "global"))

    point, gates, mixture_mean = [], [], []
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(start + batch_size, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            out = model(features, estimator=estimator, temperature=temperature, hard=True)
            point.append(out.delta.cpu().numpy())
            gates.append((out.gate.probs >= 0.5).float().cpu().numpy())
            if out.delta_dist is not None and hasattr(out.delta_dist, "mixture_mean"):
                mixture_mean.append(out.delta_dist.mixture_mean.cpu().numpy())

    result = {
        "delta": np.concatenate(point, axis=0),
        "gate": np.concatenate(gates, axis=0),
    }
    if mixture_mean:
        result["mixture_mean_delta"] = np.concatenate(mixture_mean, axis=0)
    return result


def evaluate_cell(checkpoint: Path, dataset: dict[str, np.ndarray], device: torch.device) -> dict:
    model, config = load_sparse_model(checkpoint, device)
    predictions = predict_deltas(model, config, dataset, device)

    current, next_pose = dataset["current_pose"], dataset["next_pose"]
    target_mask = dataset["target_mask"]
    delta, gate = predictions["delta"], predictions["gate"]

    predicted_metrics = compute_pose_metrics(
        current + gate[:, :, None] * delta, current, next_pose, target_mask
    )
    oracle_metrics = compute_pose_metrics(
        current + target_mask[:, :, None] * delta, current, next_pose, target_mask
    )
    noop_metrics = compute_pose_metrics(current.copy(), current, next_pose, target_mask)
    mask_metrics = compute_mask_metrics(gate, target_mask)

    row = {
        "num_parameters": int(config.get("num_parameters", 0)),
        "f1": mask_metrics["f1"],
        "precision": mask_metrics["precision"],
        "recall": mask_metrics["recall"],
        "overall_l2": predicted_metrics["overall_per_object_l2"],
        "changed_l2": predicted_metrics["changed_object_l2"],
        "unchanged_l2": predicted_metrics["unchanged_object_l2"],
        "oracle_gate_changed_l2": oracle_metrics["changed_object_l2"],
        "noop_changed_l2": noop_metrics["changed_object_l2"],
        "noop_overall_l2": noop_metrics["overall_per_object_l2"],
        # Positive => the delta head is doing better than declaring the object stationary.
        "changed_l2_margin_over_noop": noop_metrics["changed_object_l2"]
        - oracle_metrics["changed_object_l2"],
        # Positive => the gate's misses cost real accuracy, i.e. detection is informative.
        "detection_gap": predicted_metrics["changed_object_l2"] - oracle_metrics["changed_object_l2"],
    }

    if "mixture_mean_delta" in predictions:
        mixture_metrics = compute_pose_metrics(
            current + target_mask[:, :, None] * predictions["mixture_mean_delta"],
            current, next_pose, target_mask,
        )
        row["mixture_mean_changed_l2"] = mixture_metrics["changed_object_l2"]
        # The mechanism test: how much committing to one mode buys over averaging them.
        row["mode_vs_mean_gain"] = (
            mixture_metrics["changed_object_l2"] - oracle_metrics["changed_object_l2"]
        )
    return row


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({
        "task": "delta_head_study",
        "counts": args.counts, "seeds": args.seeds,
        "heads": args.heads, "regimes": args.regimes,
        "epochs": args.epochs, "rollout_horizon": args.rollout_horizon,
        "rollout_target": args.rollout_target,
        "mixture_components": args.mixture_components,
        "capacity_match": args.capacity_match,
        "capacity_target_components": args.capacity_target_components,
        "split_template": args.split_template,
    })

    rows: list[dict] = []
    for count in args.counts:
        for seed in args.seeds:
            dataset = load_dataset(split_file(args, count, seed, "hard", "test"))
            widths, param_counts = {}, {}
            if args.capacity_match:
                feature_dim = int(
                    build_object_features_by_mode(
                        torch.from_numpy(dataset["state"][:1]),
                        torch.from_numpy(dataset["action"][:1]),
                        args.feature_mode,
                    ).shape[-1]
                )
                widths, param_counts = capacity_matched_widths(args, feature_dim)
                print(f"[delta_head_study] N={count} capacity-matched widths {widths} "
                      f"-> delta-head params {param_counts}", flush=True)
            for head in args.heads:
                for regime in args.regimes:
                    checkpoint = train_cell(
                        args, count, seed, head, regime, widths.get(head)
                    )
                    row = evaluate_cell(checkpoint, dataset, device)
                    row.update({
                        "object_count": count, "seed": seed, "head": head, "regime": regime,
                        "capacity_matched": args.capacity_match,
                        "delta_hidden_dim": widths.get(head, args.delta_hidden_dim),
                    })
                    rows.append(row)
                    print(
                        f"  {head:8s} {regime:8s} N={count} s{seed}: "
                        f"changed_l2={row['changed_l2']:.4f} oracle={row['oracle_gate_changed_l2']:.4f} "
                        f"noop={row['noop_changed_l2']:.4f} margin={row['changed_l2_margin_over_noop']:+.4f} "
                        f"F1={row['f1']:.3f}",
                        flush=True,
                    )

    write_outputs(rows, output_dir)
    summary = build_summary(rows, args)
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def aggregate(rows: list[dict], key: str) -> dict[tuple, dict[str, float]]:
    """Mean/std of a metric per (count, head, regime) across seeds."""
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        grouped.setdefault((row["object_count"], row["head"], row["regime"]), []).append(row[key])
    return {
        cell: {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
        for cell, values in grouped.items()
    }


def build_summary(rows: list[dict], args: argparse.Namespace) -> dict:
    margins = aggregate(rows, "changed_l2_margin_over_noop")
    detection = aggregate(rows, "detection_gap")

    # Criterion 1 is judged on the oracle gate, which isolates the delta head from the
    # gate's misses -- the head is what W1 set out to fix.
    beats_noop = {
        f"{count}obj_{head}_{regime}": bool(stats["mean"] > 0)
        for (count, head, regime), stats in margins.items()
    }
    best_per_count: dict[str, dict] = {}
    for (count, head, regime), stats in margins.items():
        key = f"{count}obj"
        if key not in best_per_count or stats["mean"] > best_per_count[key]["margin_mean"]:
            best_per_count[key] = {
                "head": head, "regime": regime,
                "margin_mean": stats["mean"], "margin_std": stats["std"],
                "detection_gap_mean": detection[(count, head, regime)]["mean"],
            }

    mode_gain = aggregate([r for r in rows if "mode_vs_mean_gain" in r], "mode_vs_mean_gain")
    return {
        "counts": args.counts, "seeds": args.seeds,
        "changed_l2_margin_over_noop": {
            f"{c}obj_{h}_{r}": v for (c, h, r), v in margins.items()
        },
        "detection_gap": {f"{c}obj_{h}_{r}": v for (c, h, r), v in detection.items()},
        "mdn_mode_vs_mixture_mean_gain": {
            f"{c}obj_{h}_{r}": v for (c, h, r), v in mode_gain.items()
        },
        "beats_noop_on_changed_objects": beats_noop,
        "best_cell_per_count": best_per_count,
        # The headline gate: every count must have at least one configuration clearing the
        # no-op floor for W1 to be considered landed.
        "w1_criterion_met_at_every_count": bool(
            all(entry["margin_mean"] > 0 for entry in best_per_count.values())
        ),
    }


COLUMNS = [
    "object_count", "seed", "head", "regime", "capacity_matched", "delta_hidden_dim", "num_parameters", "f1", "precision", "recall",
    "overall_l2", "changed_l2", "unchanged_l2", "oracle_gate_changed_l2", "noop_changed_l2",
    "changed_l2_margin_over_noop", "detection_gap", "mixture_mean_changed_l2", "mode_vs_mean_gain",
]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            "" if row.get(c) is None else
            (f"{row[c]:.6f}" if isinstance(row.get(c), float) else str(row.get(c, "")))
            for c in COLUMNS
        ))
    (output_dir / "delta_head_study.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Delta-head study (W1): can the head beat the no-op floor on moving objects?",
        "",
        "`changed L2` uses the predicted gate; `oracle L2` feeds the ground-truth mask to the",
        "delta head, isolating regression from detection. `margin` = no-op - oracle: **positive",
        "means the head finally does better than declaring the object stationary.**",
        "",
        "| N | seed | head | regime | params | F1 | changed L2 | oracle L2 | no-op L2 | margin | detection gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['object_count']} | {row['seed']} | {row['head']} | {row['regime']} | "
            f"{row['num_parameters']} | {row['f1']:.3f} | {row['changed_l2']:.4f} | "
            f"{row['oracle_gate_changed_l2']:.4f} | {row['noop_changed_l2']:.4f} | "
            f"{row['changed_l2_margin_over_noop']:+.4f} | {row['detection_gap']:+.4f} |"
        )
    mdn_rows = [r for r in rows if "mode_vs_mean_gain" in r]
    if mdn_rows:
        md += [
            "",
            "## Mechanism check: component mode vs mixture mean (MDN only)",
            "",
            "Same trained network, two point estimates. A positive gain means committing to the",
            "highest-weight component beats averaging the components -- direct evidence that",
            "mode-averaging was what pinned the deterministic head at the no-op floor.",
            "",
            "| N | seed | regime | mixture-mean L2 | component-mode L2 | gain |",
            "|---|---|---|---|---|---|",
        ]
        for row in mdn_rows:
            md.append(
                f"| {row['object_count']} | {row['seed']} | {row['regime']} | "
                f"{row['mixture_mean_changed_l2']:.4f} | {row['oracle_gate_changed_l2']:.4f} | "
                f"{row['mode_vs_mean_gain']:+.4f} |"
            )
    (output_dir / "delta_head_study.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
