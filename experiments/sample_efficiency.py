"""Sample-efficiency sweep: retrain sparse and dense on subsets of the training
transitions and measure how test change-detection and pose accuracy scale with
the number of training samples.

Motivation
----------
A central claim of the sparse/residual model is that *explicitly modelling what
changes* is a stronger inductive bias than regressing every object's pose. A
stronger prior should pay off most when data is scarce: the sparse model should
reach good change-detection F1 and low changed-object L2 with far fewer training
transitions than the dense monolith needs. This experiment tests that directly by
retraining both models on 10/25/50/100% of the training split and evaluating each
on the *full, fixed* test split.

Mechanics
---------
For each fraction we subsample the first ``round(fraction * N)`` training
transitions (the split is already shuffled at creation, so a prefix is a valid
random subsample, and it matches the dense trainer's existing ``--max-train-samples``
semantics -- both models see the *same* subset). We then shell out to the existing
trainers and to ``compare_phase4_models`` so the metric definitions are identical
to the headline results, and read the per-fraction ``detailed_results.json`` back.

The validation split is held at full size (it only picks the checkpoint, it is not
a data-budget axis); the test split is always full. Single seed by default -- pass
``--seed`` and re-run under different ``--run-name`` values to get dispersion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import ExperimentLogger
from models import infer_num_objects_from_state_dim

MODELS = ("sparse", "dense", "no_op")
MODEL_STYLE = {
    "sparse": {"color": "#1b9e77", "marker": "o"},
    "dense": {"color": "#d95f02", "marker": "s"},
    "no_op": {"color": "#7570b3", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-efficiency sweep for sparse vs dense.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Fractions of the training split to retrain on.",
    )
    parser.add_argument("--dense-epochs", type=int, default=25)
    parser.add_argument("--sparse-epochs", type=int, default=15)
    parser.add_argument("--dense-hidden-dim", type=int, default=256)
    parser.add_argument("--dense-num-layers", type=int, default=3)
    parser.add_argument("--sparsity-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="sample_efficiency")
    return parser.parse_args()


def count_samples(path: Path) -> int:
    with np.load(path) as data:
        return int(data["s_t"].shape[0])


def infer_object_count(path: Path) -> int:
    with np.load(path) as data:
        return infer_num_objects_from_state_dim(int(data["s_t"].shape[1]))


def run_subprocess(args: list[str]) -> None:
    """Run a training / eval module, streaming failures with full context."""
    result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Subprocess failed:\n  "
            + " ".join(args)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def fraction_tag(fraction: float) -> str:
    return f"f{int(round(fraction * 100)):03d}"


def train_and_evaluate(
    args: argparse.Namespace,
    fraction: float,
    num_train_samples: int,
    checkpoint_dir: Path,
) -> dict:
    """Train both models on a subsample and evaluate on the fixed test split."""
    tag = fraction_tag(fraction)
    # Each trainer saves its checkpoint as ``<checkpoint-dir>/<run-name>.pt``, so the
    # checkpoint path is derived from the exact run-name we pass it below.
    dense_run = f"{args.run_name}/train_dense_{tag}"
    sparse_run = f"{args.run_name}/train_sparse_{tag}"
    dense_ckpt = checkpoint_dir / f"{dense_run}.pt"
    sparse_ckpt = checkpoint_dir / f"{sparse_run}.pt"
    # The trainers create --checkpoint-dir but not the nested run-name subdir the
    # slashed run name implies, so ensure the leaf directory exists first.
    dense_ckpt.parent.mkdir(parents=True, exist_ok=True)

    run_subprocess(
        [
            "experiments.train_dense_baseline",
            "--train", str(args.train),
            "--val", str(args.val),
            "--run-name", dense_run,
            "--epochs", str(args.dense_epochs),
            "--hidden-dim", str(args.dense_hidden_dim),
            "--num-layers", str(args.dense_num_layers),
            "--batch-size", str(args.batch_size),
            "--max-train-samples", str(num_train_samples),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint-dir", str(checkpoint_dir),
        ]
    )
    run_subprocess(
        [
            "experiments.train_sparse_model",
            "--train", str(args.train),
            "--val", str(args.val),
            "--run-name", sparse_run,
            "--epochs", str(args.sparse_epochs),
            "--sparsity-weight", str(args.sparsity_weight),
            "--auto-balance-bce",
            "--batch-size", str(args.batch_size),
            "--max-train-samples", str(num_train_samples),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint-dir", str(checkpoint_dir),
        ]
    )
    run_subprocess(
        [
            "experiments.compare_phase4_models",
            "--data", str(args.test),
            "--dense-checkpoint", str(dense_ckpt),
            "--sparse-checkpoint", str(sparse_ckpt),
            "--run-name", f"{args.run_name}/eval_{tag}",
            "--device", args.device,
            "--timing-iters", "1",
            "--warmup-iters", "0",
            "--num-qualitative", "0",
        ]
    )

    detailed_path = Path("experiments/runs") / args.run_name / f"eval_{tag}" / "detailed_results.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    row = {"fraction": fraction, "num_train_samples": num_train_samples, "models": {}}
    for model_name in MODELS:
        change = detailed[model_name]["change_metrics"]
        pose = detailed[model_name]["pose_metrics"]
        row["models"][model_name] = {
            "f1": change["f1"],
            "accuracy": change["accuracy"],
            "precision": change["precision"],
            "recall": change["recall"],
            "changed_object_l2": pose["changed_object_l2"],
            "overall_per_object_l2": pose["overall_per_object_l2"],
        }
    return row


def write_csv(rows: list[dict], object_count: int, path: Path) -> None:
    lines = [
        "object_count,fraction,num_train_samples,model,f1,accuracy,precision,recall,"
        "changed_object_l2,overall_per_object_l2"
    ]
    for row in rows:
        for model_name in MODELS:
            metrics = row["models"][model_name]
            lines.append(
                f"{object_count},{row['fraction']:.4f},{row['num_train_samples']},{model_name},"
                f"{metrics['f1']:.6f},{metrics['accuracy']:.6f},{metrics['precision']:.6f},"
                f"{metrics['recall']:.6f},{metrics['changed_object_l2']:.6f},"
                f"{metrics['overall_per_object_l2']:.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(rows: list[dict], object_count: int, path: Path) -> None:
    samples = [row["num_train_samples"] for row in rows]
    panels = (
        ("f1", "Change-detection F1", False),
        ("accuracy", "Change-detection accuracy", False),
        ("changed_object_l2", "Changed-object pose L2 (lower better)", True),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.5), constrained_layout=True)
    for (metric, title, lower_better), ax in zip(panels, axes):
        for model_name in MODELS:
            values = [row["models"][model_name][metric] for row in rows]
            style = MODEL_STYLE[model_name]
            ax.plot(
                samples,
                values,
                label=model_name,
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.8,
            )
        ax.set_xlabel("# training transitions")
        ax.set_ylabel(title)
        ax.set_title(title + (" ↓" if lower_better else " ↑"))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Sample efficiency | {object_count} objects | fixed test split")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    total_samples = count_samples(args.train)
    object_count = infer_object_count(args.train)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resolve each fraction to a concrete sample count; drop duplicates (e.g. two
    # fractions that round to the same count) while preserving order.
    resolved: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in sorted(args.fractions):
        num = max(int(round(fraction * total_samples)), 1)
        num = min(num, total_samples)
        if num not in seen:
            resolved.append((fraction, num))
            seen.add(num)

    logger.log_config(
        {
            "task": "sample_efficiency",
            "train": str(args.train),
            "val": str(args.val),
            "test": str(args.test),
            "object_count": object_count,
            "total_train_samples": total_samples,
            "fractions": [fraction for fraction, _ in resolved],
            "sample_counts": [num for _, num in resolved],
            "dense_epochs": args.dense_epochs,
            "sparse_epochs": args.sparse_epochs,
            "sparsity_weight": args.sparsity_weight,
            "seed": args.seed,
            "device": args.device,
        }
    )

    rows: list[dict] = []
    for fraction, num_train_samples in resolved:
        print(f"[sample_efficiency] fraction={fraction:.2f} -> {num_train_samples} samples", flush=True)
        row = train_and_evaluate(args, fraction, num_train_samples, checkpoint_dir)
        rows.append(row)
        logger.log_metrics(
            num_train_samples,
            fraction=fraction,
            sparse_f1=row["models"]["sparse"]["f1"],
            dense_f1=row["models"]["dense"]["f1"],
            sparse_accuracy=row["models"]["sparse"]["accuracy"],
            dense_accuracy=row["models"]["dense"]["accuracy"],
            sparse_changed_object_l2=row["models"]["sparse"]["changed_object_l2"],
            dense_changed_object_l2=row["models"]["dense"]["changed_object_l2"],
        )

    curves_csv = output_dir / "sample_efficiency_curves.csv"
    figure_path = output_dir / "sample_efficiency.png"
    write_csv(rows, object_count, curves_csv)
    plot_curves(rows, object_count, figure_path)

    summary = {
        "object_count": object_count,
        "total_train_samples": total_samples,
        "curves_csv": str(curves_csv),
        "figure": str(figure_path),
        "sample_counts": [num for _, num in resolved],
        "sparse_f1_by_samples": {
            str(row["num_train_samples"]): row["models"]["sparse"]["f1"] for row in rows
        },
        "dense_f1_by_samples": {
            str(row["num_train_samples"]): row["models"]["dense"]["f1"] for row in rows
        },
        "sparse_beats_dense_f1_at_min_samples": bool(
            rows[0]["models"]["sparse"]["f1"] > rows[0]["models"]["dense"]["f1"]
        ),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
