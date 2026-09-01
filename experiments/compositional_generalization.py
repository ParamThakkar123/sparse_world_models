"""Compositional generalization across object counts (train-N x test-M matrix).

The object-centric hypothesis: because the sparse model is a *per-object* change
gate + residual head with weights shared across objects, a model trained on scenes
with ``N`` objects should transfer to scenes with ``M != N`` objects with no
retraining -- the physics of "does this object move, and by how much" is the same
whether there are 3 boxes or 8. The dense monolith cannot even be *run* off the
diagonal: its input and output layers are sized to a specific object count.

The one obstacle is featurization. The default ``global`` per-object features append
a flattened all-object pose whose width grows with the object count, so a checkpoint
trained at one count cannot ingest another count's features. We therefore train the
sparse models with the count-invariant featurizer (``--feature-mode invariant``,
fixed width 20; see ``train_sparse_model.build_object_features_invariant``) and then
evaluate every (train N, test M) pair.

What this produces
------------------
* A ``train x test`` matrix of change-detection F1 and changed-object L2 for the
  count-invariant sparse model (the full matrix is populated).
* The dense diagonal (train M, test M) for reference, plus a *verified* off-diagonal
  failure: we actually attempt ``dense_N`` on test ``M`` and record the dimension
  error, evidencing that the monolith structurally cannot transfer.
* A no-op reference per test count (count-agnostic by construction).

Caveat: the 8-object data uses a different table geometry (wider bounds, tighter
spacing) than 3/5 objects, so transfer *into or out of* 8 objects also crosses a
geometry shift, not only an object-count shift. The 3<->5 transfer is the cleanest
count-only comparison; 8 is reported but flagged.
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
from experiments.compare_phase4_models import (
    evaluate_dense,
    evaluate_noop,
    evaluate_sparse,
    load_dataset,
    load_dense_model,
    load_sparse_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-object-count transfer matrix (sparse vs dense).")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sparsity-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--run-name", type=str, default="compositional_generalization")
    parser.add_argument(
        "--dense-checkpoint-template",
        type=str,
        default="models/checkpoints/dense_{n}obj_s{seed}.pt",
        help="Where to find the pre-trained (global-featured) dense checkpoint per count.",
    )
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


def run_subprocess(args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Subprocess failed:\n  "
            + " ".join(args)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def train_invariant_sparse(
    args: argparse.Namespace, count: int, checkpoint_dir: Path
) -> Path:
    """Train a count-invariant sparse model on one object count's train split."""
    run_name = f"{args.run_name}/train_sparse_invariant_{count}obj"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        [
            "experiments.train_sparse_model",
            "--train", str(split_path(count, args.seed, "train", args.split_template)),
            "--val", str(split_path(count, args.seed, "val", args.split_template)),
            "--run-name", run_name,
            "--epochs", str(args.epochs),
            "--sparsity-weight", str(args.sparsity_weight),
            "--auto-balance-bce",
            "--feature-mode", "invariant",
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint-dir", str(checkpoint_dir),
        ]
    )
    return checkpoint_path


def main() -> None:
    import torch

    args = parse_args()
    device = torch.device(args.device)
    counts = list(args.counts)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.log_config(
        {
            "task": "compositional_generalization",
            "counts": counts,
            "seed": args.seed,
            "epochs": args.epochs,
            "sparsity_weight": args.sparsity_weight,
            "feature_mode": "invariant",
            "device": args.device,
        }
    )

    # 1) Train one count-invariant sparse model per object count.
    invariant_ckpts: dict[int, Path] = {}
    for count in counts:
        print(f"[compositional] training invariant sparse on {count} objects", flush=True)
        invariant_ckpts[count] = train_invariant_sparse(args, count, checkpoint_dir)

    # Pre-load the test datasets once per count.
    test_sets = {count: load_dataset(split_path(count, args.seed, "test", args.split_template)) for count in counts}

    # 2) Full sparse transfer matrix: invariant model trained on N, evaluated on M.
    sparse_f1 = np.full((len(counts), len(counts)), np.nan)
    sparse_l2 = np.full((len(counts), len(counts)), np.nan)
    rows: list[dict] = []
    sparse_models = {
        count: load_sparse_model(path, device) for count, path in invariant_ckpts.items()
    }
    for i, train_n in enumerate(counts):
        model, config = sparse_models[train_n]
        for j, test_m in enumerate(counts):
            result = evaluate_sparse(model, config, test_sets[test_m], device, args.batch_size)
            f1 = result["mask_metrics"]["f1"]
            l2 = result["pose_metrics"]["changed_object_l2"]
            sparse_f1[i, j] = f1
            sparse_l2[i, j] = l2
            rows.append(
                {
                    "train_count": train_n,
                    "test_count": test_m,
                    "model": "sparse_invariant",
                    "f1": f1,
                    "accuracy": result["mask_metrics"]["accuracy"],
                    "changed_object_l2": l2,
                    "transferable": True,
                }
            )

    # 3) Dense reference: diagonal works; off-diagonal is verified to fail on dims.
    for i, train_n in enumerate(counts):
        dense_ckpt = Path(args.dense_checkpoint_template.format(n=train_n, seed=args.seed))
        dense_model, _ = load_dense_model(dense_ckpt, device)
        for test_m in counts:
            try:
                result = evaluate_dense(dense_model, test_sets[test_m], device, args.batch_size)
                rows.append(
                    {
                        "train_count": train_n,
                        "test_count": test_m,
                        "model": "dense",
                        "f1": result["mask_metrics"]["f1"],
                        "accuracy": result["mask_metrics"]["accuracy"],
                        "changed_object_l2": result["pose_metrics"]["changed_object_l2"],
                        "transferable": True,
                    }
                )
            except (RuntimeError, ValueError) as error:
                # A monolithic MLP sized to train_n objects cannot ingest test_m>!=n
                # state / emit test_m poses -- record the structural failure.
                rows.append(
                    {
                        "train_count": train_n,
                        "test_count": test_m,
                        "model": "dense",
                        "f1": float("nan"),
                        "accuracy": float("nan"),
                        "changed_object_l2": float("nan"),
                        "transferable": False,
                        "error": type(error).__name__,
                    }
                )

    # 4) No-op reference (count-agnostic).
    for test_m in counts:
        result = evaluate_noop(test_sets[test_m])
        rows.append(
            {
                "train_count": None,
                "test_count": test_m,
                "model": "no_op",
                "f1": result["mask_metrics"]["f1"],
                "accuracy": result["mask_metrics"]["accuracy"],
                "changed_object_l2": result["pose_metrics"]["changed_object_l2"],
                "transferable": True,
            }
        )

    write_csv(rows, output_dir / "transfer_matrix.csv")
    figure_path = plot_matrices(counts, sparse_f1, sparse_l2, output_dir / "transfer_matrix.png")

    # Transfer retention: off-diagonal F1 relative to the in-distribution diagonal.
    diag = np.diag(sparse_f1)
    retention = []
    for i in range(len(counts)):
        for j in range(len(counts)):
            if i != j and diag[j] > 0:
                retention.append(sparse_f1[i, j] / diag[j])
    dense_offdiag_transferable = [
        row["transferable"] for row in rows if row["model"] == "dense" and row["train_count"] != row["test_count"]
    ]

    summary = {
        "counts": counts,
        "transfer_matrix_csv": str(output_dir / "transfer_matrix.csv"),
        "figure": figure_path,
        "sparse_f1_matrix": {
            str(train_n): {str(test_m): float(sparse_f1[i, j]) for j, test_m in enumerate(counts)}
            for i, train_n in enumerate(counts)
        },
        "sparse_diagonal_mean_f1": float(np.mean(diag)),
        "sparse_offdiagonal_mean_f1": float(
            np.mean([sparse_f1[i, j] for i in range(len(counts)) for j in range(len(counts)) if i != j])
        ),
        "sparse_transfer_retention_mean": float(np.mean(retention)) if retention else None,
        "dense_offdiagonal_ever_transferable": bool(any(dense_offdiag_transferable)),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def write_csv(rows: list[dict], path: Path) -> None:
    lines = ["train_count,test_count,model,f1,accuracy,changed_object_l2,transferable,error"]
    for row in rows:
        train = "" if row["train_count"] is None else row["train_count"]
        f1 = "" if row["f1"] != row["f1"] else f"{row['f1']:.6f}"  # NaN check
        acc = "" if row["accuracy"] != row["accuracy"] else f"{row['accuracy']:.6f}"
        l2 = "" if row["changed_object_l2"] != row["changed_object_l2"] else f"{row['changed_object_l2']:.6f}"
        lines.append(
            f"{train},{row['test_count']},{row['model']},{f1},{acc},{l2},"
            f"{row['transferable']},{row.get('error', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_matrices(counts: list[int], f1: np.ndarray, l2: np.ndarray, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    panels = (
        (axes[0], f1, "Change-detection F1 (higher better)", "viridis", "{:.2f}"),
        (axes[1], l2, "Changed-object L2 (lower better)", "viridis_r", "{:.3f}"),
    )
    for ax, matrix, title, cmap, fmt in panels:
        image = ax.imshow(matrix, cmap=cmap, aspect="equal")
        ax.set_xticks(range(len(counts)), [f"{c}" for c in counts])
        ax.set_yticks(range(len(counts)), [f"{c}" for c in counts])
        ax.set_xlabel("Test object count")
        ax.set_ylabel("Train object count")
        ax.set_title(title)
        for i in range(len(counts)):
            for j in range(len(counts)):
                edge = "white" if i == j else "black"
                ax.text(
                    j, i, fmt.format(matrix[i, j]),
                    ha="center", va="center",
                    color=edge, fontweight="bold" if i == j else "normal",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Count-invariant sparse model: cross-object-count transfer (diagonal = in-distribution)")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
