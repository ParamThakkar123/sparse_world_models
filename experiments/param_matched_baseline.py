"""Parameter-matched dense baseline (defensibility experiment).

The headline comparison gives the dense monolith 8.6-11.1x more parameters than the
sparse model, so "sparse wins" is confounded with "sparse is smaller." This experiment
removes that confound: we train a dense MLP shrunk to the *same parameter budget* as the
sparse model (``hidden_dim=64, num_layers=3`` matches sparse to ~1.00x at every count)
and show the sparse model still wins on prediction accuracy and change detection. The
full-size dense model is kept in the comparison for reference.

Only the parameter-matched dense model is trained here (one small MLP per count); the
sparse and full-dense checkpoints are the existing canonical ones. Evaluation reuses the
exact metric definitions from ``compare_phase4_models``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    count_parameters,
    evaluate_dense,
    evaluate_noop,
    evaluate_sparse,
    load_dataset,
    load_dense_model,
    load_sparse_model,
)

# Bars are drawn (and the CSV ordered) in this sequence.
MODEL_ORDER = ("sparse", "dense_matched", "dense_full", "no_op")
MODEL_STYLE = {
    "sparse": {"color": "#1b9e77"},
    "dense_matched": {"color": "#d95f02"},
    "dense_full": {"color": "#e7a15c"},
    "no_op": {"color": "#7570b3"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter-matched dense baseline vs sparse.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64, help="Matched-dense hidden width (~sparse params).")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
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
    parser.add_argument("--run-name", type=str, default="param_matched_baseline")
    parser.add_argument("--sparse-template", type=str, default="models/checkpoints/sparse_{n}obj_s{seed}.pt")
    parser.add_argument("--dense-template", type=str, default="models/checkpoints/dense_{n}obj_s{seed}.pt")
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
            "Subprocess failed:\n  " + " ".join(args)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def train_matched_dense(args: argparse.Namespace, count: int, checkpoint_dir: Path) -> Path:
    run_name = f"{args.run_name}/train_dense_matched_{count}obj"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        [
            "experiments.train_dense_baseline",
            "--train", str(split_path(count, args.seed, "train", args.split_template)),
            "--val", str(split_path(count, args.seed, "val", args.split_template)),
            "--run-name", run_name,
            "--epochs", str(args.epochs),
            "--hidden-dim", str(args.hidden_dim),
            "--num-layers", str(args.num_layers),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--device", args.device,
            "--checkpoint-dir", str(checkpoint_dir),
        ]
    )
    return checkpoint_path


def collect_row(count: int, model_name: str, params: int, evaluation: dict) -> dict:
    return {
        "object_count": count,
        "model": model_name,
        "num_parameters": params,
        "f1": evaluation["mask_metrics"]["f1"],
        "accuracy": evaluation["mask_metrics"]["accuracy"],
        "changed_object_l2": evaluation["pose_metrics"]["changed_object_l2"],
        "overall_per_object_l2": evaluation["pose_metrics"]["overall_per_object_l2"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    counts = list(args.counts)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.log_config(
        {
            "task": "param_matched_baseline",
            "counts": counts,
            "seed": args.seed,
            "matched_hidden_dim": args.hidden_dim,
            "matched_num_layers": args.num_layers,
            "epochs": args.epochs,
            "device": args.device,
        }
    )

    rows: list[dict] = []
    for count in counts:
        print(f"[param_matched] training matched dense on {count} objects", flush=True)
        matched_ckpt = train_matched_dense(args, count, checkpoint_dir)

        dataset = load_dataset(split_path(count, args.seed, "test", args.split_template))
        sparse_model, sparse_cfg = load_sparse_model(
            Path(args.sparse_template.format(n=count, seed=args.seed)), device
        )
        matched_model, _ = load_dense_model(matched_ckpt, device)
        full_model, _ = load_dense_model(Path(args.dense_template.format(n=count, seed=args.seed)), device)

        rows.append(collect_row(count, "sparse", count_parameters(sparse_model),
                                evaluate_sparse(sparse_model, sparse_cfg, dataset, device, args.batch_size)))
        rows.append(collect_row(count, "dense_matched", count_parameters(matched_model),
                                evaluate_dense(matched_model, dataset, device, args.batch_size)))
        rows.append(collect_row(count, "dense_full", count_parameters(full_model),
                                evaluate_dense(full_model, dataset, device, args.batch_size)))
        rows.append(collect_row(count, "no_op", 0, evaluate_noop(dataset)))

    write_csv(rows, output_dir / "param_matched_results.csv")
    figure_path = plot_results(counts, rows, output_dir / "param_matched.png")

    # Defensibility claim: sparse beats the *parameter-matched* dense at every count.
    sparse_beats_matched = {}
    for count in counts:
        by_model = {row["model"]: row for row in rows if row["object_count"] == count}
        sparse_beats_matched[str(count)] = {
            "overall_l2": bool(by_model["sparse"]["overall_per_object_l2"] < by_model["dense_matched"]["overall_per_object_l2"]),
            "changed_object_l2": bool(by_model["sparse"]["changed_object_l2"] < by_model["dense_matched"]["changed_object_l2"]),
            "f1": bool(by_model["sparse"]["f1"] > by_model["dense_matched"]["f1"]),
        }

    summary = {
        "counts": counts,
        "matched_hidden_dim": args.hidden_dim,
        "results_csv": str(output_dir / "param_matched_results.csv"),
        "figure": figure_path,
        "param_ratio_matched": {
            str(count): (
                next(r["num_parameters"] for r in rows if r["object_count"] == count and r["model"] == "dense_matched")
                / next(r["num_parameters"] for r in rows if r["object_count"] == count and r["model"] == "sparse")
            )
            for count in counts
        },
        "sparse_beats_matched_dense": sparse_beats_matched,
        "sparse_beats_matched_dense_all": bool(
            all(all(v.values()) for v in sparse_beats_matched.values())
        ),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def write_csv(rows: list[dict], path: Path) -> None:
    lines = ["object_count,model,num_parameters,f1,accuracy,changed_object_l2,overall_per_object_l2"]
    for row in rows:
        lines.append(
            f"{row['object_count']},{row['model']},{row['num_parameters']},"
            f"{row['f1']:.6f},{row['accuracy']:.6f},{row['changed_object_l2']:.6f},"
            f"{row['overall_per_object_l2']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(counts: list[int], rows: list[dict], path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.2
    x = np.arange(len(counts))
    panels = (
        (axes[0], "overall_per_object_l2", "Overall per-object L2 (lower better)"),
        (axes[1], "f1", "Change-detection F1 (higher better)"),
    )
    for ax, metric, title in panels:
        for offset, model_name in enumerate(MODEL_ORDER):
            values = [
                next(r[metric] for r in rows if r["object_count"] == c and r["model"] == model_name)
                for c in counts
            ]
            ax.bar(x + (offset - 1.5) * width, values, width, label=model_name,
                   color=MODEL_STYLE[model_name]["color"])
        ax.set_xticks(x, [f"{c}" for c in counts])
        ax.set_xlabel("Object count")
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Sparse vs parameter-matched dense (matched = same param budget as sparse)")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
