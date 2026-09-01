"""How much did the split leak inflate the published numbers?

``build_clean_splits`` documents the bug: because ``create_hard_subset`` marks ``done`` at
every kept chunk, the subsequent ``split_dataset`` call fingerprints chunks rather than
episodes, and 25% of source episodes end up with rows in both train and test. This script
puts a number on what that cost.

Three conditions, deliberately separated because they answer different questions:

  ``published``   old checkpoint, old (leaky) test split
                  -- reproduces the numbers currently in RESULTS.md.
  ``old_on_clean`` old checkpoint, clean test split
                  -- the *same model* scored on data it provably never trained on. The gap
                     to ``published`` is the direct memorisation effect, with training held
                     fixed, and is the cleanest single estimate of the inflation.
  ``clean``       clean checkpoint, clean test split
                  -- the honest number to report going forward. It differs from
                     ``old_on_clean`` by training set as well, so it is not a pure leakage
                     measurement; it is the replacement result.

Note ``old_on_clean`` is mildly *pessimistic* as a leakage estimate: the clean test split
contains episodes the old model never saw at all, whereas an ideal counterfactual would
re-split the same episodes without leakage. Read the three together rather than quoting one.

Usage::

    python -m experiments.leakage_impact --counts 3 5 8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    evaluate_dense,
    evaluate_noop,
    evaluate_sparse,
    load_dataset,
    load_dense_model,
    load_sparse_model,
)

CONDITIONS = ("published", "old_on_clean", "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify the split-leakage inflation.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="leakage_impact")
    return parser.parse_args()


def old_test(count: int, seed: int) -> Path:
    tag = f"{count}obj_s{seed}"
    return Path(f"data/transitions/splits_{tag}/scale_{tag}_hard_test.npz")


def clean_test(count: int, seed: int) -> Path:
    tag = f"{count}obj_s{seed}"
    return Path(f"data/transitions/splits_clean_{tag}/scale_{tag}_hard_test.npz")


def checkpoints(count: int, seed: int, clean: bool) -> tuple[Path, Path]:
    tag = f"{count}obj_s{seed}"
    prefix = "clean_" if clean else ""
    return (
        Path(f"models/checkpoints/sparse_{prefix}{tag}.pt"),
        Path(f"models/checkpoints/dense_{prefix}{tag}.pt"),
    )


def score(count: int, seed: int, condition: str, device: torch.device) -> list[dict]:
    data_path = old_test(count, seed) if condition == "published" else clean_test(count, seed)
    sparse_path, dense_path = checkpoints(count, seed, clean=(condition == "clean"))
    for path in (data_path, sparse_path, dense_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (condition '{condition}')")

    dataset = load_dataset(data_path)
    sparse_model, sparse_config = load_sparse_model(sparse_path, device)
    dense_model, _ = load_dense_model(dense_path, device)

    evaluations = {
        "sparse": evaluate_sparse(sparse_model, sparse_config, dataset, device, 256),
        "dense": evaluate_dense(dense_model, dataset, device, 256),
        "no_op": evaluate_noop(dataset),
    }
    rows = []
    for model_name, evaluation in evaluations.items():
        rows.append({
            "object_count": count, "seed": seed, "condition": condition, "model": model_name,
            "f1": evaluation["mask_metrics"]["f1"],
            "overall_l2": evaluation["pose_metrics"]["overall_per_object_l2"],
            "changed_l2": evaluation["pose_metrics"]["changed_object_l2"],
            "unchanged_l2": evaluation["pose_metrics"]["unchanged_object_l2"],
            "num_test_rows": int(dataset["state"].shape[0]),
        })
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({
        "task": "leakage_impact", "counts": args.counts, "seeds": args.seeds,
        "conditions": list(CONDITIONS),
    })

    rows: list[dict] = []
    skipped: list[str] = []
    for count in args.counts:
        for seed in args.seeds:
            for condition in CONDITIONS:
                try:
                    rows.extend(score(count, seed, condition, device))
                except FileNotFoundError as error:
                    # A missing clean checkpoint just means clean_pipeline.sh has not run
                    # for that cell yet; report it rather than aborting the whole sweep.
                    skipped.append(f"{count}obj s{seed} {condition}: {error}")

    write_outputs(rows, output_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))
    if skipped:
        print(f"\nSkipped {len(skipped)} cells:", *skipped, sep="\n  ")


def build_summary(rows: list[dict]) -> dict:
    def stats(model: str, condition: str, count: int, metric: str) -> dict | None:
        values = [
            row[metric] for row in rows
            if row["model"] == model and row["condition"] == condition and row["object_count"] == count
        ]
        if not values:
            return None
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}

    counts = sorted({row["object_count"] for row in rows})
    inflation: dict[str, dict] = {}
    for count in counts:
        entry: dict[str, object] = {}
        for model in ("sparse", "dense"):
            published = stats(model, "published", count, "overall_l2")
            on_clean = stats(model, "old_on_clean", count, "overall_l2")
            clean = stats(model, "clean", count, "overall_l2")
            if published and on_clean:
                # Positive => error rises once the memorised rows are removed, i.e. the
                # published figure was optimistic.
                entry[f"{model}_overall_l2_inflation"] = on_clean["mean"] - published["mean"]
                entry[f"{model}_overall_l2_inflation_ratio"] = (
                    on_clean["mean"] / max(published["mean"], 1e-9)
                )
            entry[f"{model}_published"] = published
            entry[f"{model}_old_on_clean"] = on_clean
            entry[f"{model}_clean"] = clean

        # The claim that must survive: sparse still beats dense on honest splits.
        sparse_clean = stats("sparse", "clean", count, "overall_l2")
        dense_clean = stats("dense", "clean", count, "overall_l2")
        if sparse_clean and dense_clean:
            entry["sparse_beats_dense_on_clean"] = bool(sparse_clean["mean"] < dense_clean["mean"])
            entry["clean_l2_ratio_dense_over_sparse"] = (
                dense_clean["mean"] / max(sparse_clean["mean"], 1e-9)
            )
        inflation[f"{count}obj"] = entry

    ordering_holds = [
        entry["sparse_beats_dense_on_clean"]
        for entry in inflation.values()
        if "sparse_beats_dense_on_clean" in entry
    ]
    return {
        "per_count": inflation,
        "sparse_beats_dense_on_clean_everywhere": bool(all(ordering_holds)) if ordering_holds else None,
    }


COLUMNS = [
    "object_count", "seed", "condition", "model", "num_test_rows",
    "f1", "overall_l2", "changed_l2", "unchanged_l2",
]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "leakage_impact.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Leakage impact on the published numbers",
        "",
        "`published` = old checkpoint on the old (leaky) test split -- reproduces RESULTS.md.",
        "`old_on_clean` = the same model on the episode-disjoint test split, so the gap is the",
        "memorisation effect with training held fixed. `clean` = retrained on clean splits: the",
        "honest number to report going forward.",
        "",
        "| N | seed | condition | model | F1 | overall L2 | changed L2 | unchanged L2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['object_count']} | {row['seed']} | {row['condition']} | {row['model']} | "
            f"{row['f1']:.3f} | {row['overall_l2']:.4f} | {row['changed_l2']:.4f} | "
            f"{row['unchanged_l2']:.4f} |"
        )
    (output_dir / "leakage_impact.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
