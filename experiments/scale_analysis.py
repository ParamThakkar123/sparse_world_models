"""W4: how does the sparse advantage scale with object count, at one fixed geometry?

The published series confounded count with geometry -- N=3,5 used bounds +-0.18 /
separation 0.12 while N=8 used +-0.22 / 0.09, because 8 boxes did not fit the default
packing -- so any cross-N trend touching 8 crossed a geometry shift as well. The series this
script reads (``experiments/scale_series_pipeline.sh``) holds bounds and separation fixed at
+-0.26 / 0.09 for **every** count from 3 to 20, so N is the only variable.

Because the table is fixed, density rises with N (0.028 -> 0.185 objects per unit area from
N=3 to N=20). That is deliberate: it is the axis the "copying unchanged objects compounds"
prediction is actually about, and it keeps every object inside the pusher's +-0.26 reach.
Holding density constant instead would need bounds up to +-0.735 against a table of half-size
0.34, putting objects off the table and out of reach.

The prediction under test: as N grows, a larger share of the scene is static each step, so a
model that copies unchanged objects verbatim should pull further ahead of one that re-predicts
everything. Three quantities track that:

  * ``unchanged_l2`` ratio (dense / sparse) -- the mechanism, and where the effect should be
    largest.
  * ``f1`` gap -- change detection, which the ungated models cannot do at all.
  * ``margin_over_noop`` on overall L2 -- the metric known to saturate in sparse scenes, kept
    so the saturation is visible rather than hidden.

Usage::

    python -m experiments.scale_analysis --counts 3 5 8 12 20 --seeds 0 1 2 3 4
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

MODELS = ("sparse", "dense", "no_op")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W4 scale analysis at fixed geometry.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8, 12, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="scale_analysis")
    parser.add_argument("--tag", type=str, default="uni")
    return parser.parse_args()


def paths(count: int, seed: int, tag: str) -> tuple[Path, Path, Path]:
    stem = f"scale_{count}obj_{tag}_s{seed}"
    return (
        Path(f"data/transitions/splits_clean_{count}obj_{tag}_s{seed}/{stem}_hard_test.npz"),
        Path(f"models/checkpoints/sparse_{tag}_{count}obj_s{seed}.pt"),
        Path(f"models/checkpoints/dense_{tag}_{count}obj_s{seed}.pt"),
    )


def score(count: int, seed: int, tag: str, device: torch.device) -> list[dict]:
    data_path, sparse_path, dense_path = paths(count, seed, tag)
    for path in (data_path, sparse_path, dense_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")

    dataset = load_dataset(data_path)
    sparse_model, sparse_config = load_sparse_model(sparse_path, device)
    dense_model, _ = load_dense_model(dense_path, device)
    evaluations = {
        "sparse": evaluate_sparse(sparse_model, sparse_config, dataset, device, 256),
        "dense": evaluate_dense(dense_model, dataset, device, 256),
        "no_op": evaluate_noop(dataset),
    }
    noop_l2 = evaluations["no_op"]["pose_metrics"]["overall_per_object_l2"]
    target_mask = dataset["target_mask"]

    rows = []
    for name, evaluation in evaluations.items():
        pose, mask_metrics = evaluation["pose_metrics"], evaluation["mask_metrics"]
        rows.append({
            "object_count": count, "seed": seed, "model": name,
            "f1": mask_metrics["f1"],
            "overall_l2": pose["overall_per_object_l2"],
            "changed_l2": pose["changed_object_l2"],
            "unchanged_l2": pose["unchanged_object_l2"],
            "margin_over_noop": noop_l2 - pose["overall_per_object_l2"],
            # Fraction of objects that actually move -- the driver of the whole effect, so
            # it belongs in the table rather than being asserted.
            "changed_object_fraction": float(target_mask.mean()),
        })
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "scale_analysis", "counts": args.counts,
                       "seeds": args.seeds, "tag": args.tag,
                       "geometry": "bounds +-0.26, separation 0.09, identical for every count"})

    rows, skipped = [], []
    for count in args.counts:
        for seed in args.seeds:
            try:
                rows.extend(score(count, seed, args.tag, device))
            except FileNotFoundError as error:
                skipped.append(str(error))

    write_outputs(rows, output_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))
    if skipped:
        print(f"\nSkipped {len(skipped)} cells (first few):", *skipped[:5], sep="\n  ")


def build_summary(rows: list[dict]) -> dict:
    per_count: dict[str, dict] = {}
    counts = sorted({row["object_count"] for row in rows})
    for count in counts:
        entry: dict[str, object] = {}
        for model in MODELS:
            for metric in ("f1", "overall_l2", "unchanged_l2", "changed_l2", "margin_over_noop"):
                values = [r[metric] for r in rows
                          if r["object_count"] == count and r["model"] == model]
                if values:
                    entry[f"{model}_{metric}"] = {
                        "mean": float(np.mean(values)), "std": float(np.std(values)),
                        "n": len(values),
                    }
        fractions = [r["changed_object_fraction"] for r in rows if r["object_count"] == count]
        if fractions:
            entry["changed_object_fraction"] = float(np.mean(fractions))
        sparse_unchanged = entry.get("sparse_unchanged_l2")
        dense_unchanged = entry.get("dense_unchanged_l2")
        if sparse_unchanged and dense_unchanged:
            # The mechanism ratio: how much error the gate keeps out of the static majority.
            # NOTE this is a poor statistic here and is kept only for continuity with the
            # published tables. Its denominator is the sparse model's unchanged-object error,
            # which sits at ~0.001-0.002 -- essentially the gate's false-positive rate -- so
            # the ratio is dominated by noise in a near-zero quantity and swings by 2x
            # between adjacent counts. The *difference* below is the quantity that actually
            # tracks the mechanism.
            entry["unchanged_l2_ratio_dense_over_sparse"] = (
                dense_unchanged["mean"] / max(sparse_unchanged["mean"], 1e-12)
            )
            entry["unchanged_l2_gap_dense_minus_sparse"] = (
                dense_unchanged["mean"] - sparse_unchanged["mean"]
            )
        sparse_f1 = entry.get("sparse_f1")
        dense_f1 = entry.get("dense_f1")
        if sparse_f1 and dense_f1:
            entry["f1_gap_sparse_minus_dense"] = sparse_f1["mean"] - dense_f1["mean"]
        sparse_overall = entry.get("sparse_overall_l2")
        dense_overall = entry.get("dense_overall_l2")
        if sparse_overall and dense_overall:
            entry["overall_l2_ratio_dense_over_sparse"] = (
                dense_overall["mean"] / max(sparse_overall["mean"], 1e-12)
            )
        per_count[f"{count}obj"] = entry

    ratios: list[tuple[int, float]] = sorted(
        (int(key[:-3]), float(value["unchanged_l2_ratio_dense_over_sparse"]))  # type: ignore[arg-type]
        for key, value in per_count.items()
        if "unchanged_l2_ratio_dense_over_sparse" in value
    )
    def series(key: str) -> list[tuple[int, float]]:
        return sorted(
            (int(name[:-3]), float(value[key]))  # type: ignore[arg-type]
            for name, value in per_count.items() if key in value
        )

    def monotone(points: list[tuple[int, float]]) -> bool | None:
        if len(points) < 2:
            return None
        return bool(all(b >= a for (_, a), (_, b) in zip(points, points[1:])))

    gaps = series("unchanged_l2_gap_dense_minus_sparse")
    f1_gaps = series("f1_gap_sparse_minus_dense")
    dense_unchanged: list[tuple[int, float]] = sorted(
        (int(name[:-3]), float(value["dense_unchanged_l2"]["mean"]))  # type: ignore[index]
        for name, value in per_count.items() if "dense_unchanged_l2" in value
    )

    return {
        "per_count": per_count,
        "unchanged_l2_ratio_by_count": {str(count): ratio for count, ratio in ratios},
        "unchanged_l2_gap_by_count": {str(count): gap for count, gap in gaps},
        "f1_gap_by_count": {str(count): gap for count, gap in f1_gaps},
        # The W4 prediction, checked rather than eyeballed. Report all three: the ratio is
        # the statistic the published tables used, but it is noise-dominated (see above), so
        # the gap and the raw dense error are what the claim should rest on.
        "mechanism_ratio_grows_with_count": (
            bool(ratios[-1][1] > ratios[0][1]) if len(ratios) >= 2 else None
        ),
        "mechanism_ratio_monotone": monotone(ratios),
        "mechanism_gap_grows_with_count": (
            bool(gaps[-1][1] > gaps[0][1]) if len(gaps) >= 2 else None
        ),
        "mechanism_gap_monotone": monotone(gaps),
        "dense_unchanged_error_monotone": monotone(dense_unchanged),
        "f1_gap_monotone": monotone(f1_gaps),
        "f1_gap_grows_with_count": (
            bool(f1_gaps[-1][1] > f1_gaps[0][1]) if len(f1_gaps) >= 2 else None
        ),
    }


COLUMNS = ["object_count", "seed", "model", "f1", "overall_l2", "changed_l2",
           "unchanged_l2", "margin_over_noop", "changed_object_fraction"]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "scale_analysis.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts = sorted({row["object_count"] for row in rows})
    md = [
        "# Scaling at fixed geometry (W4)",
        "",
        "Bounds +-0.26 and separation 0.09 for **every** count, so N is the only variable",
        "(the published series changed geometry at N=8). Density rises with N, which is the",
        "axis the 'copying unchanged objects compounds' prediction is about.",
        "",
        "| N | changed frac | sparse F1 | dense F1 | sparse unchanged L2 | dense unchanged L2 | ratio | sparse overall | no-op overall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for count in counts:
        def cell(model: str, metric: str) -> str:
            values = [r[metric] for r in rows if r["object_count"] == count and r["model"] == model]
            return f"{np.mean(values):.4f}" if values else "—"

        sparse_unchanged = [r["unchanged_l2"] for r in rows
                            if r["object_count"] == count and r["model"] == "sparse"]
        dense_unchanged = [r["unchanged_l2"] for r in rows
                           if r["object_count"] == count and r["model"] == "dense"]
        ratio = (f"{np.mean(dense_unchanged) / max(np.mean(sparse_unchanged), 1e-12):.1f}x"
                 if sparse_unchanged and dense_unchanged else "—")
        fractions = [r["changed_object_fraction"] for r in rows if r["object_count"] == count]
        md.append(
            f"| {count} | {np.mean(fractions):.3f} | {cell('sparse', 'f1')} | {cell('dense', 'f1')} | "
            f"{cell('sparse', 'unchanged_l2')} | {cell('dense', 'unchanged_l2')} | {ratio} | "
            f"{cell('sparse', 'overall_l2')} | {cell('no_op', 'overall_l2')} |"
        )
    (output_dir / "scale_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
