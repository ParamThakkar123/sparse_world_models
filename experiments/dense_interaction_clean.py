"""Dense-interaction control, rebuilt on the episode-disjoint clean splits.

Overall per-object L2 has a known weakness in sparse scenes: as the object count rises, the
fraction of objects that genuinely move falls, so "predict no change" becomes a strong
baseline and the sparse model's margin over it shrinks toward zero. On the leaky splits
sparse and no-op *tied* at N=8 (0.071 vs 0.071); on clean splits sparse is marginally
**behind** (0.0741 vs 0.0739). The published answer is the dense-interaction variant --
objects packed at generation so pushes cascade into neighbours, roughly doubling the
multi-object-change fraction -- which restored the margin to ~12x at N=8. That variant was
built through the leaky pipeline, so it has to be re-measured here before the N=8 column can
be used for anything.

This script reports, per count and seed, the sparse margin over no-op on overall L2 in both
regimes. The question it answers is narrow and specific: **does the margin still fail to
decay with object count once the scene is dense in interactions and the splits are clean?**

Requires (via ``VARIANT=dense bash experiments/clean_pipeline.sh N SEED``):
``models/checkpoints/{sparse,dense}_clean_{N}obj_dense_s{S}.pt``.

Usage::

    python -m experiments.dense_interaction_clean --counts 3 5 8 --seeds 0 1 2
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

REGIMES = ("sparse_scene", "dense_interaction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense-interaction control on clean splits.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="dense_interaction_clean")
    return parser.parse_args()


def paths(count: int, seed: int, regime: str) -> tuple[Path, Path, Path]:
    tag = f"{count}obj_s{seed}" if regime == "sparse_scene" else f"{count}obj_dense_s{seed}"
    data = Path(f"data/transitions/splits_clean_{tag}/scale_{tag}_hard_test.npz")
    return (
        data,
        Path(f"models/checkpoints/sparse_clean_{tag}.pt"),
        Path(f"models/checkpoints/dense_clean_{tag}.pt"),
    )


def score(count: int, seed: int, regime: str, device: torch.device) -> dict:
    data_path, sparse_path, dense_path = paths(count, seed, regime)
    for path in (data_path, sparse_path, dense_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (regime '{regime}')")

    dataset = load_dataset(data_path)
    sparse_model, sparse_config = load_sparse_model(sparse_path, device)
    dense_model, _ = load_dense_model(dense_path, device)

    sparse_eval = evaluate_sparse(sparse_model, sparse_config, dataset, device, 256)
    dense_eval = evaluate_dense(dense_model, dataset, device, 256)
    noop_eval = evaluate_noop(dataset)

    sparse_l2 = sparse_eval["pose_metrics"]["overall_per_object_l2"]
    noop_l2 = noop_eval["pose_metrics"]["overall_per_object_l2"]
    target_mask = dataset["target_mask"]
    return {
        "object_count": count, "seed": seed, "regime": regime,
        "sparse_overall_l2": sparse_l2,
        "dense_overall_l2": dense_eval["pose_metrics"]["overall_per_object_l2"],
        "noop_overall_l2": noop_l2,
        # The quantity the control exists to protect: positive means the sparse model is
        # still worth more than "predict no change".
        "margin_over_noop": noop_l2 - sparse_l2,
        "sparse_f1": sparse_eval["mask_metrics"]["f1"],
        # How interaction-dense the scene actually is, so the manipulation is verifiable
        # rather than asserted.
        "multi_change_fraction": float((target_mask.sum(axis=1) >= 2).mean()),
        "changed_object_fraction": float(target_mask.mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "dense_interaction_clean", "counts": args.counts, "seeds": args.seeds})

    rows, skipped = [], []
    for count in args.counts:
        for seed in args.seeds:
            for regime in REGIMES:
                try:
                    rows.append(score(count, seed, regime, device))
                except FileNotFoundError as error:
                    skipped.append(str(error))

    write_outputs(rows, output_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))
    if skipped:
        print(f"\nSkipped {len(skipped)}:", *skipped, sep="\n  ")


def build_summary(rows: list[dict]) -> dict:
    per_count: dict[str, dict] = {}
    for count in sorted({row["object_count"] for row in rows}):
        entry: dict[str, object] = {}
        for regime in REGIMES:
            margins = [r["margin_over_noop"] for r in rows
                       if r["object_count"] == count and r["regime"] == regime]
            multi = [r["multi_change_fraction"] for r in rows
                     if r["object_count"] == count and r["regime"] == regime]
            if margins:
                entry[f"{regime}_margin"] = {
                    "mean": float(np.mean(margins)), "std": float(np.std(margins)), "n": len(margins)
                }
                entry[f"{regime}_multi_change_fraction"] = float(np.mean(multi))
                entry[f"{regime}_margin_positive_every_seed"] = bool(all(m > 0 for m in margins))
        per_count[f"{count}obj"] = entry

    dense_positive = [
        entry.get("dense_interaction_margin_positive_every_seed")
        for entry in per_count.values()
        if "dense_interaction_margin_positive_every_seed" in entry
    ]
    return {
        "per_count": per_count,
        # The control succeeds only if the margin stays positive at EVERY count under dense
        # interaction -- including N=8, where the sparse-scene margin goes negative.
        "dense_interaction_margin_positive_at_every_count": (
            bool(all(dense_positive)) if dense_positive else None
        ),
    }


COLUMNS = [
    "object_count", "seed", "regime", "sparse_overall_l2", "dense_overall_l2",
    "noop_overall_l2", "margin_over_noop", "sparse_f1",
    "multi_change_fraction", "changed_object_fraction",
]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "dense_interaction_clean.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Dense-interaction control, clean splits",
        "",
        "`margin` = no-op − sparse on overall per-object L2. **Positive means the sparse model",
        "is worth more than predicting no change.** The sparse-scene margin decays with object",
        "count and goes negative at N=8; the question is whether packing the scene restores it.",
        "",
        "| N | seed | regime | sparse | dense | no-op | margin | F1 | multi-change frac |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['object_count']} | {row['seed']} | {row['regime']} | "
            f"{row['sparse_overall_l2']:.4f} | {row['dense_overall_l2']:.4f} | "
            f"{row['noop_overall_l2']:.4f} | {row['margin_over_noop']:+.4f} | "
            f"{row['sparse_f1']:.3f} | {row['multi_change_fraction']:.3f} |"
        )
    (output_dir / "dense_interaction_clean.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
