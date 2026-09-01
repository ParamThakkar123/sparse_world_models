"""Merge delta-head study runs into one multi-seed table.

The W1 grid was executed in two batches -- seed 0 first (to decide whether the direction
was worth more compute), seeds 1 and 2 afterwards -- so the per-seed rows live in separate
``experiments/runs/<name>/delta_head_study.csv`` files. This merges them and reports
mean ± std per (count, head, regime), which is the only form any of it should be quoted in.

Single-seed numbers are load-bearing nowhere in the paper, and the capacity control in
particular showed swings of ~0.03 in changed-object L2 between widths whose validation
scores were identical -- i.e. comfortably inside seed noise. Treat any cell with ``n < 3``
in the output as provisional.

Usage::

    python -m experiments.aggregate_delta_head_study \\
      --runs delta_head_study_v2 delta_head_study_seeds12
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Metrics worth an error bar. `changed_l2_margin_over_noop` is the headline: positive means
# the delta head beats declaring the object stationary.
METRICS = (
    "changed_l2_margin_over_noop",
    "oracle_gate_changed_l2",
    "changed_l2",
    "overall_l2",
    "f1",
    "detection_gap",
    "mode_vs_mean_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate delta-head study runs across seeds.")
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out", type=str, default="delta_head_study_multiseed")
    return parser.parse_args()


def load_rows(runs_dir: Path, run_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for name in run_names:
        path = runs_dir / name / "delta_head_study.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- has '{name}' finished?")
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed: dict = {"run": name}
                for key, value in row.items():
                    if value == "":
                        continue
                    if key in {"head", "regime", "capacity_matched"}:
                        parsed[key] = value
                    else:
                        parsed[key] = float(value)
                rows.append(parsed)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (int(row["object_count"]), row["head"], row["regime"])
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for (count, head, regime), cell in sorted(grouped.items()):
        seeds = sorted({int(r["seed"]) for r in cell})
        if len(seeds) != len(cell):
            raise RuntimeError(
                f"Duplicate seeds for {count}obj/{head}/{regime}: {[int(r['seed']) for r in cell]}. "
                "Two runs covering the same seed would silently double-weight it."
            )
        entry: dict = {
            "object_count": count, "head": head, "regime": regime,
            "n_seeds": len(seeds), "seeds": seeds,
        }
        for metric in METRICS:
            values = [r[metric] for r in cell if metric in r]
            if not values:
                continue
            entry[f"{metric}_mean"] = float(np.mean(values))
            # Population std over the seeds actually run; with n=3 this is indicative, not
            # a confidence interval.
            entry[f"{metric}_std"] = float(np.std(values))
        out.append(entry)
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.runs_dir, args.runs)
    aggregated = aggregate(rows)

    output_dir = args.runs_dir / args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    header = ["object_count", "head", "regime", "n_seeds"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "std")
    ]
    lines = [",".join(header)]
    for entry in aggregated:
        lines.append(",".join(
            f"{entry[c]:.6f}" if isinstance(entry.get(c), float) else str(entry.get(c, ""))
            for c in header
        ))
    (output_dir / "delta_head_multiseed.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Delta-head study — multi-seed",
        "",
        f"Merged from: {', '.join(args.runs)}. `margin` = no-op − oracle-gate changed-object L2;",
        "**positive means the delta head beats declaring the object stationary.** Cells with",
        "n_seeds < 3 are provisional.",
        "",
        "| N | head | regime | seeds | margin (mean ± std) | oracle L2 | F1 | mode−mean gain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in aggregated:
        gain = (
            f"{e['mode_vs_mean_gain_mean']:+.4f} ± {e['mode_vs_mean_gain_std']:.4f}"
            if "mode_vs_mean_gain_mean" in e else "—"
        )
        md.append(
            f"| {e['object_count']} | {e['head']} | {e['regime']} | {e['n_seeds']} | "
            f"**{e['changed_l2_margin_over_noop_mean']:+.4f} ± {e['changed_l2_margin_over_noop_std']:.4f}** | "
            f"{e['oracle_gate_changed_l2_mean']:.4f} | {e['f1_mean']:.3f} | {gain} |"
        )
    (output_dir / "delta_head_multiseed.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # The two claims worth checking automatically rather than by eye.
    beats_noop = {
        f"{e['object_count']}obj_{e['head']}_{e['regime']}":
            bool(e["changed_l2_margin_over_noop_mean"] - e["changed_l2_margin_over_noop_std"] > 0)
        for e in aggregated
    }
    mdn_gains = [e for e in aggregated if "mode_vs_mean_gain_mean" in e]
    summary = {
        "runs": args.runs,
        "cells": len(aggregated),
        # Margin exceeds one std above zero -- a weak but honest significance proxy at n=3.
        "beats_noop_by_more_than_one_std": beats_noop,
        "mode_vs_mean_is_null": bool(
            all(abs(e["mode_vs_mean_gain_mean"]) < e["mode_vs_mean_gain_std"] + 0.005 for e in mdn_gains)
        ) if mdn_gains else None,
        "csv": str(output_dir / "delta_head_multiseed.csv"),
        "md": str(output_dir / "delta_head_multiseed.md"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
