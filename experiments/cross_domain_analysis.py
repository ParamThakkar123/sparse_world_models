"""Aggregate the cross-domain shortcut study into one table, with intervals.

Consumes what ``experiments/cross_domain_pipeline.sh`` produces -- four domains on four
engines, two benchmarks (motion and onset) built from the same episodes, three object counts,
five seeds -- and answers the two questions the whole study exists for:

  1. **Does the trivial "already moving" rule beat the learned models on the standard
     (motion-filtered) benchmark in every domain?** If it does on four independent engines,
     the shortcut is a property of physical pushing benchmarks rather than of our simulators,
     and the critique generalises beyond this project.

  2. **Does the corrected (onset-filtered) benchmark reverse that in every domain?** If it
     does, the fix generalises too -- which is what makes this a benchmark contribution
     rather than only a criticism.

Reported with bootstrap confidence intervals on the *difference* between conditions rather
than overlapping per-condition error bars, and with the exact paired sign-flip p-value and
its attainable floor. See ``experiments/statistics.py`` for why that is the right treatment
at these sample sizes.

Usage::

    python -m experiments.cross_domain_analysis --domains tabletop planar billiards clutter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import compute_mask_metrics, load_dataset
from experiments.momentum_shortcut import planar_speed, predicted_mask
from experiments.statistics import bootstrap_ci, paired_permutation_test

REST_SPEED = 2.55e-05
DOMAINS = ("tabletop", "planar", "billiards", "clutter")
ENGINES = {
    "tabletop": "MuJoCo", "planar": "ours",
    "billiards": "Box2D", "clutter": "Chipmunk2D",
}
CONDITIONS = ("trivial_already_moving", "gate_global", "gate_contact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the cross-domain shortcut study.")
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--benchmarks", nargs="+", default=["motion", "onset"])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tag-prefix", type=str, default="xd")
    parser.add_argument("--rest-speed", type=float, default=REST_SPEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="cross_domain_analysis")
    return parser.parse_args()


def paths_for(
    args: argparse.Namespace, domain: str, benchmark: str, count: int, seed: int
) -> tuple[Path, Path, Path]:
    prefix = args.tag_prefix
    split = Path(
        f"data/transitions/splits_{prefix}{benchmark}_{domain}_{count}obj_s{seed}"
        f"/{prefix}_{domain}_{count}obj_s{seed}_hard_test.npz"
    )
    checkpoints = Path("models/checkpoints")
    tag = f"{benchmark}_{domain}_{count}obj_s{seed}"
    return split, checkpoints / f"{prefix}_sparse_{tag}.pt", checkpoints / f"{prefix}_sparse_contact_{tag}.pt"


def score_cell(
    split_path: Path,
    global_checkpoint: Path,
    contact_checkpoint: Path,
    count: int,
    rest_speed: float,
    device: torch.device,
) -> dict[str, dict[str, float]] | None:
    if not split_path.exists():
        return None
    dataset = load_dataset(split_path)
    target = dataset["target_mask"]
    at_rest = planar_speed(dataset["state"], count) <= rest_speed

    predictions: dict[str, np.ndarray] = {
        # The whole baseline: one line, no parameters, no training.
        "trivial_already_moving": (~at_rest).astype(np.float32),
    }
    for name, checkpoint in (("gate_global", global_checkpoint),
                             ("gate_contact", contact_checkpoint)):
        if checkpoint.exists():
            predictions[name] = predicted_mask(checkpoint, dataset, device)

    moving = ~at_rest
    result: dict[str, dict[str, float]] = {}
    for name, prediction in predictions.items():
        result[name] = {
            "f1": compute_mask_metrics(prediction, target)["f1"],
            "onset_f1": (
                compute_mask_metrics(prediction[at_rest], target[at_rest])["f1"]
                if at_rest.any() else float("nan")
            ),
            "f1_moving": (
                compute_mask_metrics(prediction[moving], target[moving])["f1"]
                if moving.any() else float("nan")
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config(vars(args))

    rows: list[dict] = []
    missing: list[str] = []
    for domain in args.domains:
        for benchmark in args.benchmarks:
            for count in args.counts:
                for seed in args.seeds:
                    split, global_ckpt, contact_ckpt = paths_for(
                        args, domain, benchmark, count, seed
                    )
                    scored = score_cell(
                        split, global_ckpt, contact_ckpt, count, args.rest_speed, device
                    )
                    if scored is None:
                        missing.append(str(split))
                        continue
                    for condition, metrics in scored.items():
                        rows.append({
                            "domain": domain, "benchmark": benchmark, "object_count": count,
                            "seed": seed, "condition": condition, **metrics,
                        })

    summary = build_summary(rows)
    summary["missing_cells"] = len(missing)
    summary["missing_examples"] = missing[:5]
    write_outputs(rows, summary, logger.run_dir)
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def _values(rows: list[dict], **filters) -> np.ndarray:
    """Metric values matching every filter, ordered by (count, seed) so pairing lines up."""
    metric = filters.pop("metric")
    selected = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    selected.sort(key=lambda r: (r["object_count"], r["seed"]))
    return np.asarray([r[metric] for r in selected], dtype=np.float64)


def build_summary(rows: list[dict]) -> dict:
    per_domain: dict[str, dict] = {}
    for domain in sorted({r["domain"] for r in rows}):
        per_benchmark: dict[str, dict] = {}
        for benchmark in sorted({r["benchmark"] for r in rows if r["domain"] == domain}):
            entry: dict[str, object] = {"engine": ENGINES.get(domain, "?")}
            for condition in CONDITIONS:
                values = _values(rows, metric="f1", domain=domain,
                                 benchmark=benchmark, condition=condition)
                if values.size == 0:
                    continue
                mean, low, high = bootstrap_ci(values)
                onset = _values(rows, metric="onset_f1", domain=domain,
                                benchmark=benchmark, condition=condition)
                entry[condition] = {
                    "f1": mean, "f1_ci": [low, high],
                    "onset_f1": float(np.nanmean(onset)) if onset.size else None,
                    "n_cells": int(values.size),
                }

            trivial = _values(rows, metric="f1", domain=domain,
                              benchmark=benchmark, condition="trivial_already_moving")
            learned = _values(rows, metric="f1", domain=domain,
                              benchmark=benchmark, condition="gate_global")
            if trivial.size and trivial.size == learned.size:
                difference_mean, difference_low, difference_high = bootstrap_ci(trivial - learned)
                test = paired_permutation_test(trivial, learned)
                entry["trivial_minus_learned"] = {
                    "mean": difference_mean, "ci": [difference_low, difference_high],
                    "excludes_zero": bool(
                        not (difference_low <= 0.0 <= difference_high)
                        and not np.isnan(difference_low)
                    ),
                    "p_value": test["p_value"], "min_attainable_p": test["min_attainable_p"],
                    # The headline claim, per domain and per benchmark.
                    "trivial_wins": bool(difference_mean > 0.0),
                }
            per_benchmark[benchmark] = entry
        per_domain[domain] = per_benchmark

    def trivial_wins(benchmark: str) -> list[str]:
        return [
            domain for domain, benchmarks in per_domain.items()
            if benchmark in benchmarks
            and isinstance(benchmarks[benchmark].get("trivial_minus_learned"), dict)
            and benchmarks[benchmark]["trivial_minus_learned"]["trivial_wins"]
        ]

    domains_present = sorted(per_domain)
    return {
        "per_domain": per_domain,
        # The two claims the study is built to support, stated as booleans over all domains.
        "trivial_beats_learned_on_motion_in": trivial_wins("motion"),
        "trivial_beats_learned_on_onset_in": trivial_wins("onset"),
        "shortcut_generalises_across_engines": bool(
            domains_present and set(trivial_wins("motion")) == set(domains_present)
        ),
        "correction_generalises_across_engines": bool(
            domains_present and not trivial_wins("onset")
        ),
    }


COLUMNS = ["domain", "benchmark", "object_count", "seed", "condition", "f1", "onset_f1", "f1_moving"]


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "cross_domain.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# The momentum shortcut across four physics engines",
        "",
        "Same pipeline, same metrics, four independent contact solvers. `motion` is the",
        "standard filter; `onset` is the corrected one. Both benchmarks are built from the",
        "SAME generated episodes, so they differ in filter only. F1 is the mean over object",
        "counts and seeds; the difference column carries a bootstrap CI because that is the",
        "quantity the claim is about.",
        "",
        "| domain | engine | benchmark | trivial | gate (global) | gate (contact) | trivial − learned | CI excludes 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for domain, benchmarks in summary["per_domain"].items():
        for benchmark, entry in benchmarks.items():
            def cell(condition: str) -> str:
                value = entry.get(condition)
                return f"{value['f1']:.4f}" if isinstance(value, dict) else "--"

            difference = entry.get("trivial_minus_learned")
            if isinstance(difference, dict):
                gap = f"{difference['mean']:+.4f}"
                excludes = "yes" if difference["excludes_zero"] else "no"
            else:
                gap, excludes = "--", "--"
            md.append(
                f"| {domain} | {entry['engine']} | {benchmark} | "
                f"{cell('trivial_already_moving')} | {cell('gate_global')} | "
                f"{cell('gate_contact')} | {gap} | {excludes} |"
            )
    md += [
        "",
        f"- Trivial rule beats the learned gate on the **motion** benchmark in: "
        f"{', '.join(summary['trivial_beats_learned_on_motion_in']) or 'no domain'}",
        f"- ...and on the **onset** benchmark in: "
        f"{', '.join(summary['trivial_beats_learned_on_onset_in']) or 'no domain'}",
    ]
    (output_dir / "cross_domain.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
