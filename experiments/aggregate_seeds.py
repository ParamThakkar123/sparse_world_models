"""Aggregate multi-seed Phase-4 results into paper-ready tables.

Reads experiments/runs/phase4_{N}obj_s{seed}/results_table.csv for N in {3,5,8},
seeds in {0,1,2}, computes mean +/- std per (object_count, model), and emits:

  experiments/paper_tables/main_results.md / .csv   -- reframed: change-detection first
  experiments/paper_tables/efficiency.md            -- params / FLOPs / latency
  experiments/paper_tables/seed_dispersion.json     -- raw per-seed values
  experiments/paper_tables/sparsity_ablation.md     -- consolidated existing sweep

Primary metrics (reframed per the scaling analysis): change-detection F1, precision,
recall, and changed-object L2. Overall/unchanged L2 and the no-op margin are secondary.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

RUNS = Path("experiments/runs")
OUT = Path("experiments/paper_tables")
COUNTS = [3, 5, 8]
SEEDS = [0, 1, 2]
MODELS = ["sparse", "dense", "no_op"]

# Optional variant label (e.g. "dense"); "" = main sparse-scene runs.
VARIANT = ""
for i, a in enumerate(sys.argv):
    if a == "--variant" and i + 1 < len(sys.argv):
        VARIANT = sys.argv[i + 1]


def run_dir(n: int, seed: int) -> Path:
    tag = f"{n}obj" + (f"_{VARIANT}" if VARIANT else "") + f"_s{seed}"
    return RUNS / f"phase4_{tag}"


def suffix() -> str:
    return f"_{VARIANT}" if VARIANT else ""

METRIC_KEYS = [
    "overall_per_object_l2",
    "changed_object_l2",
    "unchanged_object_l2",
    "precision",
    "recall",
    "f1",
    "num_parameters",
    "flops_per_forward_estimate",
    "avg_inference_latency_ms",
]


def read_run(n: int, seed: int) -> dict[str, dict[str, float]]:
    path = run_dir(n, seed) / "results_table.csv"
    table: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            table[row["model"]] = {k: float(row[k]) for k in METRIC_KEYS}
    return table


def mean_std(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def collect() -> dict:
    data: dict = {}
    for n in COUNTS:
        per_seed = {seed: read_run(n, seed) for seed in SEEDS}
        agg: dict = {}
        for model in MODELS:
            agg[model] = {}
            for key in METRIC_KEYS:
                vals = [per_seed[s][model][key] for s in SEEDS]
                m, sd = mean_std(vals)
                agg[model][key] = {"mean": m, "std": sd, "seeds": vals}
        # no-op trivially-wins check (per seed) and margin
        margins = [per_seed[s]["no_op"]["overall_per_object_l2"] - per_seed[s]["sparse"]["overall_per_object_l2"] for s in SEEDS]
        agg["_meta"] = {
            "no_op_margin_over_sparse": {"mean": mean_std(margins)[0], "std": mean_std(margins)[1], "seeds": margins},
            "no_op_trivially_wins_any_seed": any(m <= 0 for m in margins),
        }
        data[n] = agg
    return data


def fmt(cell: dict, prec: int = 3) -> str:
    return f"{cell['mean']:.{prec}f} ± {cell['std']:.{prec}f}"


def write_main(data: dict) -> None:
    lines = [
        "# Main Results — Sparse vs Dense vs No-op (mean ± std over seeds 0/1/2)",
        "",
        "**Primary metrics** (change detection + changed-object error). Overall/unchanged L2 are secondary "
        "(overall L2 is dominated by unchanged objects and flatters no-op as scenes get sparser).",
        "",
        "| N | model | F1 | precision | recall | changed-obj L2 | overall L2 | unchanged L2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in COUNTS:
        for model in MODELS:
            a = data[n][model]
            lines.append(
                f"| {n} | {model} | {fmt(a['f1'])} | {fmt(a['precision'])} | {fmt(a['recall'])} | "
                f"{fmt(a['changed_object_l2'])} | {fmt(a['overall_per_object_l2'])} | {fmt(a['unchanged_object_l2'])} |"
            )
    lines += ["", "### No-op margin on overall L2 (sparse advantage; shrinks as scenes get sparser)", "",
              "| N | no-op minus sparse overall L2 | any seed no-op wins? |", "| --- | --- | --- |"]
    for n in COUNTS:
        m = data[n]["_meta"]["no_op_margin_over_sparse"]
        lines.append(f"| {n} | {m['mean']:.5f} ± {m['std']:.5f} | {data[n]['_meta']['no_op_trivially_wins_any_seed']} |")
    (OUT / f"main_results{suffix()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(OUT / f"main_results{suffix()}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_objects", "model"] + [f"{k}_{stat}" for k in METRIC_KEYS for stat in ("mean", "std")])
        for n in COUNTS:
            for model in MODELS:
                row = [n, model]
                for k in METRIC_KEYS:
                    row += [data[n][model][k]["mean"], data[n][model][k]["std"]]
                w.writerow(row)


def write_efficiency(data: dict) -> None:
    lines = [
        "# Efficiency — Sparse vs Dense (mean ± std over seeds 0/1/2)",
        "",
        "Parameter efficiency is the durable win. FLOP advantage erodes with N (per-object heads scale with "
        "object count); wall-clock latency favors dense (one matmul vs per-object gating), so latency is reported "
        "for transparency, not as a claim.",
        "",
        "| N | sparse params | dense params | param ratio | sparse FLOPs | dense FLOPs | FLOP ratio | sparse lat (ms) | dense lat (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in COUNTS:
        s, d = data[n]["sparse"], data[n]["dense"]
        pr = d["num_parameters"]["mean"] / s["num_parameters"]["mean"]
        fr = d["flops_per_forward_estimate"]["mean"] / s["flops_per_forward_estimate"]["mean"]
        lines.append(
            f"| {n} | {s['num_parameters']['mean']:.0f} | {d['num_parameters']['mean']:.0f} | {pr:.1f}× | "
            f"{s['flops_per_forward_estimate']['mean']:.0f} | {d['flops_per_forward_estimate']['mean']:.0f} | {fr:.1f}× | "
            f"{fmt(s['avg_inference_latency_ms'], 2)} | {fmt(d['avg_inference_latency_ms'], 2)} |"
        )
    (OUT / f"efficiency{suffix()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dispersion(data: dict) -> None:
    dump = {}
    for n in COUNTS:
        dump[n] = {model: {k: data[n][model][k] for k in METRIC_KEYS} for model in MODELS}
        dump[n]["_meta"] = data[n]["_meta"]
    (OUT / f"seed_dispersion{suffix()}.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")


def write_sparsity_ablation() -> None:
    """Consolidate the clean 3-obj seed-0 sparsity-weight ablation (ablation_sparsity_sw*)."""
    weights = [("0.0", "sw0p0"), ("0.05", "sw0p05"), ("0.2", "sw0p2"), ("0.5", "sw0p5"), ("1.0", "sw1p0")]
    rows = []
    for sw, tag in weights:
        summ = RUNS / f"ablation_sparsity_{tag}" / "summary.json"
        if summ.exists():
            rows.append((sw, json.loads(summ.read_text())))
    if not rows:
        return
    lines = ["# Sparsity-weight ablation (3 obj, seed 0, standard 15-epoch config)", "",
             "All rows share the seed-0 hard split and identical config; only `sparsity_weight` varies. "
             "Metrics are best-epoch validation values. The model is robust across weights; 0.2 is the main-run choice.", "",
             "| sparsity weight | gate F1 | gate precision | gate recall | changed-obj L2 | overall pose L2 | unchanged L2 | best epoch |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def g(d, k):
        return d.get(k)
    for sw, d in rows:
        lines.append(
            f"| {sw} | {g(d,'best_val_gate_f1'):.3f} | {g(d,'best_val_gate_precision'):.3f} | "
            f"{g(d,'best_val_gate_recall'):.3f} | {g(d,'best_val_changed_pose_l2'):.3f} | "
            f"{g(d,'best_val_pose_l2'):.3f} | {g(d,'best_val_unchanged_pose_l2'):.4f} | {g(d,'best_epoch')} |"
        )
    (OUT / "sparsity_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    write_main(data)
    write_efficiency(data)
    write_dispersion(data)
    if not VARIANT:
        write_sparsity_ablation()
    print(f"Wrote paper tables (variant={VARIANT or 'main'}) to", OUT)
    for n in COUNTS:
        s = data[n]["sparse"]
        meta = data[n]["_meta"]
        print(f"N={n}: sparse F1={fmt(s['f1'])}  changedL2={fmt(s['changed_object_l2'])}  "
              f"overallL2={fmt(s['overall_per_object_l2'])}  noop-margin={meta['no_op_margin_over_sparse']['mean']:.5f}")


if __name__ == "__main__":
    main()
