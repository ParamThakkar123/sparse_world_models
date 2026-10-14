"""Emit the ICLR paper's tables and figure directly from run artifacts.

Every number in ``paper/main.tex`` that is not prose comes from here, so the manuscript
cannot drift from ``experiments/runs/``. Missing runs degrade to ``--`` cells and a warning
on stderr rather than a crash: a stage that has not landed must be visible in the output, not
silently absent (the same rule ``SUBMISSION_RUNBOOK.md`` step 2 applies to the prose).

Outputs (all overwritten):
  paper/tables/battery.tex    -- the audit battery across benchmarks (Table 1)
  paper/tables/published.tex  -- five published models vs. the trivial rule (Table 2)
  paper/tables/provenance.tex -- run directory + cell count behind every column
  paper/figures/pattern.pdf   -- the three-benchmark figure (Figure 1)

Usage:  python -m experiments.make_iclr_tables
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path("experiments/runs")
TABLES = Path("paper/tables")
FIGURES = Path("paper/figures")

# benchmark column -> (run directory, benchmark key inside that run)
BENCHMARKS = [
    ("motion", "onset_shortcut_audit_5seed", "motion"),
    ("onset", "onset_shortcut_audit_5seed", "onset"),
    ("interaction", "audit_interaction_billiards", "interaction"),
    ("clutter-onset", "audit_onset_clutter12k", "onsetclutter12k"),
]

# Rules shown in the main table, in the order the argument needs them, with their parameter
# count. The battery is larger; the appendix prints all of it.
RULES = [
    ("already_moving", "0", r"\texttt{already\_moving} (velocity)"),
    ("nearest_to_pusher", "0", r"\texttt{nearest\_to\_pusher} (geometry)"),
    ("pusher_near", "1", r"\texttt{pusher\_near} (contact radius)"),
    ("near_a_mover", "1", r"\texttt{near\_a\_mover} (propagated)"),
    ("moving_or_near", "1", r"\texttt{moving\_or\_near}"),
    ("moving_or_near_mover", "1", r"\texttt{moving\_or\_near\_mover}"),
    ("logistic_on_distance", "2", r"\texttt{logistic\_on\_distance} (learned)"),
    ("always_change", "0", r"\texttt{always\_change} (degeneracy floor)"),
]

LEARNED = [
    ("learned_gate", r"learned gate (velocity features)"),
    ("learned_gate_contact", r"learned gate (velocity-free)"),
]

PUBLISHED = [
    ("gns", r"GNS / DPI-Net \citep{sanchezgonzalez2020gns,li2019dpinet}"),
    ("cswm", r"C-SWM \citep{kipf2020cswm}"),
    ("slotformer", r"SlotFormer \citep{wu2023slotformer}"),
    ("pets", r"PETS \citep{chua2018pets}"),
    ("nps", r"NPS \citep{goyal2021nps}"),
]


def tex_escape(s: str) -> str:
    """Escape the characters that appear in run names -- underscores, mostly."""
    return s.replace("_", r"\_")


def warn(msg: str) -> None:
    print(f"  WARN {msg}", file=sys.stderr)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x: str | None) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return v


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def fmt(v: float) -> str:
    return "--" if math.isnan(v) else f"{v:.3f}"


# ---------------------------------------------------------------------------------------
# Table 1: the battery
# ---------------------------------------------------------------------------------------


def collect_battery() -> tuple[dict, dict]:
    """Return {(column, rule): mean F1} and {column: provenance dict}."""
    values: dict[tuple[str, str], float] = {}
    prov: dict[str, dict] = {}
    for column, run, key in BENCHMARKS:
        rows = read_csv(RUNS / run / "onset_shortcut_audit.csv")
        if not rows:
            warn(f"{column}: {RUNS / run} absent -- column will be '--'")
            prov[column] = {"run": run, "present": False, "cells": 0, "seeds": [], "counts": []}
            continue
        rows = [r for r in rows if r["benchmark"] == key]
        by_rule: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_rule[r["rule"]].append(fnum(r.get("f1")))
        for rule, xs in by_rule.items():
            values[(column, rule)] = mean(xs)
        cells = {(r["object_count"], r["seed"]) for r in rows}
        prov[column] = {
            "run": run,
            "present": True,
            "cells": len(cells),
            "seeds": sorted({r["seed"] for r in rows}, key=int),
            "counts": sorted({r["object_count"] for r in rows}, key=int),
            "gate_cells": len({(r["object_count"], r["seed"]) for r in rows if r["rule"] == "learned_gate"}),
        }
    return values, prov


def best_trivial(values: dict, column: str) -> tuple[str, float]:
    """Argmax over rules that are actually trivial -- the learned rungs are excluded."""
    excluded = {"learned_gate", "learned_gate_contact", "logistic_on_distance"}
    cands = [
        (rule, v)
        for (col, rule), v in values.items()
        if col == column and rule not in excluded and not math.isnan(v)
    ]
    if not cands:
        return ("--", math.nan)
    return max(cands, key=lambda kv: kv[1])


def write_battery(values: dict, prov: dict) -> None:
    cols = [c for c, _, _ in BENCHMARKS]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"rule & params & motion & onset & interaction & clutter-onset \\",
        r"\midrule",
    ]
    for rule, params, label in RULES:
        cells = [fmt(values.get((c, rule), math.nan)) for c in cols]
        lines.append(f"{label} & {params} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    bt = {c: best_trivial(values, c) for c in cols}
    lines.append(
        r"\textbf{best trivial rule} & & "
        + " & ".join(rf"\textbf{{{fmt(bt[c][1])}}}" for c in cols)
        + r" \\"
    )
    for key, label in LEARNED:
        cells = [fmt(values.get((c, key), math.nan)) for c in cols]
        lines.append(f"{label} & $\\sim$7k & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "battery.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Provenance, so a reviewer can see exactly how many cells each column rests on.
    plines = [
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"benchmark & run directory & cells & seeds $\times$ object counts \\",
        r"\midrule",
    ]
    for c in cols:
        p = prov.get(c, {})
        if not p.get("present"):
            plines.append(
                rf"{c} & \texttt{{{tex_escape(p.get('run','?'))}}} & 0 & \emph{{not run}} \\"
            )
            continue
        seeds = ",".join(p["seeds"])
        counts = ",".join(p["counts"])
        plines.append(
            rf"{c} & \texttt{{{tex_escape(p['run'])}}} & {p['cells']} "
            rf"& seeds {seeds} $\times$ N={counts} \\"
        )
    plines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "provenance.tex").write_text("\n".join(plines) + "\n", encoding="utf-8")

    # Full battery for the appendix: every rule the audit scored, in the order the battery
    # defines them, so a reader can see the ones the main table has no room for.
    all_rules = sorted({rule for (_, rule) in values}
                       - {"learned_gate", "learned_gate_contact"})
    group_of = {
        "already_moving": "kinematic",
        "nearest_to_pusher": "geometric", "second_nearest_to_pusher": "geometric",
        "two_nearest_to_pusher": "geometric", "pusher_near": "geometric",
        "pusher_approaching": "geometric",
        "near_a_mover": "propagated", "moving_or_near_mover": "propagated",
        "near_pusher_or_mover": "propagated",
        # Not propagated: it combines the kinematic and geometric signals by hand and never
        # looks at another object's motion. Grouping it correctly matters, because it is the
        # rule that wins two of the four benchmarks.
        "moving_or_near": "combination",
        "always_change": "floor", "logistic_on_distance": "control",
    }
    order = {"kinematic": 0, "geometric": 1, "propagated": 2, "combination": 3,
             "floor": 4, "control": 5}
    all_rules.sort(key=lambda r: (order.get(group_of.get(r, ""), 9), r))
    flines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"rule & group & motion & onset & interaction & clutter-onset \\", r"\midrule",
    ]
    previous = None
    for rule in all_rules:
        group = group_of.get(rule, "--")
        if previous is not None and group != previous:
            flines.append(r"\midrule")
        previous = group
        label = r"\texttt{" + rule.replace("_", r"\_") + "}"
        flines.append(
            f"{label} & {group} & "
            + " & ".join(fmt(values.get((c, rule), math.nan)) for c in cols) + r" \\"
        )
    flines.append(r"\midrule")
    for key, label in LEARNED:
        flines.append(
            f"{label} & learned & "
            + " & ".join(fmt(values.get((c, key), math.nan)) for c in cols) + r" \\"
        )
    flines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "battery_full.tex").write_text("\n".join(flines) + "\n", encoding="utf-8")

    print("battery.tex / battery_full.tex / provenance.tex written")
    for c in cols:
        rule, v = bt[c]
        p = prov.get(c, {})
        gate = values.get((c, "learned_gate"), math.nan)
        gc = values.get((c, "learned_gate_contact"), math.nan)
        bestgate = max([x for x in (gate, gc) if not math.isnan(x)], default=math.nan)
        print(
            f"  {c:14s} best trivial {rule:22s} {fmt(v)}   best gate {fmt(bestgate)}"
            f"   delta {fmt(bestgate - v) if not math.isnan(v) and not math.isnan(bestgate) else '--'}"
            f"   [{p.get('cells', 0)} cells, gate {p.get('gate_cells', 0)}]"
        )


# ---------------------------------------------------------------------------------------
# Table 2: the five published models
# ---------------------------------------------------------------------------------------


def collect_published() -> tuple[dict, dict]:
    out: dict[tuple[str, str, str], float] = {}
    prov: dict[str, dict] = {}
    for bench, runs in (("motion", ["literature_baselines_motion", "literature_baselines_motion_s34"]),
                        ("onset", ["literature_baselines_onset", "literature_baselines_onset_s34"])):
        rows: list[dict] = []
        found = []
        for run in runs:
            r = read_csv(RUNS / run / "literature_baselines.csv")
            if r:
                rows += r
                found.append(run)
            else:
                warn(f"{bench}: {RUNS / run} absent")
        if not rows:
            prov[bench] = {"runs": [], "cells": 0}
            continue
        by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for metric in ("f1", "onset_f1", "recall", "unchanged_object_l2"):
                by_model[r["model"]][metric].append(fnum(r.get(metric)))
        for model, metrics in by_model.items():
            for metric, xs in metrics.items():
                out[(bench, model, metric)] = mean(xs)
        prov[bench] = {
            "runs": found,
            "cells": len({(r["object_count"], r["seed"]) for r in rows}),
            "seeds": sorted({r["seed"] for r in rows}, key=int),
        }
    return out, prov


def write_published(vals: dict, prov: dict) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{motion benchmark} & \multicolumn{3}{c}{onset benchmark} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"model & F1 & onset F1 & recall & F1 & onset F1 & recall \\",
        r"\midrule",
    ]

    def row(model: str, label: str, bold: bool = False) -> str:
        cells = []
        for bench in ("motion", "onset"):
            for metric in ("f1", "onset_f1", "recall"):
                v = fmt(vals.get((bench, model, metric), math.nan))
                cells.append(rf"\textbf{{{v}}}" if bold else v)
        return f"{label} & " + " & ".join(cells) + r" \\"

    lines.append(row("trivial_velocity", r"\textbf{trivial: already moving}", bold=True))
    lines.append(r"\midrule")
    for model, label in PUBLISHED:
        lines.append(row(model, label))
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "published.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("published.tex written")
    for bench in ("motion", "onset"):
        p = prov.get(bench, {})
        rec = [vals.get((bench, m, "recall"), math.nan) for m, _ in PUBLISHED]
        rec = [r for r in rec if not math.isnan(r)]
        print(
            f"  {bench:8s} cells {p.get('cells', 0):3d}  runs {p.get('runs', [])}"
            f"  recall range {fmt(min(rec)) if rec else '--'}-{fmt(max(rec)) if rec else '--'}"
        )


# ---------------------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------------------


def make_figure(values: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warn("matplotlib unavailable -- figure skipped")
        return

    cols = [c for c, _, _ in BENCHMARKS]
    present = [c for c in cols if not math.isnan(best_trivial(values, c)[1])]
    if not present:
        warn("no audit data -- figure skipped")
        return

    # Okabe-Ito, colourblind-safe. The hatching is not decoration: converted to greyscale for
    # print, these three map to 87, 162 and 106 of 255, so velocity and propagated are only 19
    # levels apart and colour alone does not separate them.
    series = [
        ("already_moving", "velocity rule", "#0072B2", "///"),
        ("nearest_to_pusher", "geometry rule (0 params)", "#E69F00", ""),
        ("moving_or_near_mover", "propagated rule", "#009E73", "xxx"),
    ]
    x = np.arange(len(present), dtype=float)
    width = 0.2

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (rule, label, colour, hatch) in enumerate(series):
        ys = [values.get((c, rule), math.nan) for c in present]
        ax.bar(x + (i - 1) * width, ys, width, label=label, color=colour,
               edgecolor="black", linewidth=0.4, hatch=hatch)

    bt = [best_trivial(values, c)[1] for c in present]
    ax.plot(x, bt, "k--o", markersize=5, linewidth=1.5, label="best rule in the battery", zorder=5)
    for xi, v in zip(x, bt):
        if not math.isnan(v):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, fontweight="bold")

    gates = []
    for c in present:
        g = [values.get((c, k), math.nan) for k, _ in LEARNED]
        g = [v for v in g if not math.isnan(v)]
        gates.append(max(g) if g else math.nan)
    ax.plot(x, gates, color="#D55E00", marker="s", markersize=5, linewidth=1.5,
            label="best learned gate", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel("change-detection F1")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.93, color="grey", linewidth=0.6, linestyle=":", zorder=0)
    # Below the axes: with four benchmark groups an inset legend covers the bars it explains.
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              frameon=False, columnspacing=1.2, handletextpad=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "pattern.pdf")
    fig.savefig(FIGURES / "pattern.png", dpi=220)
    print("pattern.pdf / pattern.png written")


def main() -> None:
    print("== battery")
    values, prov = collect_battery()
    write_battery(values, prov)
    print("== published models")
    pub, pprov = collect_published()
    write_published(pub, pprov)
    print("== figure")
    make_figure(values)
    (TABLES / "generated.json").write_text(
        json.dumps({"battery_provenance": prov, "published_provenance": pprov}, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
