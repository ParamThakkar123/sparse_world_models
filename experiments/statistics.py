"""Uncertainty and significance for the project's per-seed result tables.

Why this exists
---------------
Almost every table in RESULTS.md reports ``mean +/- std`` over 3 or 5 seeds, and conclusions
are drawn by eye from whether the intervals appear to overlap. That is weaker than it looks,
in two specific ways this module fixes.

**1. "+/- std over 3 seeds" is not a confidence interval.** With n=3 the sample standard
deviation is itself extremely noisy, and non-overlapping +/-1 std bars are neither necessary
nor sufficient for a difference to be real. Worse, the informal rule "the bars don't touch,
so it's significant" is conservative for overlapping bars and anti-conservative for the
comparison people actually care about. Bootstrap intervals over the seed distribution are
reported instead, and the *difference* between conditions gets its own interval, which is
the quantity the claim is about.

**2. Seeds are paired, and unpaired tests throw that away.** Two conditions trained at seed 0
share the data split, the initialisation stream and the episode set. Comparing their marginal
distributions ignores that pairing and loses most of the power a small seed budget has. A
paired permutation test over sign flips is exact for the null "the two conditions are
exchangeable within each seed", makes no normality assumption, and is valid at n=3 -- where a
t-test's assumptions are unverifiable.

With 3 seeds a sign-flip permutation test has only 2^3 = 8 possible outcomes, so the smallest
attainable two-sided p-value is 0.25. That is a hard floor, not a defect of the test, and it
is reported explicitly (``min_attainable_p``) so that "p = 0.25 at n=3" is never mistaken for
a null result. It is also the strongest possible argument for the 5-seed runs: at n=5 the
floor drops to 0.0625, and at n=6 to 0.03125.

Everything here operates on the per-seed CSVs the experiments already write, so no run has to
be repeated to get intervals on it.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

DEFAULT_BOOTSTRAP = 10000


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    iterations: int = DEFAULT_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """``(mean, low, high)`` percentile bootstrap interval over the seed distribution.

    Percentile rather than BCa: with 3-5 seeds the bias-correction and acceleration terms are
    estimated from the same handful of points they are meant to correct, which adds variance
    without adding accuracy. The interval is honestly wide at small n and should be.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float("nan"), float("nan")
    generator = rng or np.random.default_rng(0)
    samples = generator.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def paired_permutation_test(
    first: np.ndarray, second: np.ndarray
) -> dict[str, float]:
    """Exact two-sided paired sign-flip permutation test on ``first - second``.

    Enumerates all ``2^n`` sign assignments when ``n <= 20`` -- which every table here is --
    so the p-value is exact rather than sampled. Returns the observed mean difference, the
    p-value, and the smallest p-value this ``n`` could possibly produce.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("paired test needs one value per condition per seed.")
    difference = first - second
    difference = difference[~np.isnan(difference)]
    n = difference.size
    if n == 0:
        return {"mean_difference": float("nan"), "p_value": float("nan"),
                "min_attainable_p": float("nan"), "n": 0}

    observed = abs(difference.mean())
    if n <= 20:
        extreme = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            if abs((difference * np.asarray(signs)).mean()) >= observed - 1e-12:
                extreme += 1
        p_value = extreme / (2 ** n)
    else:  # pragma: no cover - no table here is this large
        generator = np.random.default_rng(0)
        signs = generator.choice((-1.0, 1.0), size=(DEFAULT_BOOTSTRAP, n))
        p_value = float((np.abs((difference * signs).mean(axis=1)) >= observed - 1e-12).mean())

    return {
        "mean_difference": float(difference.mean()),
        "p_value": float(p_value),
        # With n seeds the all-same-sign assignment is the most extreme, and two of the 2^n
        # assignments achieve it, so no result can be significant below 2/2^n.
        "min_attainable_p": float(2.0 / (2 ** n)),
        "n": int(n),
    }


def summarise_comparison(
    values_by_condition: dict[str, np.ndarray],
    reference: str,
    confidence: float = 0.95,
) -> dict[str, dict]:
    """Bootstrap CI per condition, plus a paired test of each against ``reference``."""
    if reference not in values_by_condition:
        raise KeyError(f"reference condition '{reference}' not present.")
    baseline = values_by_condition[reference]
    result: dict[str, dict] = {}
    for condition, values in values_by_condition.items():
        mean, low, high = bootstrap_ci(values, confidence)
        entry: dict[str, object] = {"mean": mean, "ci_low": low, "ci_high": high,
                                    "n_seeds": int(np.asarray(values).size)}
        if condition != reference:
            entry.update(paired_permutation_test(np.asarray(values), np.asarray(baseline)))
            difference_mean, difference_low, difference_high = bootstrap_ci(
                np.asarray(values) - np.asarray(baseline), confidence
            )
            entry["difference_ci"] = [difference_low, difference_high]
            entry["difference_mean"] = difference_mean
            # The interval on the DIFFERENCE is the one the claim rests on. Two overlapping
            # marginal intervals are perfectly compatible with a difference interval that
            # excludes zero, which is exactly why the pairing is worth keeping.
            entry["difference_excludes_zero"] = bool(
                not (difference_low <= 0.0 <= difference_high)
                and not np.isnan(difference_low)
            )
        result[condition] = entry
    return result


# --------------------------------------------------------------------------- CLI

def load_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs and paired permutation tests for a per-seed results CSV."
    )
    parser.add_argument("csv", type=Path, help="A per-seed results CSV written by an experiment.")
    parser.add_argument("--condition-column", type=str, required=True,
                        help="Column naming the condition, e.g. 'model' or 'condition'.")
    parser.add_argument("--metric", type=str, required=True, help="Numeric column to analyse.")
    parser.add_argument("--seed-column", type=str, default="seed")
    parser.add_argument("--group-column", type=str, default=None,
                        help="Analyse separately within each value of this column, e.g. "
                             "'object_count'. Omit to pool.")
    parser.add_argument("--reference", type=str, required=True,
                        help="Condition every other is compared against.")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.csv)

    groups = ([None] if args.group_column is None
              else sorted({row[args.group_column] for row in rows}))
    report: dict[str, dict] = {}
    for group in groups:
        selected = [r for r in rows
                    if group is None or r[args.group_column] == group]
        by_condition: dict[str, list[tuple[str, float]]] = {}
        for row in selected:
            try:
                value = float(row[args.metric])
            except ValueError:
                continue
            by_condition.setdefault(row[args.condition_column], []).append(
                (row[args.seed_column], value)
            )
        # Sort by seed so the pairing across conditions lines up seed-for-seed. Without this
        # the "paired" test would pair arbitrary rows and silently become nonsense.
        aligned = {
            condition: np.asarray([value for _, value in sorted(entries)])
            for condition, entries in by_condition.items()
        }
        lengths = {len(v) for v in aligned.values()}
        if len(lengths) > 1:
            print(f"  WARNING group={group}: unequal seed counts {lengths}; "
                  "paired tests are skipped for conditions that do not match the reference.")
        usable = {c: v for c, v in aligned.items()
                  if args.reference in aligned and len(v) == len(aligned[args.reference])}
        report[str(group)] = summarise_comparison(usable, args.reference, args.confidence)

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
