"""Tests for the W4 scaling summary statistics.

These decide a pre-registered claim ("does the mechanism advantage compound with object
count?"), so a bug here would flip a paper conclusion rather than raise an error. The
ratio-vs-gap distinction is the specific thing worth pinning down: the ratio's denominator is
the sparse model's unchanged-object error, which sits near zero, so the ratio is noise-driven
while the gap is not.
"""

from __future__ import annotations

import pytest

from experiments.scale_analysis import build_summary


def _rows(spec: dict[int, dict[str, float]]) -> list[dict]:
    """Build rows from ``{count: {model: unchanged_l2}}`` with everything else held fixed."""
    rows = []
    for count, models in spec.items():
        for model, unchanged in models.items():
            rows.append({
                "object_count": count, "seed": 0, "model": model,
                "f1": 0.8 if model == "sparse" else 0.4,
                "overall_l2": 0.1, "changed_l2": 0.4,
                "unchanged_l2": unchanged,
                "margin_over_noop": 0.0, "changed_object_fraction": 0.2,
            })
    return rows


def test_gap_is_monotone_where_the_ratio_is_not() -> None:
    """The exact pathology seen in the real data, reproduced in miniature.

    Dense error grows steadily while sparse error jitters near zero: the gap rises
    monotonically, the ratio does not.
    """
    rows = _rows({
        3: {"sparse": 0.0011, "dense": 0.271, "no_op": 0.0},
        8: {"sparse": 0.0010, "dense": 0.325, "no_op": 0.0},
        20: {"sparse": 0.0022, "dense": 0.438, "no_op": 0.0},
    })
    summary = build_summary(rows)
    assert summary["mechanism_gap_monotone"] is True
    assert summary["mechanism_gap_grows_with_count"] is True
    assert summary["dense_unchanged_error_monotone"] is True
    # The ratio collapses at N=20 purely because the denominator doubled.
    assert summary["mechanism_ratio_monotone"] is False
    assert summary["mechanism_ratio_grows_with_count"] is False


def test_gap_and_ratio_agree_when_the_denominator_is_stable() -> None:
    rows = _rows({
        3: {"sparse": 0.001, "dense": 0.20, "no_op": 0.0},
        8: {"sparse": 0.001, "dense": 0.30, "no_op": 0.0},
        20: {"sparse": 0.001, "dense": 0.40, "no_op": 0.0},
    })
    summary = build_summary(rows)
    assert summary["mechanism_gap_monotone"] is True
    assert summary["mechanism_ratio_monotone"] is True


def test_gap_detects_a_shrinking_advantage() -> None:
    """A genuine failure of the prediction must be reported as one."""
    rows = _rows({
        3: {"sparse": 0.001, "dense": 0.40, "no_op": 0.0},
        8: {"sparse": 0.001, "dense": 0.30, "no_op": 0.0},
        20: {"sparse": 0.001, "dense": 0.20, "no_op": 0.0},
    })
    summary = build_summary(rows)
    assert summary["mechanism_gap_monotone"] is False
    assert summary["mechanism_gap_grows_with_count"] is False


def test_f1_gap_tracked_separately_from_the_pose_mechanism() -> None:
    rows = []
    for count, (sparse_f1, dense_f1) in {3: (0.87, 0.53), 20: (0.75, 0.19)}.items():
        for model, f1 in (("sparse", sparse_f1), ("dense", dense_f1), ("no_op", 0.0)):
            rows.append({
                "object_count": count, "seed": 0, "model": model, "f1": f1,
                "overall_l2": 0.1, "changed_l2": 0.4, "unchanged_l2": 0.01,
                "margin_over_noop": 0.0, "changed_object_fraction": 0.2,
            })
    summary = build_summary(rows)
    assert summary["f1_gap_grows_with_count"] is True
    assert summary["f1_gap_by_count"]["3"] == pytest.approx(0.34)
    assert summary["f1_gap_by_count"]["20"] == pytest.approx(0.56)


def test_means_and_stds_are_computed_across_seeds() -> None:
    rows = []
    for seed, unchanged in enumerate((0.10, 0.20, 0.30)):
        rows.append({
            "object_count": 3, "seed": seed, "model": "dense", "f1": 0.4,
            "overall_l2": 0.1, "changed_l2": 0.4, "unchanged_l2": unchanged,
            "margin_over_noop": 0.0, "changed_object_fraction": 0.2,
        })
    summary = build_summary(rows)
    entry = summary["per_count"]["3obj"]["dense_unchanged_l2"]
    assert entry["n"] == 3
    assert entry["mean"] == pytest.approx(0.20)
    assert entry["std"] == pytest.approx(0.0816496, abs=1e-6)


def test_single_count_yields_undecidable_rather_than_a_false_claim() -> None:
    """With one count there is no trend; the checks must say so, not default to False."""
    rows = _rows({3: {"sparse": 0.001, "dense": 0.2, "no_op": 0.0}})
    summary = build_summary(rows)
    assert summary["mechanism_gap_monotone"] is None
    assert summary["mechanism_gap_grows_with_count"] is None
