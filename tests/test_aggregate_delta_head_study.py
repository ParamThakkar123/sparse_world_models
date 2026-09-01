"""Tests for the multi-seed delta-head aggregator.

This script produces the numbers that would go in a paper, so its failure modes matter more
than most: silently double-weighting a seed, or averaging cells that should not be pooled.
"""

from __future__ import annotations

import csv

import pytest

from experiments.aggregate_delta_head_study import aggregate, load_rows

BASE = {
    "object_count": 3.0, "head": "mdn", "regime": "onestep",
    "changed_l2_margin_over_noop": 0.05, "oracle_gate_changed_l2": 0.36,
    "changed_l2": 0.36, "overall_l2": 0.12, "f1": 0.86, "detection_gap": 0.001,
}


def _row(seed: float, **overrides) -> dict:
    return {**BASE, "seed": seed, **overrides}


def test_aggregate_reports_mean_and_std_over_seeds() -> None:
    rows = [
        _row(0, changed_l2_margin_over_noop=0.04),
        _row(1, changed_l2_margin_over_noop=0.05),
        _row(2, changed_l2_margin_over_noop=0.06),
    ]
    [entry] = aggregate(rows)
    assert entry["n_seeds"] == 3
    assert entry["seeds"] == [0, 1, 2]
    assert entry["changed_l2_margin_over_noop_mean"] == pytest.approx(0.05)
    assert entry["changed_l2_margin_over_noop_std"] == pytest.approx(0.008164, abs=1e-5)


def test_aggregate_rejects_duplicate_seeds() -> None:
    """Merging two runs that both cover seed 0 must fail loudly, not average it twice."""
    with pytest.raises(RuntimeError, match="Duplicate seeds"):
        aggregate([_row(0), _row(0), _row(1)])


def test_aggregate_keeps_cells_separate() -> None:
    rows = [
        _row(0),
        _row(0, head="mse"),
        _row(0, regime="rollout"),
        _row(0, object_count=5.0),
    ]
    entries = aggregate(rows)
    assert len(entries) == 4
    assert all(entry["n_seeds"] == 1 for entry in entries)


def test_aggregate_tolerates_metrics_present_only_for_some_heads() -> None:
    """Only the MDN emits mode_vs_mean_gain; its absence elsewhere must not crash."""
    rows = [_row(0, mode_vs_mean_gain=0.001), _row(1, head="mse")]
    entries = aggregate(rows)
    by_head = {entry["head"]: entry for entry in entries}
    assert "mode_vs_mean_gain_mean" in by_head["mdn"]
    assert "mode_vs_mean_gain_mean" not in by_head["mse"]


def test_load_rows_skips_blank_cells_and_keeps_labels(tmp_path) -> None:
    run_dir = tmp_path / "a_run"
    run_dir.mkdir()
    path = run_dir / "delta_head_study.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object_count", "seed", "head", "regime", "f1", "mode_vs_mean_gain"])
        writer.writerow([3, 0, "mse", "onestep", 0.87, ""])  # blank: mse has no mixture
    [row] = load_rows(tmp_path, ["a_run"])
    assert row["head"] == "mse"
    assert row["f1"] == pytest.approx(0.87)
    assert "mode_vs_mean_gain" not in row


def test_load_rows_names_the_missing_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="unfinished"):
        load_rows(tmp_path, ["unfinished"])
