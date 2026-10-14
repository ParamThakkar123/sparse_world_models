"""Pin the paper's table generator.

``experiments/make_iclr_tables.py`` is the only path from run artifacts into the manuscript,
so the two things that would silently corrupt the paper are pinned here: that a missing run
degrades to a visible ``--`` rather than to a plausible-looking wrong number, and that
``best_trivial`` never reports a *learned* rung as the best trivial rule -- which is exactly
the mislabelling that reached the abstract once already (see the 2026-09-03 correction in
``experiments/RESULTS.md``).
"""

from __future__ import annotations

import csv
import math

import pytest

from experiments import make_iclr_tables as mit


def _write_audit(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["benchmark", "object_count", "seed", "rule", "fitted_radius",
                        "f1", "onset_f1", "precision", "recall"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(rule, f1, *, benchmark="motion", n=3, seed=0):
    return {
        "benchmark": benchmark, "object_count": n, "seed": seed, "rule": rule,
        "fitted_radius": "nan", "f1": f1, "onset_f1": f1, "precision": f1, "recall": f1,
    }


def test_best_trivial_excludes_learned_rungs():
    """A learned rung must never be reported as the best *trivial* rule."""
    values = {
        ("motion", "already_moving"): 0.80,
        ("motion", "moving_or_near"): 0.90,
        ("motion", "logistic_on_distance"): 0.99,  # two parameters: learned, not trivial
        ("motion", "learned_gate"): 0.995,
        ("motion", "learned_gate_contact"): 0.999,
    }
    rule, value = mit.best_trivial(values, "motion")
    assert rule == "moving_or_near"
    assert value == pytest.approx(0.90)


def test_missing_run_is_visible_not_silent(tmp_path, monkeypatch):
    """An absent run directory yields '--' cells and a zero-cell provenance row."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mit, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(mit, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(mit, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(
        mit, "BENCHMARKS",
        [("motion", "present_run", "motion"), ("interaction", "absent_run", "interaction")],
    )
    _write_audit(
        tmp_path / "runs" / "present_run" / "onset_shortcut_audit.csv",
        [_row("already_moving", 0.8), _row("moving_or_near", 0.9)],
    )

    values, prov = mit.collect_battery()
    mit.write_battery(values, prov)

    assert prov["motion"]["present"] is True
    assert prov["interaction"]["present"] is False
    assert prov["interaction"]["cells"] == 0
    assert math.isnan(mit.best_trivial(values, "interaction")[1])

    battery = (tmp_path / "tables" / "battery.tex").read_text(encoding="utf-8")
    # The absent column must render as '--' on every row, never as a number.
    assert "--" in battery
    provenance = (tmp_path / "tables" / "provenance.tex").read_text(encoding="utf-8")
    assert r"\emph{not run}" in provenance
    # Run names carry underscores; unescaped they abort the LaTeX build.
    assert "present\\_run" in provenance


def test_means_are_over_all_cells(tmp_path, monkeypatch):
    """The reported value is the mean across seeds and object counts, not the first cell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mit, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(mit, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(mit, "BENCHMARKS", [("motion", "r", "motion")])
    _write_audit(
        tmp_path / "runs" / "r" / "onset_shortcut_audit.csv",
        [
            _row("already_moving", 0.6, n=3, seed=0),
            _row("already_moving", 1.0, n=5, seed=1),
        ],
    )
    values, prov = mit.collect_battery()
    assert values[("motion", "already_moving")] == pytest.approx(0.8)
    assert prov["motion"]["cells"] == 2


def test_tex_escape_handles_underscores():
    assert mit.tex_escape("audit_onset_clutter12k") == r"audit\_onset\_clutter12k"
