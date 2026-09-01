"""Tests for the standalone audit battery.

The battery is the paper's deliverable, so the thing worth pinning is not that it runs but
that it *detects degeneracy it is pointed at* and *reports what it could not run*. A battery
that silently skips its velocity rules on a benchmark with no velocity channel would hand
somebody a clean bill of health it never earned, which is the exact failure mode the whole
line of work exists to expose.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.audit_battery import (
    NEEDS_SPEED,
    build_report,
    fit_radii,
    load_project,
    mask_metrics,
    score,
    trivial_rules,
)


def synthetic(rows: int = 200, count: int = 4, seed: int = 0) -> dict:
    """A benchmark whose label is exactly "the object nearest the actuator".

    Degenerate by construction, so the battery must find it. If `nearest_to_pusher` does not
    score 1.0 here, the rule is wrong rather than the benchmark being hard.
    """
    rng = np.random.default_rng(seed)
    object_xy = rng.uniform(-0.25, 0.25, size=(rows, count, 2))
    actuator = rng.uniform(-0.25, 0.25, size=(rows, 2))
    distance = np.linalg.norm(object_xy - actuator[:, None, :], axis=-1)
    target = np.zeros((rows, count), dtype=np.float32)
    target[np.arange(rows), np.argmin(distance, axis=1)] = 1.0
    return {
        "object_xy": object_xy,
        "actuator_xy": actuator,
        "actuator_xy_next": actuator,
        "object_speed": rng.uniform(0.0, 0.01, size=(rows, count)),
        "target_mask": target,
    }


def test_finds_the_shortcut_it_is_pointed_at():
    data = synthetic()
    rules = trivial_rules(data, radius=0.05, rest_speed=0.0)
    assert mask_metrics(rules["nearest_to_pusher"], data["target_mask"])["f1"] == pytest.approx(1.0)


def test_always_change_is_the_degeneracy_floor():
    """`always_change` must be present on every benchmark: a filter that raises the positive
    rate can make the degenerate baseline competitive, which is its own kind of degeneracy
    and was missed once already on the interaction benchmark."""
    data = synthetic()
    rules = trivial_rules(data, radius=0.05, rest_speed=0.0)
    metrics = mask_metrics(rules["always_change"], data["target_mask"])
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0 / 4, abs=1e-6)


def test_missing_inputs_skip_rules_rather_than_scoring_them_zero():
    data = synthetic()
    del data["object_speed"]
    rules = trivial_rules(data, radius=0.05, rest_speed=0.0)
    for name in NEEDS_SPEED:
        assert name not in rules, f"{name} ran without a velocity channel"
    # The position-only rules still work, so a benchmark shipping only poses gets a real audit.
    assert "nearest_to_pusher" in rules


def test_report_says_degenerate_when_the_model_loses():
    data = synthetic()
    rules = trivial_rules(data, radius=0.05, rest_speed=0.0)
    rows = [{"rule": name, "params": 0, "radius": None, **score(prediction, data, 0.0)}
            for name, prediction in rules.items()]
    weak = np.zeros_like(data["target_mask"])
    weak[:, 0] = 1.0
    model = {"rule": "model", "params": None, "radius": None, **score(weak, data, 0.0)}
    report = build_report(rows, model, {}, {"rows": 200, "count": 4, "positive_rate": 0.25})
    assert "DEGENERATE" in report
    assert "nearest_to_pusher" in report


def test_report_flags_a_marginal_win_rather_than_calling_it_a_pass():
    rows = [{"rule": "nearest_to_pusher", "params": 0, "radius": None,
             "f1": 0.90, "onset_f1": 0.90, "precision": 0.9, "recall": 0.9}]
    model = {"rule": "model", "params": None, "radius": None,
             "f1": 0.92, "onset_f1": 0.92, "precision": 0.92, "recall": 0.92}
    report = build_report(rows, model, {}, {"rows": 200, "count": 4, "positive_rate": 0.25})
    assert "MARGINAL" in report
    assert "SURVIVES" not in report


def test_skipped_rules_are_named_in_the_report():
    rows = [{"rule": "always_change", "params": 0, "radius": None,
             "f1": 0.3, "onset_f1": float("nan"), "precision": 0.3, "recall": 1.0}]
    report = build_report(rows, None, {"already_moving": "needs `object_speed`"},
                          {"rows": 10, "count": 3, "positive_rate": 0.3})
    assert "unrun" in report
    assert "already_moving" in report


def test_radius_is_fitted_on_the_split_it_is_given():
    """The fitted radius has to come from validation. A rule tuned on test would be the
    mirror image of the mistake the battery exists to detect."""
    data = synthetic()
    radii = np.round(np.arange(0.02, 0.221, 0.01), 4)
    fitted = fit_radii(data, radii, rest_speed=0.0)
    assert set(fitted) >= {"pusher_near", "moving_or_near"}
    assert all(radii[0] <= value <= radii[-1] for value in fitted.values())


def test_agrees_with_the_experiment_on_a_real_split():
    """The standalone battery and `onset_shortcut_audit` must produce identical rules.

    They are separate implementations -- one generic, one wired to `StateLayout` -- and the
    generic one is what other people will run. If they disagree, the published numbers and
    the deliverable are measuring different things.
    """
    from pathlib import Path

    from experiments.compare_phase4_models import load_dataset
    from experiments.onset_shortcut_audit import trivial_rules as experiment_rules

    split = Path("data/transitions/splits_onset_5obj_s0/onset_5obj_s0_hard_test.npz")
    if not split.exists():
        pytest.skip("onset splits not generated in this checkout")

    generic = load_project(split, count=5)
    theirs = experiment_rules(load_dataset(split), count=5, radius=0.07)
    ours = trivial_rules(generic, radius=0.07, rest_speed=2.55e-05)

    assert set(theirs) == set(ours)
    for name in theirs:
        np.testing.assert_allclose(ours[name], theirs[name], atol=1e-6,
                                   err_msg=f"rule `{name}` differs between implementations")
