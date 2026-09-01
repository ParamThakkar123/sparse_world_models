"""Tests for the onset filter, which defines the corrected benchmark.

The motion filter it replaces produced a task a one-line velocity rule wins (F1 0.90 tabletop,
0.99 planar, against learned models' 0.86 and 0.44-0.74). The onset filter keeps only
transitions where a *stationary* object starts moving, which can only happen through contact.
If this selection is wrong, every number on the corrected benchmark is wrong, so the
properties below are asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.create_hard_subset import (
    REST_SPEED,
    compute_keep_mask,
    compute_onset_keep_mask,
)
from models import StateLayout

NUM_OBJECTS = 3
LAYOUT = StateLayout(num_objects=NUM_OBJECTS)


def _dataset(speeds: np.ndarray, displacements: np.ndarray) -> dict[str, np.ndarray]:
    """Build a minimal dataset from per-row (speed, displacement) per object.

    ``speeds`` and ``displacements`` are both ``(rows, num_objects)``. The change mask is
    derived from the displacement so the fixture cannot disagree with itself.
    """
    rows = speeds.shape[0]
    state = np.zeros((rows, LAYOUT.state_dim), dtype=np.float32)
    velocity = np.zeros((rows, NUM_OBJECTS, 6), dtype=np.float32)
    # Columns 3:5 are the planar linear components in the shared velocity layout.
    velocity[:, :, 3] = speeds
    state[:, LAYOUT.object_velocity_slice] = velocity.reshape(rows, -1)

    delta = np.zeros((rows, NUM_OBJECTS, 3), dtype=np.float32)
    delta[:, :, 0] = displacements
    return {
        "s_t": state,
        "object_delta": delta,
        "object_change_mask": (displacements > 1e-6).astype(np.float32),
    }


def test_onset_kept_when_a_resting_object_starts_moving() -> None:
    data = _dataset(
        speeds=np.array([[0.0, 0.0, 0.0]]),
        displacements=np.array([[0.05, 0.0, 0.0]]),
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [True]


def test_continuation_is_rejected() -> None:
    """The case the whole filter exists to exclude: an already-moving object keeps moving."""
    data = _dataset(
        speeds=np.array([[1.0, 0.0, 0.0]]),          # object 0 already in motion
        displacements=np.array([[0.05, 0.0, 0.0]]),  # and it moves
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [False]
    # The motion filter, by contrast, happily keeps it -- which is the bug.
    assert compute_keep_mask(data, 0.02, 1).tolist() == [True]


def test_stationary_and_unmoving_is_rejected() -> None:
    data = _dataset(
        speeds=np.array([[0.0, 0.0, 0.0]]),
        displacements=np.array([[0.0, 0.0, 0.0]]),
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [False]


def test_onset_below_the_displacement_threshold_is_rejected() -> None:
    """A resting object that twitches by less than the threshold is not an onset event."""
    data = _dataset(
        speeds=np.array([[0.0, 0.0, 0.0]]),
        displacements=np.array([[0.005, 0.0, 0.0]]),
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [False]
    assert compute_onset_keep_mask(data, 0.001).tolist() == [True]


def test_one_onset_among_continuations_is_enough() -> None:
    """A realistic frame: something is already sliding while a new object gets struck."""
    data = _dataset(
        speeds=np.array([[1.0, 0.0, 2.0]]),           # objects 0 and 2 already moving
        displacements=np.array([[0.05, 0.04, 0.06]]),  # object 1 was at rest and moved
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [True]


def test_rest_threshold_is_respected() -> None:
    just_moving = REST_SPEED * 2.0
    just_resting = REST_SPEED * 0.5
    data = _dataset(
        speeds=np.array([[just_moving, 0.0, 0.0], [just_resting, 0.0, 0.0]]),
        displacements=np.array([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]),
    )
    assert compute_onset_keep_mask(data, 0.02).tolist() == [False, True]


def test_onset_is_a_strict_subset_of_the_motion_filter() -> None:
    """Every onset transition also has real motion, so onset ⊆ motion must hold exactly.

    Verified on the real dataset rather than a fixture, because this is the invariant that
    lets the two benchmarks be compared as "same data, harder selection".
    """
    from pathlib import Path

    path = Path("data/transitions/scale_3obj_s0.npz")
    if not path.exists():
        pytest.skip(f"{path} not generated")
    data = dict(np.load(path))
    motion = compute_keep_mask(data, 0.02, 1)
    onset = compute_onset_keep_mask(data, 0.02)
    assert not np.any(onset & ~motion)
    # And it must be strictly harder: onset events are a small minority of moving steps.
    assert 0.0 < onset.mean() < motion.mean()


def test_filter_selects_far_fewer_rows() -> None:
    """Quantifies the cost the pipeline compensates for by generating more episodes."""
    rng = np.random.default_rng(0)
    rows = 2000
    speeds = np.where(rng.random((rows, NUM_OBJECTS)) < 0.3, 1.0, 0.0)
    displacements = np.where(rng.random((rows, NUM_OBJECTS)) < 0.35, 0.05, 0.0)
    data = _dataset(speeds, displacements)
    motion = compute_keep_mask(data, 0.02, 1).mean()
    onset = compute_onset_keep_mask(data, 0.02).mean()
    assert onset < motion
