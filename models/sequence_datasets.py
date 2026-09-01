"""Contiguous-window dataset for multi-step (rollout) training.

Every metric in RESULTS.md up to the rollout section scores a *single* one-step
prediction, and the models are trained that way too. But the world-model use -- and the
CEM planner in particular -- consumes an autoregressive rollout, where the model's own
output becomes its next input. A model trained only on ground-truth inputs never sees the
drifted states it will actually be queried on. This dataset supplies the contiguous
windows needed to close that gap.

**Contiguity is detected from the data, not from ``done``.** ``create_hard_subset`` keeps
only steps with real motion, which punches holes in the middle of episodes while leaving
the ``done`` flags of the surviving rows untouched -- so a ``done``-based reader would
happily stitch a window across a discontinuity. We instead require ``s_t1[i] == s_t[i+1]``,
which is true exactly when row ``i+1`` genuinely follows row ``i``.

This matters in practice: the 3-object hard training split has 1129 rows but only 517
adjacent pairs and 56 windows of length 5, because the filter is aggressive. Rollout
training should therefore run on splits of the *unfiltered* dataset (250 episodes,
mean length 26), where windows of length 20 are plentiful. ``window_statistics`` is
provided so callers can check what they actually got rather than assume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .layout import StateLayout, infer_num_objects_from_state_dim

# Tolerance for declaring two float32 state rows identical. States are metre-scale, so
# 1e-6 is far below any real motion but comfortably above float32 round-trip noise.
CONTIGUITY_ATOL = 1e-6


def contiguous_run_bounds(
    state: np.ndarray, next_state: np.ndarray, atol: float = CONTIGUITY_ATOL
) -> list[tuple[int, int]]:
    """Return ``[start, stop)`` index pairs for each maximal run of consecutive transitions."""
    if state.shape != next_state.shape:
        raise ValueError("state and next_state must have the same shape.")
    num_rows = state.shape[0]
    if num_rows == 0:
        return []
    linked = np.all(np.isclose(next_state[:-1], state[1:], atol=atol, rtol=0.0), axis=1)

    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(num_rows - 1):
        if not linked[index]:
            runs.append((start, index + 1))
            start = index + 1
    runs.append((start, num_rows))
    return runs


class TransitionSequenceDataset(Dataset):
    """Yields length-``horizon`` windows of consecutive transitions.

    Each item stacks the per-step tensors along a leading horizon axis, so a batch has
    shape ``(batch, horizon, ...)``. ``horizon=1`` reproduces the one-step dataset's
    content (every row becomes its own window), which makes it safe to use this dataset
    for both regimes and keeps the training code paths from diverging.
    """

    def __init__(
        self,
        path: str | Path,
        horizon: int = 1,
        stride: int = 1,
        max_windows: int | None = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if stride < 1:
            raise ValueError("stride must be at least 1.")

        data = np.load(Path(path))
        self.state = torch.from_numpy(data["s_t"]).float()
        self.action = torch.from_numpy(data["a_t"]).float()
        self.next_state = torch.from_numpy(data["s_t1"]).float()
        self.object_change_mask = torch.from_numpy(data["object_change_mask"]).float()
        self.object_delta = torch.from_numpy(data["object_delta"]).float()

        self.num_objects = int(self.object_change_mask.shape[1])
        inferred = infer_num_objects_from_state_dim(int(self.state.shape[1]))
        if inferred != self.num_objects:
            raise ValueError(
                f"Dataset inconsistency: state implies {inferred} objects but mask has {self.num_objects}."
            )
        self.layout = StateLayout(num_objects=self.num_objects)
        self.horizon = horizon

        self.runs = contiguous_run_bounds(data["s_t"], data["s_t1"])
        starts: list[int] = []
        for run_start, run_stop in self.runs:
            last_start = run_stop - horizon
            if last_start < run_start:
                continue
            starts.extend(range(run_start, last_start + 1, stride))
        if not starts:
            raise ValueError(
                f"No contiguous windows of length {horizon} in {path}. "
                f"Longest run is {max((stop - start) for start, stop in self.runs) if self.runs else 0} "
                "steps -- use a shorter horizon or an unfiltered dataset."
            )
        if max_windows is not None:
            starts = starts[:max_windows]
        self.window_starts = torch.tensor(starts, dtype=torch.long)

    def window_statistics(self) -> dict[str, float | int]:
        lengths = np.array([stop - start for start, stop in self.runs], dtype=np.int64)
        return {
            "num_rows": int(self.state.shape[0]),
            "num_runs": int(lengths.size),
            "max_run_length": int(lengths.max()) if lengths.size else 0,
            "mean_run_length": float(lengths.mean()) if lengths.size else 0.0,
            "horizon": self.horizon,
            "num_windows": int(self.window_starts.numel()),
        }

    def __len__(self) -> int:
        return int(self.window_starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.window_starts[index])
        window = slice(start, start + self.horizon)
        pose_slice = self.layout.object_pose_slice
        return {
            "state": self.state[window],
            "action": self.action[window],
            "next_state": self.next_state[window],
            "current_object_pose": self.state[window][:, pose_slice],
            "next_object_pose": self.next_state[window][:, pose_slice],
            "object_change_mask": self.object_change_mask[window],
            "object_delta": self.object_delta[window],
        }
