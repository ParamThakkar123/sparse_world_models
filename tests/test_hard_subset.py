from __future__ import annotations

import numpy as np

from experiments.create_hard_subset import compute_keep_mask, filter_episode_indices


def test_compute_keep_mask_enforces_motion_and_count() -> None:
    dataset = {
        'object_delta': np.array([
            [[0.0, 0.0, 0.0], [0.010, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.030, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.030, 0.0, 0.0], [0.025, 0.0, 0.0]],
        ], dtype=np.float32),
        'object_change_mask': np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float32),
    }

    keep = compute_keep_mask(dataset, min_max_xy_delta=0.02, min_changed_objects=2)

    np.testing.assert_array_equal(keep, np.array([False, False, True]))


def test_filter_episode_indices_splits_discontiguous_kept_steps() -> None:
    keep_mask = np.array([False, True, True, False, True], dtype=bool)
    chunks = filter_episode_indices(type('Episode', (), {'start': 0, 'end': 5})(), keep_mask)

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], np.array([1, 2]))
    np.testing.assert_array_equal(chunks[1], np.array([4]))
