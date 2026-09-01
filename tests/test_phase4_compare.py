from __future__ import annotations

import numpy as np

from experiments.compare_phase4_models import compute_mask_metrics, predicted_change_mask_from_pose_delta


def test_predicted_change_mask_from_pose_delta_matches_thresholds() -> None:
    delta = np.array(
        [
            [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.0, 0.02]],
        ],
        dtype=np.float32,
    )

    pred_mask = predicted_change_mask_from_pose_delta(delta)

    np.testing.assert_array_equal(pred_mask, np.array([[0.0, 1.0, 1.0]], dtype=np.float32))


def test_compute_mask_metrics_handles_all_negative_predictions() -> None:
    pred_mask = np.zeros((2, 3), dtype=np.float32)
    target_mask = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    metrics = compute_mask_metrics(pred_mask, target_mask)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 5 / 6
