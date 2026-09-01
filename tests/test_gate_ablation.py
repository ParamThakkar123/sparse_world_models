from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from experiments.gate_ablation import (
    BASE_RUNGS,
    EXTENDED_RUNGS,
    MODEL_ORDER,
    ObjectCentricPredictor,
    evaluate_object_centric,
    resolve_hidden_dim,
    split_path,
    write_csv,
)


def _args(width_mode: str = "matched", hidden_dim: int = 128, num_layers: int = 2) -> Namespace:
    """Minimal stand-in for the parsed argparse namespace."""
    return Namespace(width_mode=width_mode, hidden_dim=hidden_dim, num_layers=num_layers)


def test_split_path_matches_canonical_layout() -> None:
    assert split_path(3, 0, "train").as_posix() == "data/transitions/splits_3obj_s0/scale_3obj_s0_hard_train.npz"
    assert split_path(8, 2, "test").as_posix() == "data/transitions/splits_8obj_s2/scale_8obj_s2_hard_test.npz"


def test_ladder_order_runs_from_neither_ingredient_to_both() -> None:
    # The ladder only reads as an ablation if the rungs stay in this order. The W2
    # baselines interleave extra rungs, so the invariant is that the original five still
    # appear in their original relative order, not that they are the only entries.
    assert BASE_RUNGS == ("dense", "oc_absolute", "oc_residual", "sparse", "no_op")
    positions = [MODEL_ORDER.index(rung) for rung in BASE_RUNGS]
    assert positions == sorted(positions)


def test_extended_rungs_place_each_new_baseline_before_sparse() -> None:
    # Every W2 baseline is an alternative the sparse model has to beat, so each must sit
    # to the left of `sparse` for the figure and table to read as a ladder.
    for rung in ("dense_l1", "gnn", "set_transformer", "soft_gate"):
        assert MODEL_ORDER.index(rung) < MODEL_ORDER.index("sparse")
    assert set(EXTENDED_RUNGS) == set(MODEL_ORDER)


def test_residual_mode_adds_to_current_pose_and_absolute_ignores_it() -> None:
    torch.manual_seed(0)
    features = torch.randn(4, 3, 24)
    current_pose = torch.randn(4, 3, 3)

    residual = ObjectCentricPredictor(object_feature_dim=24, hidden_dim=16, num_layers=2, mode="residual")
    absolute = ObjectCentricPredictor(object_feature_dim=24, hidden_dim=16, num_layers=2, mode="absolute")

    raw_residual = residual.mlp(features)
    torch.testing.assert_close(residual(features, current_pose), current_pose + raw_residual)
    # The absolute head must not see current_pose at all, so shifting it changes nothing.
    torch.testing.assert_close(
        absolute(features, current_pose), absolute(features, current_pose + 100.0)
    )


def test_matched_width_lands_close_to_the_sparse_parameter_budget() -> None:
    device = torch.device("cpu")
    for feature_dim in (24, 30, 39):
        hidden_dim, target = resolve_hidden_dim(_args(width_mode="matched"), feature_dim, device)
        model = ObjectCentricPredictor(
            object_feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=2, mode="residual"
        )
        params = sum(p.numel() for p in model.parameters())
        # Capacity matching is the whole point of the ladder; 1% is a generous bound on
        # what the integer width search can achieve.
        assert abs(params - target) / target < 0.01


def test_identical_width_mode_uses_the_requested_width() -> None:
    hidden_dim, _ = resolve_hidden_dim(_args(width_mode="identical", hidden_dim=128), 24, torch.device("cpu"))
    assert hidden_dim == 128


def test_ungated_model_that_moves_nothing_detects_no_change() -> None:
    # A model predicting the current pose verbatim must score zero recall, which is the
    # control that makes the "recall == 1 for every ungated model" finding meaningful.
    num_samples, num_objects = 5, 3
    current_pose = np.random.RandomState(0).randn(num_samples, num_objects, 3).astype(np.float32)
    dataset = {
        "state": np.zeros((num_samples, 2 + num_objects * 9 + 2), dtype=np.float32),
        "action": np.zeros((num_samples, 2), dtype=np.float32),
        "current_pose": current_pose,
        "next_pose": current_pose.copy(),
        "target_mask": np.ones((num_samples, num_objects), dtype=np.float32),
        "target_delta": np.zeros((num_samples, num_objects, 3), dtype=np.float32),
        "num_objects": np.array(num_objects, dtype=np.int32),
    }

    class _NoOp(torch.nn.Module):
        def forward(self, _object_features, current_pose):
            return current_pose

    # State layout for 3 objects: pusher(2) + pose(9) + velocity(18) + goal(2).
    dataset["state"][:, 2 : 2 + num_objects * 3] = current_pose.reshape(num_samples, -1)
    result = evaluate_object_centric(_NoOp(), dataset, torch.device("cpu"), feature_mode="global")
    assert result["mask_metrics"]["recall"] == 0.0
    assert result["pose_metrics"]["overall_per_object_l2"] == 0.0


def test_write_csv_column_order_and_format(tmp_path) -> None:
    rows = [
        {
            "object_count": 3,
            "model": "oc_residual",
            "num_parameters": 6919,
            "f1": 0.538217,
            # Onset F1 restricts to objects at rest: the part of change detection that
            # requires prediction rather than momentum continuation. See
            # experiments/momentum_shortcut.py for why it is reported alongside f1.
            "onset_f1": 0.071000,
            "precision": 0.368192,
            "recall": 1.0,
            "overall_per_object_l2": 0.220800,
            "changed_object_l2": 0.444400,
            "unchanged_object_l2": 0.090500,
        },
    ]
    path = tmp_path / "ladder.csv"
    write_csv(rows, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == (
        "object_count,model,num_parameters,f1,onset_f1,precision,recall,"
        "overall_per_object_l2,changed_object_l2,unchanged_object_l2"
    )
    assert lines[1] == (
        "3,oc_residual,6919,0.538217,0.071000,0.368192,1.000000,0.220800,0.444400,0.090500"
    )
