from __future__ import annotations

from experiments.oracle_gate_diagnostic import split_path as oracle_split_path
from experiments.param_matched_baseline import split_path as pm_split_path
from experiments.param_matched_baseline import write_csv


def test_split_paths_match_canonical_layout() -> None:
    assert pm_split_path(3, 0, "train").as_posix() == "data/transitions/splits_3obj_s0/scale_3obj_s0_hard_train.npz"
    assert oracle_split_path(8, 0, "test").as_posix() == "data/transitions/splits_8obj_s0/scale_8obj_s0_hard_test.npz"


def test_param_matched_write_csv_row_order_and_format(tmp_path) -> None:
    rows = [
        {
            "object_count": 3,
            "model": "sparse",
            "num_parameters": 6916,
            "f1": 0.884615,
            "accuracy": 0.921569,
            "changed_object_l2": 0.312053,
            "overall_per_object_l2": 0.115741,
        },
        {
            "object_count": 3,
            "model": "dense_matched",
            "num_parameters": 6921,
            "f1": 0.538217,
            "accuracy": 0.368192,
            "changed_object_l2": 0.580785,
            "overall_per_object_l2": 0.363422,
        },
    ]
    path = tmp_path / "pm.csv"
    write_csv(rows, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "object_count,model,num_parameters,f1,accuracy,changed_object_l2,overall_per_object_l2"
    assert lines[1] == "3,sparse,6916,0.884615,0.921569,0.312053,0.115741"
    assert lines[2] == "3,dense_matched,6921,0.538217,0.368192,0.580785,0.363422"
