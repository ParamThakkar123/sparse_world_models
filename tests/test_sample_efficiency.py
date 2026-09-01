from __future__ import annotations

import numpy as np

from experiments.sample_efficiency import (
    count_samples,
    fraction_tag,
    infer_object_count,
    write_csv,
)


def test_fraction_tag_zero_pads_percent() -> None:
    assert fraction_tag(0.1) == "f010"
    assert fraction_tag(0.25) == "f025"
    assert fraction_tag(1.0) == "f100"


def _write_toy_npz(path, num_samples: int, num_objects: int) -> None:
    # State layout: pusher(2) + pose(3N) + velocity(6N) + goal(2).
    state_dim = 2 + 3 * num_objects + 6 * num_objects + 2
    np.savez(
        path,
        s_t=np.zeros((num_samples, state_dim), dtype=np.float32),
        a_t=np.zeros((num_samples, 2), dtype=np.float32),
        s_t1=np.zeros((num_samples, state_dim), dtype=np.float32),
        object_change_mask=np.zeros((num_samples, num_objects), dtype=np.float32),
        object_delta=np.zeros((num_samples, num_objects, 3), dtype=np.float32),
        done=np.zeros((num_samples,), dtype=bool),
    )


def test_count_samples_and_infer_object_count(tmp_path) -> None:
    path = tmp_path / "toy.npz"
    _write_toy_npz(path, num_samples=17, num_objects=5)
    assert count_samples(path) == 17
    assert infer_object_count(path) == 5


def test_write_csv_emits_header_and_row_per_model(tmp_path) -> None:
    rows = [
        {
            "fraction": 0.5,
            "num_train_samples": 100,
            "models": {
                "sparse": {
                    "f1": 0.8,
                    "accuracy": 0.9,
                    "precision": 0.95,
                    "recall": 0.7,
                    "changed_object_l2": 0.3,
                    "overall_per_object_l2": 0.1,
                },
                "dense": {
                    "f1": 0.5,
                    "accuracy": 0.6,
                    "precision": 0.4,
                    "recall": 1.0,
                    "changed_object_l2": 0.4,
                    "overall_per_object_l2": 0.3,
                },
                "no_op": {
                    "f1": 0.0,
                    "accuracy": 0.7,
                    "precision": 0.0,
                    "recall": 0.0,
                    "changed_object_l2": 0.5,
                    "overall_per_object_l2": 0.15,
                },
            },
        }
    ]
    path = tmp_path / "curves.csv"
    write_csv(rows, object_count=3, path=path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert lines[0].startswith("object_count,fraction,num_train_samples,model,f1")
    # One data row per model.
    assert len(lines) == 1 + 3
    sparse_row = next(line for line in lines[1:] if ",sparse," in line)
    assert sparse_row.startswith("3,0.5000,100,sparse,0.800000")
