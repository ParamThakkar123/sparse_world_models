from __future__ import annotations

import torch

from experiments.compositional_generalization import split_path, write_csv
from experiments.train_sparse_model import (
    build_object_features,
    build_object_features_by_mode,
    build_object_features_invariant,
)
from models import StateLayout


def _random_state(num_objects: int, batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    state_dim = StateLayout(num_objects=num_objects).state_dim
    return torch.randn(batch, state_dim), torch.randn(batch, 2)


def test_invariant_feature_dim_is_constant_across_counts() -> None:
    dims = set()
    for num_objects in (3, 5, 8):
        state, action = _random_state(num_objects)
        features = build_object_features_invariant(state, action)
        assert features.shape[:2] == (4, num_objects)
        dims.add(int(features.shape[-1]))
    # The whole point: one fixed per-object width for every object count.
    assert dims == {20}


def test_global_feature_dim_grows_with_count() -> None:
    # Contrast: the default global featurizer is count-dependent (15 + 3N).
    widths = {}
    for num_objects in (3, 5, 8):
        state, action = _random_state(num_objects)
        widths[num_objects] = int(build_object_features(state, action).shape[-1])
    assert widths == {3: 24, 5: 30, 8: 39}


def test_feature_mode_dispatch_matches_direct_builders() -> None:
    state, action = _random_state(5)
    torch.testing.assert_close(
        build_object_features_by_mode(state, action, "global"),
        build_object_features(state, action),
    )
    torch.testing.assert_close(
        build_object_features_by_mode(state, action, "invariant"),
        build_object_features_invariant(state, action),
    )


def test_feature_mode_rejects_unknown_mode() -> None:
    state, action = _random_state(3)
    try:
        build_object_features_by_mode(state, action, "bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an unknown feature mode.")


def test_invariant_features_are_permutation_equivariant() -> None:
    # Reordering the objects must reorder the per-object features identically -- the
    # neighbour aggregate must not depend on object indexing.
    num_objects = 5
    layout = StateLayout(num_objects=num_objects)
    state, action = _random_state(num_objects, batch=1)
    base = build_object_features_invariant(state, action)[0]

    perm = [2, 0, 4, 1, 3]
    permuted = state.clone()
    pose = state[:, layout.object_pose_slice].reshape(1, num_objects, 3)[:, perm].reshape(1, -1)
    velocity = state[:, layout.object_velocity_slice].reshape(1, num_objects, 6)[:, perm].reshape(1, -1)
    permuted[:, layout.object_pose_slice] = pose
    permuted[:, layout.object_velocity_slice] = velocity

    permuted_features = build_object_features_invariant(permuted, action)[0]
    torch.testing.assert_close(permuted_features, base[perm], atol=1e-5, rtol=1e-4)


def test_split_path_matches_canonical_layout() -> None:
    path = split_path(8, 0, "test")
    assert path.as_posix() == "data/transitions/splits_8obj_s0/scale_8obj_s0_hard_test.npz"


def test_write_csv_blanks_nan_and_records_error(tmp_path) -> None:
    rows = [
        {
            "train_count": 3,
            "test_count": 5,
            "model": "dense",
            "f1": float("nan"),
            "accuracy": float("nan"),
            "changed_object_l2": float("nan"),
            "transferable": False,
            "error": "RuntimeError",
        },
        {
            "train_count": None,
            "test_count": 5,
            "model": "no_op",
            "f1": 0.0,
            "accuracy": 0.7,
            "changed_object_l2": 0.44,
            "transferable": True,
        },
    ]
    path = tmp_path / "matrix.csv"
    write_csv(rows, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    # NaNs render as empty fields; the caught error type is preserved.
    assert lines[1] == "3,5,dense,,,,False,RuntimeError"
    # A None train_count (no-op has no training count) renders as an empty field.
    assert lines[2] == ",5,no_op,0.000000,0.700000,0.440000,True,"
