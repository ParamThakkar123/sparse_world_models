"""Tests for the W2 relational baselines and the capacity-matching helpers."""

from __future__ import annotations

import pytest
import torch

from experiments.gate_ablation import BASE_RUNGS, MODEL_ORDER, MODEL_STYLE, split_path
from models.delta_heads import delta_head_parameters, match_delta_hidden_dim
from models.relational import (
    InteractionNetworkPredictor,
    SetTransformerPredictor,
    match_hidden_dim,
)

FEATURE_DIM = 24


def _inputs(batch: int = 4, num_objects: int = 5):
    torch.manual_seed(0)
    return torch.randn(batch, num_objects, FEATURE_DIM), torch.randn(batch, num_objects, 3)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InteractionNetworkPredictor(FEATURE_DIM, hidden_dim=16, message_dim=16),
        lambda: SetTransformerPredictor(FEATURE_DIM, hidden_dim=16, num_heads=4),
    ],
)
def test_relational_models_are_permutation_equivariant(factory) -> None:
    """Object order in the state vector is arbitrary, so predictions must permute with it.

    Without this these are not credible object-centric baselines, and the ablation could
    not attribute anything to the sparse model's structure.
    """
    features, pose = _inputs()
    model = factory().eval()
    permutation = torch.randperm(features.shape[1])
    with torch.no_grad():
        permuted_output = model(features, pose)[:, permutation]
        output_of_permuted = model(features[:, permutation], pose[:, permutation])
    torch.testing.assert_close(permuted_output, output_of_permuted, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "factory",
    [
        lambda mode: InteractionNetworkPredictor(FEATURE_DIM, hidden_dim=16, message_dim=16, mode=mode),
        lambda mode: SetTransformerPredictor(FEATURE_DIM, hidden_dim=16, num_heads=4, mode=mode),
    ],
)
def test_residual_mode_adds_current_pose_and_absolute_ignores_it(factory) -> None:
    features, pose = _inputs()
    absolute = factory("absolute").eval()
    with torch.no_grad():
        torch.testing.assert_close(absolute(features, pose), absolute(features, pose + 100.0))

    residual = factory("residual").eval()
    with torch.no_grad():
        shifted = residual(features, pose + 100.0) - residual(features, pose)
    torch.testing.assert_close(shifted, torch.full_like(shifted, 100.0))


def test_interaction_network_excludes_self_edges() -> None:
    """A self-message would leak the node's own embedding into the 'relational' term.

    With a single object there are no other objects, so the aggregated message must be
    exactly zero -- which shows up as the one-object output being invariant to the edge
    model's parameters.
    """
    torch.manual_seed(0)
    model = InteractionNetworkPredictor(FEATURE_DIM, hidden_dim=8, message_dim=8).eval()
    features, pose = _inputs(batch=2, num_objects=1)
    with torch.no_grad():
        before = model(features, pose)
        for parameter in model.edge_model.parameters():
            parameter.add_(10.0)
        after = model(features, pose)
    torch.testing.assert_close(before, after)


def test_set_transformer_rejects_width_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SetTransformerPredictor(FEATURE_DIM, hidden_dim=18, num_heads=4)


@pytest.mark.parametrize("mode", ["sideways", ""])
def test_relational_models_reject_unknown_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="mode must be"):
        InteractionNetworkPredictor(FEATURE_DIM, hidden_dim=8, mode=mode)
    with pytest.raises(ValueError, match="mode must be"):
        SetTransformerPredictor(FEATURE_DIM, hidden_dim=8, num_heads=4, mode=mode)


def test_match_hidden_dim_lands_close_to_the_target() -> None:
    target = 7000
    width, count = match_hidden_dim(
        lambda hidden_dim: InteractionNetworkPredictor(
            FEATURE_DIM, hidden_dim=hidden_dim, message_dim=hidden_dim
        ),
        target, range(4, 129),
    )
    assert 4 <= width <= 128
    # Within 10%: close enough that no rung wins the ablation on capacity.
    assert abs(count - target) / target < 0.10


def test_match_hidden_dim_skips_invalid_widths() -> None:
    """Widths indivisible by the head count raise; the search must step over them."""
    width, _ = match_hidden_dim(
        lambda hidden_dim: SetTransformerPredictor(FEATURE_DIM, hidden_dim=hidden_dim, num_heads=4),
        7000, range(4, 65),
    )
    assert width % 4 == 0


def test_delta_head_capacity_matching_equalises_parameter_counts() -> None:
    """The MDN is larger at equal width, so the comparison needs the others widened."""
    target = delta_head_parameters("mdn", object_feature_dim=FEATURE_DIM, hidden_dim=128)
    default_mse = delta_head_parameters("mse", object_feature_dim=FEATURE_DIM, hidden_dim=128)
    assert target > default_mse  # the confound this exists to remove

    for head in ("mse", "gaussian"):
        width, count = match_delta_hidden_dim(head, target, object_feature_dim=FEATURE_DIM)
        assert width > 128, f"{head} should need widening to reach the MDN's budget"
        assert abs(count - target) / target < 0.01


def test_new_rungs_have_plot_styles() -> None:
    # A missing style entry crashes plot_results only at the very end of a long run.
    for rung in MODEL_ORDER:
        assert rung in MODEL_STYLE
        assert {"color", "label"} <= MODEL_STYLE[rung].keys()


def test_split_path_template_overrides_the_default_layout() -> None:
    default = split_path(3, 0, "train")
    assert default.as_posix() == "data/transitions/splits_3obj_s0/scale_3obj_s0_hard_train.npz"
    clean = split_path(
        3, 0, "test",
        "data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz",
    )
    assert clean.as_posix() == (
        "data/transitions/splits_clean_3obj_s0/scale_3obj_s0_hard_test.npz"
    )


def test_base_rungs_are_a_subset_of_the_full_ladder() -> None:
    assert set(BASE_RUNGS) <= set(MODEL_ORDER)
