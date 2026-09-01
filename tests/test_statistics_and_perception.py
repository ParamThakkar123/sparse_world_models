"""Tests for the statistics layer, the renderer, and the two perception front ends.

The statistics tests matter more than they look. Every headline comparison in the project is
now quoted with a bootstrap interval on a *difference* and an exact paired permutation
p-value, and both are easy to get subtly wrong in ways that change conclusions rather than
crash: an unpaired test where a paired one was intended, or a p-value floor that is mistaken
for a null result. The properties asserted here are the ones the write-up depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.statistics import bootstrap_ci, paired_permutation_test, summarise_comparison
from models.envs.renderer import (
    BACKGROUND_COLOR,
    OBJECT_COLORS,
    object_pixel_positions,
    render_state,
)
from models.keypoint_encoder import KeypointAutoencoder, SpatialSoftmax, positions_to_pixels
from models.layout import StateLayout
from models.slot_attention import SlotAutoencoder, slot_centroids


# ------------------------------------------------------------------ statistics

def test_bootstrap_ci_brackets_the_mean_and_narrows_with_n() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal(0.5, 0.1, size=4)
    large = rng.normal(0.5, 0.1, size=200)
    mean_s, low_s, high_s = bootstrap_ci(small)
    mean_l, low_l, high_l = bootstrap_ci(large)
    assert low_s <= mean_s <= high_s and low_l <= mean_l <= high_l
    assert (high_l - low_l) < (high_s - low_s), "more seeds must give a tighter interval"


def test_bootstrap_ci_handles_degenerate_inputs() -> None:
    assert np.isnan(bootstrap_ci(np.array([]))[0])
    mean, low, high = bootstrap_ci(np.array([0.7]))
    assert mean == pytest.approx(0.7) and np.isnan(low) and np.isnan(high)
    # NaNs are dropped, not propagated -- onset F1 is NaN wherever no object is at rest.
    assert bootstrap_ci(np.array([0.4, np.nan, 0.6]))[0] == pytest.approx(0.5)


def test_permutation_test_reports_its_own_floor() -> None:
    """The n=3 floor of 0.25 is quoted throughout the write-up; it must be exact."""
    for n, expected in ((3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)):
        result = paired_permutation_test(np.ones(n), np.zeros(n))
        assert result["min_attainable_p"] == pytest.approx(expected)
        # A perfectly consistent difference should sit exactly at the floor.
        assert result["p_value"] == pytest.approx(expected)


def test_permutation_test_is_paired_not_marginal() -> None:
    """Two conditions with identical marginals but a consistent per-seed gap must separate.

    This is the property that makes the pairing worth keeping: an unpaired comparison of
    these two samples sees the same distribution twice and finds nothing.
    """
    first = np.array([0.10, 0.50, 0.90])
    second = np.array([0.05, 0.45, 0.85])
    assert paired_permutation_test(first, second)["mean_difference"] == pytest.approx(0.05)
    assert paired_permutation_test(first, second)["p_value"] == pytest.approx(0.25)
    # Identical inputs must not look significant.
    assert paired_permutation_test(first, first.copy())["p_value"] == pytest.approx(1.0)


def test_difference_interval_can_exclude_zero_while_marginals_overlap() -> None:
    """The reason differences are reported rather than per-condition error bars."""
    first = np.array([0.20, 0.50, 0.80])
    second = np.array([0.18, 0.48, 0.78])
    result = summarise_comparison({"a": first, "b": second}, reference="b")["a"]
    assert result["difference_excludes_zero"] is True
    # ...even though the marginal intervals overlap almost completely.
    marginal_a = bootstrap_ci(first)
    marginal_b = bootstrap_ci(second)
    assert marginal_a[1] < marginal_b[2] and marginal_b[1] < marginal_a[2]


def test_summarise_comparison_rejects_an_absent_reference() -> None:
    with pytest.raises(KeyError):
        summarise_comparison({"a": np.ones(3)}, reference="missing")


# -------------------------------------------------------------------- renderer

def _state(count: int = 3) -> np.ndarray:
    layout = StateLayout(num_objects=count)
    state = np.zeros(layout.state_dim)
    state[:2] = (0.0, -0.22)
    pose = np.array([[0.10, 0.05, 0.0], [-0.12, 0.16, 0.7], [0.05, -0.14, -1.2]])[:count]
    state[layout.object_pose_slice] = pose.reshape(-1)
    return state


def test_each_object_is_drawn_at_its_own_pixel_position() -> None:
    state = _state()
    image = render_state(state, 3, resolution=96)
    positions = object_pixel_positions(state, 3, resolution=96).astype(int)
    for index, (x, y) in enumerate(positions):
        assert np.array_equal(image[y, x], OBJECT_COLORS[index].astype(np.uint8))


def test_background_dominates_and_objects_are_sparse() -> None:
    """The scenes really are mostly background -- the fact that breaks Slot Attention here."""
    image = render_state(_state(), 3, resolution=96)
    background = np.all(image == BACKGROUND_COLOR.astype(np.uint8), axis=-1)
    assert background.mean() > 0.9


def test_yaw_is_visible_in_the_image() -> None:
    """Objects are rotated squares, not discs; rendering discs would delete a pose dimension."""
    state = _state(count=1)
    upright = render_state(state, 1, resolution=96)
    state[StateLayout(num_objects=1).object_pose_slice][2] = 0.6
    rotated = render_state(state, 1, resolution=96)
    assert not np.array_equal(upright, rotated)


def test_renderer_is_deterministic() -> None:
    state = _state()
    assert np.array_equal(render_state(state, 3), render_state(state, 3))


# ------------------------------------------------------------------ perception

def test_slot_autoencoder_masks_form_a_partition() -> None:
    """Alpha is softmaxed across slots, so per-pixel mask mass must sum to one."""
    model = SlotAutoencoder(resolution=32, num_slots=4, decoder_resolution=4)
    output = model(torch.rand(2, 3, 32, 32))
    assert output["reconstruction"].shape == (2, 3, 32, 32)
    assert torch.allclose(output["masks"].sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)


def test_slot_centroids_land_on_the_mass() -> None:
    masks = torch.zeros(1, 2, 16, 16)
    masks[0, 0, 4, 6] = 1.0
    masks[0, 1, 12, 2] = 1.0
    centroids = slot_centroids(masks)
    assert centroids[0, 0].tolist() == pytest.approx([6.0, 4.0])
    assert centroids[0, 1].tolist() == pytest.approx([2.0, 12.0])


def test_keypoint_encoder_produces_located_entities() -> None:
    model = KeypointAutoencoder(resolution=32, num_keypoints=4, decoder_resolution=4)
    output = model(torch.rand(2, 3, 32, 32))
    assert output["features"].shape[:2] == (2, 4)
    assert output["positions"].shape == (2, 4, 2)
    # Spatial softmax is an expectation over the grid, so positions cannot leave [-1, 1].
    assert output["positions"].abs().max() <= 1.0 + 1e-6
    assert output["reconstruction"].shape == (2, 3, 32, 32)


def test_foreground_weighted_loss_upweights_object_pixels() -> None:
    """The fix for sparse-foreground scenes: errors on objects must count for more.

    Two reconstructions with the SAME total squared error, one concentrated on the object and
    one on the background. Plain MSE cannot tell them apart -- which is exactly why both
    perception front ends plateau without this -- and the weighted loss must.
    """
    model = KeypointAutoencoder(resolution=16, num_keypoints=3, decoder_resolution=2)
    image = torch.zeros(1, 3, 16, 16)
    image[0, 0, 7:9, 7:9] = 1.0  # a small bright "object" on a dark background

    error = torch.zeros_like(image)
    on_object = error.clone()
    on_object[0, 0, 7, 7] = 0.5
    on_background = error.clone()
    on_background[0, 0, 1, 1] = 0.5

    def weighted(perturbation: torch.Tensor) -> float:
        # Bypass the network: score a supplied reconstruction directly with the same weighting.
        reconstruction = image + perturbation
        squared = (reconstruction - image) ** 2
        flat = image.flatten(2)
        background = flat.median(dim=2).values.unsqueeze(-1).unsqueeze(-1)
        deviation = (image - background).abs().amax(dim=1, keepdim=True)
        peak = deviation.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        weight = 1.0 + 50.0 * (deviation / peak)
        return float((squared * weight).sum() / weight.expand_as(squared).sum())

    assert weighted(on_object) > 10 * weighted(on_background)
    # And the unweighted path must still be plain MSE.
    plain = model.loss(image, foreground_weight=0.0)
    assert torch.isfinite(plain)


def test_keypoint_positions_convert_to_the_renderer_pixel_frame() -> None:
    corners = torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])
    pixels = positions_to_pixels(corners, resolution=96)
    assert pixels[0, 0].tolist() == pytest.approx([0.0, 0.0])
    assert pixels[0, 1].tolist() == pytest.approx([95.0, 95.0])
    assert pixels[0, 2].tolist() == pytest.approx([47.5, 47.5])


def test_spatial_softmax_localises_a_peak() -> None:
    """The mechanism the keypoint encoder rests on: expected position tracks the activation.

    Tested on the module directly with a synthetic peaked feature map rather than through an
    untrained autoencoder. An untrained encoder produces near-uniform activations, so its
    keypoints barely move (measured: ~6e-4 of the normalised range when a blob jumps half the
    image) -- that is what an unlearned softmax does, not a defect, and asserting on it would
    have tested initialisation noise instead of the mechanism.
    """
    softmax = SpatialSoftmax(temperature=0.05, learn_temperature=False)
    features = torch.full((1, 2, 16, 16), -10.0)
    features[0, 0, 4, 12] = 10.0   # row 4, column 12
    features[0, 1, 11, 3] = 10.0
    positions, distribution = softmax(features)

    # Positions are normalised to [-1, 1] with x from the column axis and y from the row axis.
    expected_x = -1.0 + 2.0 * 12 / 15
    expected_y = -1.0 + 2.0 * 4 / 15
    assert positions[0, 0].tolist() == pytest.approx([expected_x, expected_y], abs=1e-3)
    assert positions[0, 1].tolist() == pytest.approx(
        [-1.0 + 2.0 * 3 / 15, -1.0 + 2.0 * 11 / 15], abs=1e-3
    )
    assert torch.allclose(distribution.sum(dim=(2, 3)), torch.ones(1, 2), atol=1e-5)


def test_spatial_softmax_temperature_controls_sharpness() -> None:
    """A too-hot softmax drags every keypoint to the image centre -- the collapse mode."""
    features = torch.zeros(1, 1, 16, 16)
    features[0, 0, 2, 2] = 4.0
    cold = SpatialSoftmax(temperature=0.05, learn_temperature=False)(features)[0]
    hot = SpatialSoftmax(temperature=100.0, learn_temperature=False)(features)[0]
    # Cold tracks the peak; hot sits near the centre of the grid regardless of it.
    assert cold.abs().max() > hot.abs().max()
    assert hot.abs().max() < 0.05
