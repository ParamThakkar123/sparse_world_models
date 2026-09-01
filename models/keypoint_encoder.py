"""Spatial-softmax keypoint autoencoder -- a second, independent perception front end.

Why a second one
----------------
The pixel version of this study needs *a* perception module, not a specific one. Slot
Attention is the natural choice from the object-centric world-model literature, but it is
known to be finicky on scenes where objects occupy a small fraction of the image -- which is
exactly the regime here, since a 5 cm object on a 60 cm table covers about 0.7% of the frame
and reconstruction MSE is then dominated by background. If the pixel result rested on one
front end and that front end failed to segment, the experiment would measure a broken encoder
rather than the dynamics claim under test.

This is the other standard choice, and it comes from the manipulation literature rather than
the CLEVR literature: the spatial-softmax keypoint autoencoder of Finn et al. 2016 ("Deep
Spatial Autoencoders for Visuomotor Learning"), also the perception front end in Levine et
al. 2016. A convolutional trunk produces ``K`` feature maps; each is turned into a
*probability distribution over pixels* by a spatial softmax, and its expected position is a
keypoint. Because each keypoint is forced to be a location, the module cannot solve the task
by ignoring small objects the way a reconstruction-only decoder can -- a keypoint has to sit
somewhere, and the reconstruction objective pushes it onto the parts of the image that
actually vary.

Having both means the pixel finding can be reported as robust to the perception module, and
if the two disagree that disagreement is itself informative and reportable.

What each keypoint carries
--------------------------
``encode`` returns ``(features, positions)``: the per-keypoint feature vector is the trunk's
activation bilinearly sampled at the keypoint, concatenated with the keypoint's own
normalised ``(x, y)``. Position is included explicitly because the change-detection task is
fundamentally about *where* objects are and where they will be, and a feature vector sampled
at a point does not otherwise encode that point.

Training is unsupervised: reconstruct the image from the keypoints alone. No object labels
enter, exactly as with Slot Attention. Keypoint-to-object correspondence is established
afterwards by Hungarian matching on position, and is used only to order the keypoints.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialSoftmax(nn.Module):
    """Turn each feature map into a probability distribution and take its expected position.

    Returns ``(positions, distributions)`` with positions in ``[-1, 1]`` normalised image
    coordinates, shape ``(B, K, 2)``, and the per-map distribution ``(B, K, H, W)``.
    """

    def __init__(self, temperature: float = 1.0, learn_temperature: bool = True):
        super().__init__()
        value = torch.tensor(float(temperature)).log()
        # Temperature is learned in log space so it stays positive; a too-hot softmax makes
        # every keypoint drift to the image centre, which is the exact failure this module is
        # meant to avoid.
        self.log_temperature = nn.Parameter(value) if learn_temperature else value

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = features.shape
        temperature = self.log_temperature.exp()
        flat = (features.reshape(batch, channels, height * width) / temperature).softmax(dim=-1)
        distribution = flat.reshape(batch, channels, height, width)

        xs = torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
        expected_x = (distribution.sum(dim=2) * xs).sum(dim=-1)
        expected_y = (distribution.sum(dim=3) * ys).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1), distribution


class KeypointAutoencoder(nn.Module):
    """CNN trunk -> spatial-softmax keypoints -> reconstruction from keypoints alone.

    The decoder deliberately sees ONLY the keypoints (their positions and sampled features),
    never the trunk's full feature map. That bottleneck is what forces the keypoints to land
    on the informative parts of the image: anything the decoder needs must be routed through
    a location.
    """

    def __init__(
        self,
        resolution: int = 96,
        num_keypoints: int = 5,
        hidden_dim: int = 64,
        feature_dim: int = 16,
        decoder_resolution: int = 12,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim
        self.decoder_resolution = decoder_resolution
        # Per-keypoint descriptor: sampled trunk features plus the keypoint's own position.
        self.keypoint_dim = feature_dim + 2

        self.trunk = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
        )
        self.to_keypoint_maps = nn.Conv2d(hidden_dim, num_keypoints, 1)
        self.to_features = nn.Conv2d(hidden_dim, feature_dim, 1)
        self.spatial_softmax = SpatialSoftmax()

        self.decoder_input = nn.Linear(num_keypoints * self.keypoint_dim, hidden_dim * decoder_resolution ** 2)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``image`` is ``(B, 3, H, W)`` in [0, 1]. Returns ``(features, positions)``.

        ``features`` is ``(B, K, feature_dim + 2)``; ``positions`` is ``(B, K, 2)`` in
        normalised ``[-1, 1]`` coordinates.
        """
        trunk = self.trunk(image)
        positions, _ = self.spatial_softmax(self.to_keypoint_maps(trunk))
        descriptors = self.to_features(trunk)
        # Bilinearly sample the descriptor map at each keypoint. grid_sample expects the
        # sampling grid as (B, out_h, out_w, 2) with (x, y) order, which is why the keypoints
        # are unsqueezed into a 1-row grid rather than reshaped.
        grid = positions.unsqueeze(1)  # (B, 1, K, 2)
        sampled = F.grid_sample(descriptors, grid, align_corners=True, mode="bilinear")
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, K, feature_dim)
        return torch.cat([sampled, positions], dim=-1), positions

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        hidden = self.decoder_input(features.reshape(batch, -1))
        hidden = hidden.reshape(batch, -1, self.decoder_resolution, self.decoder_resolution)
        return self.decoder_cnn(hidden)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features, positions = self.encode(image)
        return {
            "features": features,
            "positions": positions,
            "reconstruction": self.decode(features),
        }

    def loss(self, image: torch.Tensor, foreground_weight: float = 0.0) -> torch.Tensor:
        """Reconstruction error, optionally weighted toward the foreground.

        Plain MSE fails on these scenes and the failure is quantitative, not subtle: objects
        cover about 0.7% of a frame, so a decoder that reproduces the dark background and
        nothing else already achieves most of the attainable loss. Measured, with
        ``foreground_weight=0``, the keypoints spread apart but never land on objects -- match
        distance plateaus at ~22px on a 96px image against ~33px for chance. Slot Attention
        collapses harder on the same data for the same reason.

        ``foreground_weight`` scales each pixel's contribution by how far it is from the
        image's own **median colour**, which on these renders is the background. That is
        computed per batch from the pixels themselves: no labels, no masks, and no use of the
        known background constant, so the perception model stays unsupervised. It changes
        which errors the objective cares about, not what information it is given.

        Set to 0 to recover the standard objective.
        """
        reconstruction = self.forward(image)["reconstruction"]
        error = (reconstruction - image) ** 2
        if foreground_weight <= 0.0:
            return error.mean()

        # Median over the spatial axes, per image and channel -> the modal (background) colour.
        flat = image.flatten(2)
        background = flat.median(dim=2).values.unsqueeze(-1).unsqueeze(-1)
        deviation = (image - background).abs().amax(dim=1, keepdim=True)
        # Normalise per image so the weight scale does not depend on the palette.
        peak = deviation.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        weight = 1.0 + foreground_weight * (deviation / peak)
        # ``weight`` has one channel and broadcasts over the three colour channels. Summing
        # it via ``expand_as(error)`` would materialise a full B x 3 x H x W tensor purely to
        # add it up; multiplying the single-channel sum by the channel count is identical and
        # allocates nothing. This mattered in practice -- the expand version died with a bare
        # allocation failure on a loaded machine.
        return (error * weight).sum() / (weight.sum() * error.shape[1])


def positions_to_pixels(positions: torch.Tensor, resolution: int) -> torch.Tensor:
    """Normalised ``[-1, 1]`` keypoints to pixel coordinates, matching the renderer's frame.

    Kept here so the one place that converts between the two coordinate conventions is next
    to the module that produces them.
    """
    return (positions + 1.0) * 0.5 * (resolution - 1)
