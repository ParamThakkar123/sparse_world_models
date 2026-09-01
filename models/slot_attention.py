"""Slot Attention perception front end (Locatello et al., NeurIPS 2020).

Why this exists
---------------
The whole project consumes structured per-object state: poses and velocities are handed to
the model, which never has to find the objects. For the central finding that is not an
incidental scope limit -- it is load-bearing. The momentum shortcut is available *because*
velocity is served as an input feature. The question a reviewer will ask, and the question
this module exists to answer, is whether the shortcut survives when velocity has to be
inferred from images.

This is a faithful implementation of Slot Attention: a CNN encoder with an additive
positional embedding, an iterative attention module that maps the feature map onto ``K``
permutation-equivariant slots competing via softmax **over slots** (this is the paper's
central inversion of ordinary attention, and it is what makes slots specialise), a GRU slot
update, and a spatial-broadcast decoder producing per-slot RGB plus an alpha mask that is
normalised across slots. Training is unsupervised reconstruction -- no object labels enter
the perception model at any point.

The one place ground truth is used, and why that is standard
------------------------------------------------------------
Slots come out in arbitrary order and that order is not stable between consecutive frames,
so a change-detection task -- which must compare object ``i`` at ``t`` against object ``i``
at ``t+1`` -- needs a correspondence. It is established by Hungarian matching between slot
attention-mask centroids and ground-truth object positions, exactly the protocol the Slot
Attention paper itself uses for its property-prediction experiments.

That is disclosed rather than buried: ground truth is used **only to order the slots**, never
to produce their contents, and never as a model input. The consequence for how the results
must be read is that the pixel numbers are an *upper bound* -- a fully self-contained system
would additionally have to solve tracking, and any error there would subtract from these
numbers rather than add to them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """Four-channel positional grid ``(1, resolution, resolution, 4)``.

    The paper uses a soft position embedding built from linear ramps in both directions and
    their complements, so the network can represent an absolute position with a linear
    readout. Without it the CNN's features are translation-equivariant and slots have no way
    to encode *where* an object is -- which is precisely the quantity this study needs.
    """
    ramp = torch.linspace(0.0, 1.0, resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ramp, ramp, indexing="ij")
    stacked = torch.stack([grid_x, grid_y], dim=-1)
    return torch.cat([stacked, 1.0 - stacked], dim=-1).unsqueeze(0)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_dim: int, resolution: int):
        super().__init__()
        self.projection = nn.Linear(4, hidden_dim)
        self.resolution = resolution

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        grid = build_grid(self.resolution, inputs.device)
        return inputs + self.projection(grid)


class SlotAttention(nn.Module):
    """Iterative slot attention with softmax competition ACROSS slots.

    ``num_iterations`` rounds of: project slots to queries, inputs to keys/values, softmax the
    attention logits over the *slot* axis so slots compete for pixels, take a weighted mean
    of values, and update each slot with a GRU followed by a residual MLP.
    """

    def __init__(
        self,
        num_slots: int,
        input_dim: int,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        num_iterations: int = 3,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.scale = slot_dim ** -0.5

        # Slots are sampled from a LEARNED Gaussian shared by every slot. Sharing is what
        # keeps the module permutation-equivariant and stops slot k from specialising to a
        # fixed object index; per-slot parameters would silently destroy that.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim)
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``inputs`` is ``(B, num_positions, input_dim)``.

        Returns ``(slots, attention)`` with shapes ``(B, K, slot_dim)`` and
        ``(B, K, num_positions)``.
        """
        batch = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        keys = self.to_k(inputs)
        values = self.to_v(inputs)

        mu = self.slots_mu.expand(batch, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attention = None
        for _ in range(self.num_iterations):
            previous = slots
            normed = self.norm_slots(slots)
            queries = self.to_q(normed)

            logits = torch.einsum("bkd,bnd->bkn", queries, keys) * self.scale
            # Softmax over the SLOT axis: slots compete for each input position. This is the
            # inversion that makes the module decompose rather than pool.
            attention = logits.softmax(dim=1)
            # Renormalise over positions to get a weighted MEAN per slot.
            weights = attention + self.epsilon
            weights = weights / weights.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", weights, values)

            # The final iteration is the only one that keeps its gradient path through the
            # recurrence in the reference implementation's "implicit" variant; here every
            # iteration is differentiated, which is the paper's default and is stable at this
            # scale.
            slots = self.gru(
                updates.reshape(-1, self.slot_dim), previous.reshape(-1, self.slot_dim)
            ).reshape(batch, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        assert attention is not None  # num_iterations >= 1 is enforced by construction
        return slots, attention


class SlotAutoencoder(nn.Module):
    """Slot Attention encoder + spatial-broadcast decoder, trained by reconstruction only."""

    def __init__(
        self,
        resolution: int = 96,
        num_slots: int = 4,
        slot_dim: int = 64,
        hidden_dim: int = 64,
        num_iterations: int = 3,
        decoder_resolution: int = 12,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.decoder_resolution = decoder_resolution

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
        )
        self.encoder_position = SoftPositionEmbed(hidden_dim, resolution)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.slot_attention = SlotAttention(
            num_slots=num_slots, input_dim=hidden_dim, slot_dim=slot_dim,
            hidden_dim=2 * hidden_dim, num_iterations=num_iterations,
        )

        self.decoder_position = SoftPositionEmbed(slot_dim, decoder_resolution)
        # Transposed convolutions upsample decoder_resolution -> resolution. With 12 -> 96
        # that is three stride-2 stages.
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(slot_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 4, 3, padding=1),  # 3 RGB channels + 1 alpha
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``image`` is ``(B, 3, H, W)`` in [0, 1]; returns ``(slots, attention)``."""
        features = self.encoder_cnn(image)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C) for the position embedding
        features = self.encoder_position(features)
        features = features.flatten(1, 2)
        features = self.encoder_mlp(self.encoder_norm(features))
        return self.slot_attention(features)

    def decode(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial-broadcast decode. Returns ``(reconstruction, masks)``."""
        batch, num_slots, _ = slots.shape
        grid = slots.reshape(-1, self.slot_dim, 1, 1).expand(
            -1, -1, self.decoder_resolution, self.decoder_resolution
        )
        grid = grid.permute(0, 2, 3, 1)
        grid = self.decoder_position(grid).permute(0, 3, 1, 2)
        decoded = self.decoder_cnn(grid)
        decoded = decoded.reshape(batch, num_slots, 4, self.resolution, self.resolution)
        channels, alpha = decoded.split([3, 1], dim=2)
        # Alpha is normalised ACROSS SLOTS so the composite is a convex combination and each
        # pixel is explained by exactly one unit of mask -- the decoder-side counterpart of
        # the encoder's slot-axis softmax.
        masks = alpha.softmax(dim=1)
        reconstruction = (channels * masks).sum(dim=1)
        return reconstruction, masks.squeeze(2)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        slots, attention = self.encode(image)
        reconstruction, masks = self.decode(slots)
        return {
            "slots": slots,
            "attention": attention,
            "reconstruction": reconstruction,
            "masks": masks,
        }

    def loss(self, image: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(image)["reconstruction"], image)


@torch.no_grad()
def slot_centroids(masks: torch.Tensor) -> torch.Tensor:
    """Centre of mass of each slot's decoder mask, in pixel coordinates.

    ``masks`` is ``(B, K, H, W)``; returns ``(B, K, 2)`` as ``(x, y)``. Used to match slots
    to ground-truth objects -- see the module docstring for why that matching is legitimate
    and what it costs.
    """
    height, width = masks.shape[-2:]
    device = masks.device
    ys = torch.arange(height, device=device, dtype=masks.dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=device, dtype=masks.dtype).view(1, 1, 1, width)
    total = masks.sum(dim=(2, 3)).clamp_min(1e-8)
    centre_x = (masks * xs).sum(dim=(2, 3)) / total
    centre_y = (masks * ys).sum(dim=(2, 3)) / total
    return torch.stack([centre_x, centre_y], dim=-1)
