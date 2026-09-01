"""Latent dynamics model — the learned-model control baseline (W5).

Comparing the sparse world model only against scripted and random policies is not a real
control result: neither is a *learned model*, so the comparison cannot say whether
object-centric structure helps or whether any learned model would do. The standard modern
baselines are TD-MPC2 and Dreamer, both of which plan through a **latent** dynamics model
rather than a structured state.

This is the shared core of those methods, without the RL machinery neither this task nor this
comparison needs: an encoder to a latent, a latent transition function, and a decoder back to
object poses. Dreamer's stochastic RSSM and TD-MPC2's value function and policy prior are
deliberately absent -- there is no reward learning here because the planner already has an
analytic cost, and no recurrence because the task is Markov in the full state. What remains is
exactly the ingredient under test: *does prediction have to be object-factored, or is a
monolithic latent enough?*

Training combines three terms, which is what makes it a latent-dynamics model rather than a
plain autoencoder:

  * ``reconstruction`` -- ``decode(encode(s))`` must recover the object poses, so the latent
    has to retain them.
  * ``latent consistency`` -- ``dynamics(encode(s), a)`` must match ``encode(s')``, so the
    latent is predictable in its own space (TD-MPC2's central idea).
  * ``prediction`` -- ``decode(dynamics(encode(s), a))`` must match the true next poses, which
    is what the planner actually consumes.

The encode target is detached: without that, the consistency term can be trivially minimised
by collapsing the latent to a constant, which satisfies predictability while destroying every
other term's signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(max(0, num_layers - 1)):
        layers += [nn.Linear(current, hidden_dim), nn.ReLU()]
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


@dataclass
class LatentLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    consistency: torch.Tensor
    prediction: torch.Tensor


class LatentDynamicsModel(nn.Module):
    """Encoder / latent transition / decoder over object poses.

    ``forward`` mirrors the interface the planner needs: given a batch of full states and
    actions, return predicted next object poses ``(batch, num_objects, 3)``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_objects: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        pose_dim: int = 3,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp(state_dim, hidden_dim, latent_dim, num_layers)
        self.dynamics = _mlp(latent_dim + action_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = _mlp(latent_dim, hidden_dim, num_objects * pose_dim, num_layers)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(latent)
        return flat.reshape(flat.shape[0], self.num_objects, self.pose_dim)

    def step_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([latent, action], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.decode(self.step_latent(self.encode(state), action))

    def losses(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        current_pose: torch.Tensor,
        next_pose: torch.Tensor,
        consistency_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> LatentLosses:
        latent = self.encode(state)
        predicted_latent = self.step_latent(latent, action)

        reconstruction = nn.functional.mse_loss(self.decode(latent), current_pose)
        # Detached target: an undetached one lets the model satisfy consistency by driving
        # the latent to a constant, which is a degenerate optimum of this term alone.
        target_latent = self.encode(next_state).detach()
        consistency = nn.functional.mse_loss(predicted_latent, target_latent)
        prediction = nn.functional.mse_loss(self.decode(predicted_latent), next_pose)

        total = (
            prediction
            + reconstruction_weight * reconstruction
            + consistency_weight * consistency
        )
        return LatentLosses(
            total=total,
            reconstruction=reconstruction,
            consistency=consistency,
            prediction=prediction,
        )
