from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GateOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    gates: torch.Tensor


def sample_gumbel(shape: torch.Size, device: torch.device, eps: float = 1e-8) -> torch.Tensor:
    uniform = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def gumbel_sigmoid(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    noise = sample_gumbel(logits.shape, logits.device) - sample_gumbel(logits.shape, logits.device)
    y_soft = torch.sigmoid((logits + noise) / temperature)
    if not hard:
        return y_soft

    y_hard = (y_soft >= 0.5).to(y_soft.dtype)
    return y_hard.detach() - y_soft.detach() + y_soft


def straight_through_bernoulli(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    hard = (probs >= 0.5).to(probs.dtype)
    return hard.detach() - probs.detach() + probs


class ObjectChangeGate(nn.Module):
    """Per-object binary gate for sparse/residual world model updates.

    Input is expected as `(batch, num_objects, object_feature_dim)`.
    Output is one gate per object.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers: list[nn.Module] = []
        current_dim = object_feature_dim
        for _ in range(max(0, num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        object_features: torch.Tensor,
        *,
        estimator: str = "gumbel_st",
        temperature: float = 1.0,
        hard: bool = True,
    ) -> GateOutput:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")

        logits = self.mlp(object_features).squeeze(-1)
        probs = torch.sigmoid(logits)

        if estimator == "sigmoid":
            gates = probs
        elif estimator == "st":
            gates = straight_through_bernoulli(logits)
        elif estimator == "gumbel":
            gates = gumbel_sigmoid(logits, temperature=temperature, hard=False)
        elif estimator == "gumbel_st":
            gates = gumbel_sigmoid(logits, temperature=temperature, hard=hard)
        else:
            raise ValueError(
                f"Unsupported estimator '{estimator}'. "
                "Expected one of: sigmoid, st, gumbel, gumbel_st."
            )

        return GateOutput(logits=logits, probs=probs, gates=gates)
