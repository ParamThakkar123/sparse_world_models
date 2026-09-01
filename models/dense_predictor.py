from __future__ import annotations

import torch
from torch import nn


class DenseStatePredictor(nn.Module):
    """Dense MLP baseline that predicts target state features from `(s_t, a_t)`."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        layers: list[nn.Module] = []
        input_dim = state_dim + action_dim
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))
