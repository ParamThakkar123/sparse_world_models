"""Permutation-equivariant relational dynamics baselines (W2).

The gate ablation currently interpolates from a dense monolith to the sparse model through
per-object MLPs, and concludes the change gate earns the largest single share of the win.
That ladder has a gap a reviewer will find immediately: its ungated rungs process each
object *independently*, so "ungated" is confounded with "no interaction modelling". The
obvious objection is that a proper relational model -- one that can see object ``j`` while
predicting object ``i`` -- would close the gap without needing any gate.

These two baselines close that hole. Both are permutation-equivariant, share weights across
objects, consume exactly the same per-object features as the sparse model, and predict a
residual that is *always* applied (matching ``oc_residual``, the strongest ungated rung).
The only ingredient they add over ``oc_residual`` is inter-object communication:

  * :class:`InteractionNetworkPredictor` -- explicit pairwise edge messages, summed per
    node (Battaglia et al.-style interaction network).
  * :class:`SetTransformerPredictor` -- multi-head self-attention over the object set, so
    each object attends to every other with learned, input-dependent weights.

If the sparse model still wins against these, "model what changes" is doing something that
neither capacity nor relational structure supplies. If they close the gap, the paper's
central attribution claim needs rewriting -- which is exactly why the experiment is worth
running rather than assuming.
"""

from __future__ import annotations

import torch
from torch import nn

POSE_OUTPUT_DIM = 3


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(current, hidden_dim), nn.ReLU()]
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class InteractionNetworkPredictor(nn.Module):
    """Interaction network over the object set, predicting a residual pose delta.

    Encode each object, exchange a message along every ordered pair, sum the incoming
    messages per object, then decode a delta from ``[node embedding, aggregated message]``.
    Self-edges are excluded so the message term carries only *relational* information --
    otherwise the node's own embedding would leak in twice and the ablation could not
    attribute anything to interaction.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        num_layers: int = 2,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        self.mode = mode
        self.node_encoder = _mlp(object_feature_dim, hidden_dim, hidden_dim, num_layers)
        self.edge_model = _mlp(2 * hidden_dim, hidden_dim, message_dim, num_layers)
        self.node_decoder = _mlp(hidden_dim + message_dim, hidden_dim, POSE_OUTPUT_DIM, num_layers)

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")
        batch, num_objects, _ = object_features.shape
        nodes = self.node_encoder(object_features)  # (B, N, H)

        sender = nodes.unsqueeze(2).expand(batch, num_objects, num_objects, nodes.shape[-1])
        receiver = nodes.unsqueeze(1).expand(batch, num_objects, num_objects, nodes.shape[-1])
        messages = self.edge_model(torch.cat([receiver, sender], dim=-1))  # (B, N, N, M)

        # Zero the diagonal so an object never messages itself.
        eye = torch.eye(num_objects, dtype=torch.bool, device=object_features.device)
        messages = messages.masked_fill(eye.view(1, num_objects, num_objects, 1), 0.0)
        aggregated = messages.sum(dim=2)  # (B, N, M)

        out = self.node_decoder(torch.cat([nodes, aggregated], dim=-1))
        return out if self.mode == "absolute" else current_pose + out


class SetTransformerPredictor(nn.Module):
    """Pre-norm multi-head self-attention over the object set, predicting a residual delta.

    No positional encoding is used: object order in the state vector is arbitrary, so the
    model must stay permutation-equivariant to be a fair stand-in for an object-centric
    architecture.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}.")
        self.mode = mode
        self.input_projection = nn.Linear(object_feature_dim, hidden_dim)
        self.attention_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_blocks))
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_blocks)
        )
        self.feedforward_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_blocks))
        self.feedforwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim)
            )
            for _ in range(num_blocks)
        )
        self.output_projection = nn.Linear(hidden_dim, POSE_OUTPUT_DIM)

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")
        hidden = self.input_projection(object_features)
        for norm, attention, ff_norm, feedforward in zip(
            self.attention_norms, self.attentions, self.feedforward_norms, self.feedforwards
        ):
            normed = norm(hidden)
            attended, _ = attention(normed, normed, normed, need_weights=False)
            hidden = hidden + attended
            hidden = hidden + feedforward(ff_norm(hidden))
        out = self.output_projection(hidden)
        return out if self.mode == "absolute" else current_pose + out


def match_hidden_dim(
    factory, target_parameters: int, candidates: range, **kwargs
) -> tuple[int, int]:
    """Pick the width whose parameter count lands closest to ``target_parameters``.

    Mirrors ``gate_ablation.resolve_hidden_dim`` so every rung of the ladder is sized by the
    same rule and no rung can win on capacity. Returns ``(width, parameter_count)``.
    """
    best_width, best_count, best_gap = None, None, None
    for width in candidates:
        try:
            probe = factory(hidden_dim=width, **kwargs)
        except ValueError:
            # e.g. a width not divisible by the head count.
            continue
        count = sum(p.numel() for p in probe.parameters())
        gap = abs(count - target_parameters)
        if best_gap is None or gap < best_gap:
            best_width, best_count, best_gap = width, count, gap
    if best_width is None:
        raise ValueError("No candidate width produced a valid model.")
    return best_width, int(best_count)
