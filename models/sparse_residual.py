from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .delta_heads import (
    DELTA_HEAD_TYPES,
    GaussianDelta,
    MixtureDelta,
    ObjectDeltaHead,
    build_delta_head,
)
from .sparse_gating import GateOutput, ObjectChangeGate


@dataclass
class SparseResidualOutput:
    gate: GateOutput
    delta: torch.Tensor
    masked_delta: torch.Tensor
    # Present only for the probabilistic heads ('gaussian' / 'mdn'); ``None`` for 'mse'.
    # ``delta`` above is always the *point estimate*, so every downstream consumer that
    # reconstructs poses keeps working unchanged regardless of head type.
    delta_dist: GaussianDelta | MixtureDelta | None = None


@dataclass
class SparseResidualLoss:
    total: torch.Tensor
    gate_bce: torch.Tensor
    delta_l2: torch.Tensor
    sparsity_penalty: torch.Tensor
    positive_class_weight: torch.Tensor
    # Negative log-likelihood of the true delta under a probabilistic head. This is the
    # term actually optimised when one is active; ``delta_l2`` is then still computed from
    # the point estimate and reported, so MSE and NLL runs stay comparable on one metric.
    delta_nll: torch.Tensor | None = None


class SparseResidualHead(nn.Module):
    """Combines a binary change gate with an object-wise delta regressor.

    ``delta_head_type`` selects the delta parameterisation:

    * ``mse``      -- deterministic point regressor (the original head; default, so
                      existing checkpoints load against an unchanged ``delta_head``
                      submodule).
    * ``gaussian`` -- heteroscedastic diagonal Gaussian trained by NLL.
    * ``mdn``      -- ``num_mixture_components``-component Gaussian mixture trained by NLL.

    See :mod:`models.delta_heads` for why the probabilistic heads exist: the oracle-gate
    diagnostic shows the deterministic head is pinned at the no-op floor on the objects
    that actually move, because squared error on multimodal contact is minimised by a
    near-zero conditional mean.
    """

    def __init__(
        self,
        object_feature_dim: int,
        gate_hidden_dim: int = 128,
        gate_num_layers: int = 2,
        delta_hidden_dim: int = 128,
        delta_num_layers: int = 2,
        delta_head_type: str = "mse",
        num_mixture_components: int = 5,
    ):
        super().__init__()
        if delta_head_type not in DELTA_HEAD_TYPES:
            raise ValueError(
                f"Unsupported delta_head_type '{delta_head_type}'. Expected one of {DELTA_HEAD_TYPES}."
            )
        self.delta_head_type = delta_head_type
        self.num_mixture_components = num_mixture_components
        self.gate = ObjectChangeGate(
            object_feature_dim=object_feature_dim,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_num_layers,
        )
        self.delta_head = build_delta_head(
            delta_head_type,
            object_feature_dim=object_feature_dim,
            hidden_dim=delta_hidden_dim,
            num_layers=delta_num_layers,
            output_dim=3,
            num_components=num_mixture_components,
        )

    @property
    def is_probabilistic(self) -> bool:
        return self.delta_head_type != "mse"

    def forward(
        self,
        object_features: torch.Tensor,
        *,
        estimator: str = "gumbel_st",
        temperature: float = 1.0,
        hard: bool = True,
    ) -> SparseResidualOutput:
        gate = self.gate(
            object_features,
            estimator=estimator,
            temperature=temperature,
            hard=hard,
        )
        raw = self.delta_head(object_features)
        if isinstance(raw, torch.Tensor):
            delta, delta_dist = raw, None
        else:
            # Probabilistic head: the deployed prediction is the distribution's point
            # estimate (component mode for the MDN, mean for the Gaussian), never the
            # mixture mean.
            delta, delta_dist = raw.point_estimate, raw
        masked_delta = gate.gates.unsqueeze(-1) * delta
        return SparseResidualOutput(
            gate=gate, delta=delta, masked_delta=masked_delta, delta_dist=delta_dist
        )


def masked_delta_l2_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    changed_mask: torch.Tensor,
    gate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 regression only on objects labeled changed.

    Shapes:
    - `pred_delta`: `(batch, num_objects, 3)`
    - `target_delta`: `(batch, num_objects, 3)`
    - `changed_mask`: `(batch, num_objects)`
    - `gate_weights`: optional `(batch, num_objects)` predicted gate weighting
    """

    if pred_delta.shape != target_delta.shape:
        raise ValueError("pred_delta and target_delta must have the same shape.")
    if changed_mask.shape != pred_delta.shape[:2]:
        raise ValueError("changed_mask must have shape (batch, num_objects).")
    if gate_weights is not None and gate_weights.shape != pred_delta.shape[:2]:
        raise ValueError("gate_weights must have shape (batch, num_objects).")

    squared_error = (pred_delta - target_delta) ** 2
    mask = changed_mask.unsqueeze(-1).to(squared_error.dtype)
    if gate_weights is not None:
        mask = mask * gate_weights.unsqueeze(-1).to(squared_error.dtype)
    masked_error = squared_error * mask
    denom = mask.sum() * pred_delta.shape[-1]
    if float(denom.item()) == 0.0:
        return masked_error.sum() * 0.0
    return masked_error.sum() / denom


def masked_delta_nll_loss(
    delta_dist: GaussianDelta | MixtureDelta,
    target_delta: torch.Tensor,
    changed_mask: torch.Tensor,
    gate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood of the true delta, averaged over objects labeled changed.

    The mask structure mirrors :func:`masked_delta_l2_loss` exactly -- same objects
    supervised, same optional gate weighting -- so swapping the objective changes *only*
    the objective. Unchanged objects are excluded because their delta is zero by
    construction and a density head would otherwise spend all its capacity spiking on
    that trivial point mass.
    """
    log_prob = delta_dist.log_prob(target_delta)  # (batch, num_objects)
    if changed_mask.shape != log_prob.shape:
        raise ValueError("changed_mask must have shape (batch, num_objects).")
    if gate_weights is not None and gate_weights.shape != log_prob.shape:
        raise ValueError("gate_weights must have shape (batch, num_objects).")

    mask = changed_mask.to(log_prob.dtype)
    if gate_weights is not None:
        mask = mask * gate_weights.to(log_prob.dtype)
    denom = mask.sum()
    if float(denom.item()) == 0.0:
        return -(log_prob * mask).sum() * 0.0
    return -(log_prob * mask).sum() / denom


def sparse_residual_loss(
    gate_logits: torch.Tensor,
    gate_probs: torch.Tensor,
    pred_delta: torch.Tensor,
    target_changed_mask: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    gate_loss_weight: float = 1.0,
    delta_loss_weight: float = 1.0,
    sparsity_weight: float = 1e-2,
    positive_class_weight: float | None = None,
    delta_gate: torch.Tensor | None = None,
    delta_dist: GaussianDelta | MixtureDelta | None = None,
) -> SparseResidualLoss:
    """Gate BCE + delta regression + sparsity penalty.

    When ``delta_dist`` is supplied the delta term becomes the masked NLL of that
    distribution instead of masked squared error; ``pred_delta`` is then the point
    estimate and its L2 is still reported for cross-run comparability.
    """
    gate_target = target_changed_mask.to(gate_logits.dtype)
    pos_weight = None
    if positive_class_weight is not None:
        pos_weight = torch.tensor(
            positive_class_weight,
            dtype=gate_logits.dtype,
            device=gate_logits.device,
        )
    gate_bce = F.binary_cross_entropy_with_logits(
        gate_logits,
        gate_target,
        pos_weight=pos_weight,
    )
    delta_l2 = masked_delta_l2_loss(
        pred_delta,
        target_delta,
        target_changed_mask,
        gate_weights=delta_gate,
    )
    delta_nll = None
    if delta_dist is not None:
        delta_nll = masked_delta_nll_loss(
            delta_dist, target_delta, target_changed_mask, gate_weights=delta_gate
        )
    delta_term = delta_l2 if delta_nll is None else delta_nll

    sparsity_penalty = gate_probs.mean()
    total = (
        gate_loss_weight * gate_bce
        + delta_loss_weight * delta_term
        + sparsity_weight * sparsity_penalty
    )
    return SparseResidualLoss(
        total=total,
        gate_bce=gate_bce,
        delta_l2=delta_l2,
        sparsity_penalty=sparsity_penalty,
        positive_class_weight=(
            pos_weight
            if pos_weight is not None
            else torch.tensor(1.0, dtype=gate_logits.dtype, device=gate_logits.device)
        ),
        delta_nll=delta_nll,
    )
