"""Probabilistic per-object delta heads.

Motivation (see RESULTS.md "Oracle-gate diagnostic"): feeding the *ground-truth* changed
mask to the deterministic delta head barely moves changed-object L2 -- at ``N=8`` it is
0.474 against a no-op reference of 0.470, i.e. the head does no better than declaring the
object stationary. The gate is not the bottleneck; the regression is.

The diagnosed cause is that one-step contact deltas are **multimodal**. A pusher grazing
a box edge can send it left or right; whether contact happens at all can hinge on a
sub-millimetre difference in approach. Squared error is minimised by the *conditional
mean* of those outcomes, which for a symmetric bimodal push is approximately zero
displacement -- exactly the no-op prediction the diagnostic observes. No amount of extra
capacity or data fixes this, because the deterministic head is converging correctly to
the wrong estimator.

These heads replace the point regressor with a conditional *density* trained by negative
log-likelihood:

  * :class:`GaussianDeltaHead` -- heteroscedastic diagonal Gaussian. Predicts a per-object,
    per-dimension scale, so the model can say "this object will move, but I am unsure how
    far" instead of hedging toward zero. Unimodal, so it does not by itself solve
    multimodality; it is the controlled intermediate rung between MSE and the mixture, and
    the rung that turned out to isolate which of the two ingredients matters.
  * :class:`MixtureDeltaHead` -- ``K``-component diagonal Gaussian mixture (MDN). Each
    component can claim one contact outcome, and the mixture weights carry the ambiguity.

**Measured outcome (seed 0, clean splits): the diagnosis above was right about the data and
wrong about the remedy.** The MDN clears the no-op floor at every object count, by the
widest margin where the diagnostic was worst (``N=8``: +0.048 against +0.019 for squared
error), and holds it under parameter matching. Three tests pin down why:

1. *Not mode selection.* Scoring the same trained MDN with the mixture mean instead of the
   highest-weight component moves changed-object L2 by between +0.003 and -0.003 across all
   six trained cells -- noise around zero. Committing to a mode buys nothing.
2. *Not heteroscedasticity alone.* At matched parameter count the unimodal
   :class:`GaussianDeltaHead` is **not** better than squared error (worse at ``N=5`` and
   ``N=8``), so "the likelihood discounts unpredictable deltas" cannot be the explanation.
   (Heteroscedasticity does earn its keep on a different axis: it is what keeps the head
   stable under *rollout* training, where the deterministic head falls below no-op at every
   count -- a result that survives the ``--rollout-target recorded`` control.)
3. *Yes, multimodality -- via the fit, not the point estimate.* A component-count sweep with
   every K pinned to one parameter budget gives the same shape at all three counts: monotone
   gain from K=1 to K=3, plateau at K=3-5, slight decline at K=10 (``N=8``: +0.023 / +0.031 /
   +0.044 / +0.048 / +0.041). Only a genuinely multimodal conditional with a few modes
   produces a gain that scales with component count at fixed capacity.

The account consistent with all three: a unimodal density fit to a multimodal target is
**biased**, so its mean is pulled off the truth; a mixture that fits the modes has an
unbiased mean. Modelling the multimodality is what matters -- which point estimate you read
off the fitted density does not. About three components suffice; K=10 splits on noise.

Both point estimates are kept and reported precisely because that comparison is what
redirected the story; ``component_mode`` remains the deployed estimator so the reported
numbers do not silently depend on which one happens to read better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# Scale clamps keep the NLL finite: without a floor a component can collapse onto a single
# training point and drive log_scale to -inf. The floor (~4.5e-4) sits well below the
# 2 cm motion threshold that defines a "changed" object, so it does not blunt the model.
MIN_LOG_SCALE = -7.7
MAX_LOG_SCALE = 2.0
# Deltas are metre-scale but small (a hard-subset push moves an object >= 0.02 m). Starting
# the scale near 0.02 rather than 1.0 keeps the initial NLL in a sane range and stops the
# optimiser spending its first epochs just shrinking variance.
DEFAULT_INIT_LOG_SCALE = math.log(0.02)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class ObjectDeltaHead(nn.Module):
    """Per-object residual regressor for planar pose deltas `(dx, dy, dtheta)`.

    Defined here rather than in ``sparse_residual`` so the probabilistic heads below can
    reuse it as a trunk without a circular import; ``models.sparse_residual`` re-exports
    it, so every existing import path is unchanged.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
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
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    @property
    def final_layer(self) -> nn.Linear:
        """The output ``Linear``, exposed so subclasses can initialise specific bias slices."""
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        return final

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")
        return self.mlp(object_features)


def _diagonal_normal_log_prob(
    target: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """Log density of a diagonal Gaussian, summed over the last (feature) dimension."""
    z = (target - mean) * torch.exp(-log_scale)
    return (-0.5 * z**2 - log_scale - _HALF_LOG_2PI).sum(dim=-1)


@dataclass
class GaussianDelta:
    """Per-object diagonal Gaussian over the pose delta. Shapes ``(batch, num_objects, dim)``."""

    mean: torch.Tensor
    log_scale: torch.Tensor

    @property
    def point_estimate(self) -> torch.Tensor:
        """Mean and mode coincide for a Gaussian, so there is no estimator choice here."""
        return self.mean

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """Per-object log density, shape ``(batch, num_objects)``."""
        return _diagonal_normal_log_prob(target, self.mean, self.log_scale)

    def entropy_proxy(self) -> torch.Tensor:
        """Mean predicted log-scale -- a scalar readout of how uncertain the head is."""
        return self.log_scale.mean()


@dataclass
class MixtureDelta:
    """Per-object diagonal Gaussian mixture over the pose delta.

    Shapes: ``logits (batch, num_objects, K)``, ``means``/``log_scales``
    ``(batch, num_objects, K, dim)``.
    """

    logits: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def mixture_mean(self) -> torch.Tensor:
        """Weighted average of the components -- the MSE-optimal estimator.

        Kept for diagnosis only. This is the quantity that collapses toward zero under
        symmetric multimodal contact, so it must not be used to reconstruct poses.
        """
        return (self.weights.unsqueeze(-1) * self.means).sum(dim=-2)

    @property
    def component_mode(self) -> torch.Tensor:
        """Mean of the highest-weight component -- the estimator we deploy.

        Not the exact mode of the mixture density (which has no closed form). It was
        chosen to commit to a single outcome rather than blend mutually exclusive ones;
        measurement then showed it makes no material difference here (see the module
        docstring), so it is retained as the deployed estimator mainly for stability of
        the reported numbers, not because it wins.
        """
        best = self.logits.argmax(dim=-1)  # (B, N)
        index = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, self.means.shape[-1])
        return torch.gather(self.means, dim=-2, index=index).squeeze(-2)

    @property
    def point_estimate(self) -> torch.Tensor:
        return self.component_mode

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """Per-object mixture log density, shape ``(batch, num_objects)``."""
        expanded = target.unsqueeze(-2)  # (B, N, 1, D) broadcasts over components
        component_log_prob = _diagonal_normal_log_prob(expanded, self.means, self.log_scales)
        return torch.logsumexp(torch.log_softmax(self.logits, dim=-1) + component_log_prob, dim=-1)

    def effective_components(self) -> torch.Tensor:
        """Perplexity of the mixture weights: 1.0 means fully collapsed, K means uniform.

        Worth logging -- an MDN that collapses to one component has silently reverted to
        the heteroscedastic Gaussian, and the comparison between the two rungs would then
        be measuring nothing.
        """
        weights = self.weights.clamp_min(1e-9)
        return torch.exp(-(weights * weights.log()).sum(dim=-1)).mean()


def _bias_init_log_scale(final_layer: nn.Linear, slice_: slice, value: float) -> None:
    with torch.no_grad():
        final_layer.bias[slice_].fill_(value)


class GaussianDeltaHead(nn.Module):
    """Heteroscedastic diagonal-Gaussian delta head.

    Shares :class:`~models.sparse_residual.ObjectDeltaHead` as its trunk so the only
    architectural difference from the deterministic head is the width of the final layer
    (``2 * output_dim`` instead of ``output_dim``). That keeps the MSE-vs-NLL comparison a
    comparison of *objectives*, not of network topology.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        init_log_scale: float = DEFAULT_INIT_LOG_SCALE,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.trunk = ObjectDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2 * output_dim,
        )
        _bias_init_log_scale(self.trunk.final_layer, slice(output_dim, 2 * output_dim), init_log_scale)

    def forward(self, object_features: torch.Tensor) -> GaussianDelta:
        raw = self.trunk(object_features)
        mean, log_scale = raw.split(self.output_dim, dim=-1)
        return GaussianDelta(mean=mean, log_scale=log_scale.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE))


class MixtureDeltaHead(nn.Module):
    """Mixture-density delta head: ``K`` diagonal Gaussians per object.

    Output layout per object is ``[K mixture logits | K*D means | K*D log-scales]``.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        num_components: int = 5,
        init_log_scale: float = DEFAULT_INIT_LOG_SCALE,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be at least 1.")
        self.output_dim = output_dim
        self.num_components = num_components
        total = num_components * (1 + 2 * output_dim)
        self.trunk = ObjectDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=total,
        )
        scale_start = num_components + num_components * output_dim
        _bias_init_log_scale(self.trunk.final_layer, slice(scale_start, total), init_log_scale)
        # Break the symmetry between components. Identical initial means give identical
        # gradients, so every component would track the conditional mean forever and the
        # mixture would never separate the contact modes it exists to separate.
        with torch.no_grad():
            self.trunk.final_layer.bias[num_components:scale_start].normal_(0.0, 0.01)

    def forward(self, object_features: torch.Tensor) -> MixtureDelta:
        raw = self.trunk(object_features)
        batch_shape = raw.shape[:-1]
        k, d = self.num_components, self.output_dim
        logits = raw[..., :k]
        means = raw[..., k : k + k * d].reshape(*batch_shape, k, d)
        log_scales = raw[..., k + k * d :].reshape(*batch_shape, k, d)
        return MixtureDelta(
            logits=logits,
            means=means,
            log_scales=log_scales.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE),
        )


DELTA_HEAD_TYPES = ("mse", "gaussian", "mdn")


def delta_head_parameters(
    delta_head_type: str,
    *,
    object_feature_dim: int,
    hidden_dim: int,
    num_layers: int = 2,
    output_dim: int = 3,
    num_components: int = 5,
) -> int:
    head = build_delta_head(
        delta_head_type,
        object_feature_dim=object_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        num_components=num_components,
    )
    return sum(p.numel() for p in head.parameters())


def match_delta_hidden_dim(
    delta_head_type: str,
    target_parameters: int,
    *,
    object_feature_dim: int,
    num_layers: int = 2,
    output_dim: int = 3,
    num_components: int = 5,
    candidates: range = range(8, 513),
) -> tuple[int, int]:
    """Find the hidden width whose head parameter count is closest to ``target_parameters``.

    Needed because the heads are not the same size at equal width: an MDN's final layer is
    ``K*(1 + 2D)`` wide against the deterministic head's ``D``, which at 3 objects makes the
    MDN 11.0k parameters against 6.9k. Without this, "the MDN wins" is confounded with "the
    MDN is 60% bigger" -- the same confound ``param_matched_baseline.py`` removes for the
    dense monolith. Returns ``(hidden_dim, parameter_count)``.
    """
    best = min(
        candidates,
        key=lambda width: abs(
            delta_head_parameters(
                delta_head_type,
                object_feature_dim=object_feature_dim,
                hidden_dim=width,
                num_layers=num_layers,
                output_dim=output_dim,
                num_components=num_components,
            )
            - target_parameters
        ),
    )
    return best, delta_head_parameters(
        delta_head_type,
        object_feature_dim=object_feature_dim,
        hidden_dim=best,
        num_layers=num_layers,
        output_dim=output_dim,
        num_components=num_components,
    )


def build_delta_head(
    delta_head_type: str,
    *,
    object_feature_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    output_dim: int = 3,
    num_components: int = 5,
) -> nn.Module:
    """Construct a delta head by name.

    ``mse`` returns the original deterministic :class:`ObjectDeltaHead`, so existing
    checkpoints keep loading against an unchanged submodule.
    """
    if delta_head_type == "mse":
        return ObjectDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )
    if delta_head_type == "gaussian":
        return GaussianDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )
    if delta_head_type == "mdn":
        return MixtureDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            num_components=num_components,
        )
    raise ValueError(
        f"Unsupported delta_head_type '{delta_head_type}'. Expected one of {DELTA_HEAD_TYPES}."
    )
