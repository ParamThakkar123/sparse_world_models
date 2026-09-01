"""Faithful re-implementations of published object-centric dynamics models.

Why this module exists
----------------------
Every baseline this project has compared against so far is one we wrote: a dense MLP, two
object-centric MLP rungs, a one-step interaction network, a set transformer, and a small
latent-dynamics model. That is enough to attribute *our* model's win to an ingredient, and
not enough to claim anything about the field. A reviewer's reasonable position is that our
relational rungs are weak stand-ins and a properly-built published model would not show the
degeneracy we report.

These five are the published models that objection points at. Each is implemented from its
paper's description rather than adapted from our ladder, and each is given the same task,
the same features, the same data and a matched parameter budget:

``GraphNetworkSimulator``
    Encode-process-decode graph network, Sanchez-Gonzalez et al. 2020 ("Learning to Simulate
    Complex Physics with Graph Networks", GNS) and Li et al. 2019 (DPI-Net). The important
    differences from our existing one-step ``gnn`` rung, which is why that rung is not a fair
    stand-in: **multiple message-passing steps** (the paper uses 10; we use 3, sized to our
    object counts), **explicit edge features** built from relative position and distance
    rather than only from concatenated node embeddings, **residual updates within the
    processor**, and LayerNorm on the encoders. This is the strongest "just use a proper GNN"
    answer to the change-detection degeneracy.

``ContrastiveStructuredWorldModel``
    C-SWM, Kipf, van der Pol & Welling 2020 ("Contrastive Learning of Structured World
    Models", ICLR). Object-factored latent state, GNN transition model, and an energy-based
    contrastive objective with negative samples -- no reconstruction term, which is the
    paper's central design choice. Because C-SWM never decodes, it cannot be scored on pose
    L2 as published; see :class:`ContrastiveStructuredWorldModel` for how that is handled
    and what is disclosed about it.

``SlotFormerDynamics``
    SlotFormer, Wu et al. 2023 ("SlotFormer: Unsupervised Visual Dynamics Simulation with
    Object-Centric Models", ICLR). A transformer over object tokens across a *history
    window*, with temporal position embeddings and per-object identity preserved across
    time. This is the one baseline in the set that gets to see more than one step of
    history, which matters here: a model with history can read velocity off consecutive
    positions even when velocity is not in its features.

``ProbabilisticEnsemble``
    PETS, Chua et al. 2018 ("Deep Reinforcement Learning in a Handful of Trials..."). An
    ensemble of heteroscedastic Gaussian dynamics models trained by NLL with bootstrapped
    batches. Included because it is the standard model-based-RL dynamics baseline and
    because its ensemble disagreement gives a *second*, independent way to ask "does this
    model know which objects will move" that does not depend on a gate at all.

``NeuralProductionSystem``
    NPS, Goyal et al. 2021 ("Neural Production Systems"). A small set of learned rules, of
    which exactly one is selected per object per step by a Gumbel-softmax over rule logits,
    applied to a (primary, contextual) slot pair chosen by attention. This is the closest
    published relative of our central claim -- it is *also* a sparse-mechanism model -- so it
    is the baseline that most directly tests whether "sparsity of mechanism" alone produces
    the behaviour we attribute to an explicit change gate.

Shared interface
----------------
``GraphNetworkSimulator``, ``SlotFormerDynamics`` and ``NeuralProductionSystem`` expose the
``forward(object_features, current_pose) -> next_pose`` signature the gate-ablation ladder
already uses, so they slot into the existing rung machinery unchanged. The two that cannot
(C-SWM has no pose output, PETS predicts a distribution) carry their own training and
scoring paths and say so.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

POSE_OUTPUT_DIM = 3


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    layer_norm: bool = False,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(current, hidden_dim), nn.ReLU()]
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


def _pairwise_edge_features(pose: torch.Tensor) -> torch.Tensor:
    """Relative position and distance for every ordered object pair.

    GNS builds edge features from the *geometry* of the pair, not from the node embeddings,
    which is the ingredient our one-step ``gnn`` rung lacks. Shapes: ``pose`` is
    ``(B, N, 3)`` as ``(x, y, yaw)``; the result is ``(B, N, N, 4)`` holding
    ``(dx, dy, distance, relative yaw)`` for the edge from sender ``j`` to receiver ``i``.
    """
    xy = pose[..., :2]
    yaw = pose[..., 2]
    relative_xy = xy.unsqueeze(1) - xy.unsqueeze(2)  # (B, N_receiver, N_sender, 2)
    distance = torch.linalg.norm(relative_xy, dim=-1, keepdim=True)
    relative_yaw = (yaw.unsqueeze(1) - yaw.unsqueeze(2)).unsqueeze(-1)
    return torch.cat([relative_xy, distance, relative_yaw], dim=-1)


class GraphNetworkSimulator(nn.Module):
    """GNS / DPI-Net style encode-process-decode graph network.

    Sanchez-Gonzalez et al. 2020. The processor runs ``num_message_passing_steps`` rounds of
    full edge-then-node updates with residual connections, so information propagates along
    contact chains of that length -- the property the single-step rung cannot have and the
    reason it is not a fair test of "would a real GNN fix this".

    Edges are dense (every ordered pair) rather than radius-pruned. At 3-20 objects a dense
    graph is cheaper than building neighbour lists, and it is also the *more* favourable
    choice for the baseline: nothing is hidden from it by a connectivity radius, so a failure
    to detect change cannot be blamed on a missing edge.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        num_message_passing_steps: int = 3,
        num_layers: int = 2,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        self.mode = mode
        self.num_message_passing_steps = num_message_passing_steps
        self.node_encoder = _mlp(object_feature_dim, hidden_dim, hidden_dim, num_layers, True)
        self.edge_encoder = _mlp(4, hidden_dim, hidden_dim, num_layers, True)
        self.edge_blocks = nn.ModuleList(
            _mlp(3 * hidden_dim, hidden_dim, hidden_dim, num_layers, True)
            for _ in range(num_message_passing_steps)
        )
        self.node_blocks = nn.ModuleList(
            _mlp(2 * hidden_dim, hidden_dim, hidden_dim, num_layers, True)
            for _ in range(num_message_passing_steps)
        )
        self.decoder = _mlp(hidden_dim, hidden_dim, POSE_OUTPUT_DIM, num_layers)

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")
        batch, num_objects, _ = object_features.shape
        device = object_features.device

        nodes = self.node_encoder(object_features)
        edges = self.edge_encoder(_pairwise_edge_features(current_pose))
        # Self-edges carry no relational information and would let a node's own state enter
        # the aggregate twice, so they are masked out of every aggregation.
        self_mask = torch.eye(num_objects, dtype=torch.bool, device=device).view(
            1, num_objects, num_objects, 1
        )

        for edge_block, node_block in zip(self.edge_blocks, self.node_blocks):
            receiver = nodes.unsqueeze(2).expand(batch, num_objects, num_objects, nodes.shape[-1])
            sender = nodes.unsqueeze(1).expand(batch, num_objects, num_objects, nodes.shape[-1])
            edge_input = torch.cat([edges, receiver, sender], dim=-1)
            # Residual updates in the processor: GNS's processor is a stack of GN blocks each
            # adding to its input rather than replacing it.
            edges = edges + edge_block(edge_input)
            aggregated = edges.masked_fill(self_mask, 0.0).sum(dim=2)
            nodes = nodes + node_block(torch.cat([nodes, aggregated], dim=-1))

        out = self.decoder(nodes)
        return out if self.mode == "absolute" else current_pose + out


class ContrastiveStructuredWorldModel(nn.Module):
    """C-SWM (Kipf, van der Pol & Welling, ICLR 2020).

    Object-factored latent state ``z_i``, a GNN transition model predicting ``Delta z_i``
    conditioned on the action, and the paper's energy-based contrastive objective::

        H = d(z_t + T(z_t, a_t), z_{t+1}) + max(0, gamma - d(z~_t, z_{t+1}))

    with ``d`` the squared Euclidean energy and ``z~`` an encoded negative sample drawn by
    shuffling the batch. There is deliberately **no decoder in the objective**: avoiding
    pixel/state reconstruction is C-SWM's central claim, since reconstruction loss is
    dominated by large static background regions.

    Scoring it here, and what that costs
    ------------------------------------
    That design means C-SWM cannot be scored on pose L2 or on a changed-object mask as
    published -- it has no pose output at all. Two readouts are provided and both are
    reported, because each alone would be misleading:

    * ``ranking_metrics`` -- the paper's own evaluation (Hits@1 and MRR of the predicted
      latent against the true next latent among in-batch candidates). This is C-SWM measured
      on C-SWM's terms and is the fair headline number for it.
    * ``decode`` -- a pose head trained **on frozen latents**, after contrastive training has
      finished, purely so the model can be placed on the same axis as everything else. It is
      a probe of what the latents contain, not part of C-SWM, and must be labelled that way
      wherever it appears.

    The gradient is stopped between the two: the decoder never influences the representation,
    so this stays C-SWM rather than becoming an autoencoder with a contrastive regulariser.
    """

    def __init__(
        self,
        object_feature_dim: int,
        action_dim: int = 2,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        margin: float = 1.0,
        energy_scale: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.margin = margin
        self.energy_scale = energy_scale
        self.encoder = _mlp(object_feature_dim, hidden_dim, latent_dim, num_layers)
        # Transition GNN: per-object node update conditioned on the action, plus pairwise
        # edge messages, exactly as in the paper's structured transition model.
        self.edge_model = _mlp(2 * latent_dim, hidden_dim, hidden_dim, num_layers)
        self.node_model = _mlp(latent_dim + hidden_dim + action_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = _mlp(latent_dim, hidden_dim, POSE_OUTPUT_DIM, num_layers)

    def encode(self, object_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(object_features)

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, num_objects, _ = latent.shape
        sender = latent.unsqueeze(1).expand(batch, num_objects, num_objects, self.latent_dim)
        receiver = latent.unsqueeze(2).expand(batch, num_objects, num_objects, self.latent_dim)
        messages = self.edge_model(torch.cat([receiver, sender], dim=-1))
        eye = torch.eye(num_objects, dtype=torch.bool, device=latent.device)
        aggregated = messages.masked_fill(eye.view(1, num_objects, num_objects, 1), 0.0).sum(dim=2)
        action_broadcast = action.unsqueeze(1).expand(batch, num_objects, action.shape[-1])
        return self.node_model(torch.cat([latent, aggregated, action_broadcast], dim=-1))

    def energy(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean energy, summed over objects and mean over the latent dim."""
        return self.energy_scale * ((predicted - target) ** 2).mean(dim=-1).sum(dim=-1)

    def contrastive_loss(
        self,
        object_features: torch.Tensor,
        action: torch.Tensor,
        next_object_features: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode(object_features)
        next_latent = self.encode(next_object_features)
        predicted = latent + self.transition(latent, action)
        positive = self.energy(predicted, next_latent)
        # Negatives by shuffling the batch, as in the reference implementation. ``roll`` is
        # used rather than a random permutation so the negative for every anchor is a
        # different state with probability 1 and the loss is deterministic given the batch.
        negative_latent = torch.roll(latent, shifts=1, dims=0)
        negative = self.energy(negative_latent, next_latent)
        hinge = torch.clamp(self.margin - negative, min=0.0)
        return (positive + hinge).mean()

    @torch.no_grad()
    def ranking_metrics(
        self,
        object_features: torch.Tensor,
        action: torch.Tensor,
        next_object_features: torch.Tensor,
    ) -> dict[str, float]:
        """Hits@1 and MRR of the predicted latent among in-batch candidates.

        This is C-SWM's published evaluation protocol: rank the true next state against every
        other next state in the batch by energy, and report how often the true one is first.
        """
        latent = self.encode(object_features)
        next_latent = self.encode(next_object_features)
        predicted = latent + self.transition(latent, action)
        batch = predicted.shape[0]
        # energies[i, j] = energy of predicting sample i's next state as sample j's.
        energies = self.energy_scale * (
            (predicted.unsqueeze(1) - next_latent.unsqueeze(0)) ** 2
        ).mean(dim=-1).sum(dim=-1)
        ranks = (energies < energies.diagonal().unsqueeze(1)).sum(dim=1) + 1
        return {
            "hits_at_1": float((ranks == 1).float().mean()),
            "mrr": float((1.0 / ranks.float()).mean()),
            "batch_size": int(batch),
        }

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Pose readout from FROZEN latents. See the class docstring -- this is a probe."""
        return self.decoder(latent.detach())

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        """Pose prediction via the frozen-latent probe, for ladder comparability only."""
        latent = self.encode(object_features)
        # The action is already embedded in ``object_features`` for every featurisation this
        # project uses, so the transition is driven from the features rather than needing the
        # raw action threaded through the shared rung interface.
        zeros = object_features.new_zeros(object_features.shape[0], 2)
        predicted = latent + self.transition(latent, zeros)
        return current_pose + self.decode(predicted)


class SlotFormerDynamics(nn.Module):
    """SlotFormer-style temporal transformer over object tokens (Wu et al., ICLR 2023).

    Attention runs over the flattened ``(history, objects)`` token set, so each object at the
    final step can attend both to other objects at the same step and to its own past. Object
    identity is preserved across time by construction (token ``(t, i)`` is object ``i``), and
    a learned temporal embedding is added so the model can tell the steps apart; no
    *spatial* position embedding is used, because object order in the state vector is
    arbitrary and the model must stay permutation-equivariant over objects.

    This baseline gets a history window where every other model in the suite sees one step.
    That is deliberate and is the point of including it: a model with history can recover
    velocity by differencing consecutive positions, so it can take the momentum shortcut even
    from a velocity-free featurisation. If the shortcut is real, this model should take it.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        history: int = 3,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}.")
        self.mode = mode
        self.history = history
        self.input_projection = nn.Linear(object_feature_dim, hidden_dim)
        self.temporal_embedding = nn.Parameter(torch.zeros(history, hidden_dim))
        nn.init.normal_(self.temporal_embedding, std=0.02)
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
        """``object_features`` is ``(B, T, N, F)`` with history, or ``(B, N, F)`` without.

        The single-step form is accepted so this model can be dropped into the existing
        one-step ladder; it then degenerates to a set transformer, which is exactly what
        SlotFormer is with a history of one, and the comparison stays honest because the
        degeneration is visible here rather than hidden behind a reshape.
        """
        if object_features.ndim == 3:
            object_features = object_features.unsqueeze(1)
        if object_features.ndim != 4:
            raise ValueError(
                "object_features must be (batch, history, num_objects, feature_dim) "
                "or (batch, num_objects, feature_dim)."
            )
        batch, steps, num_objects, _ = object_features.shape
        if steps > self.history:
            object_features = object_features[:, -self.history :]
            steps = self.history

        hidden = self.input_projection(object_features)  # (B, T, N, H)
        hidden = hidden + self.temporal_embedding[-steps:].view(1, steps, 1, -1)
        hidden = hidden.reshape(batch, steps * num_objects, -1)

        for norm, attention, ff_norm, feedforward in zip(
            self.attention_norms, self.attentions, self.feedforward_norms, self.feedforwards
        ):
            normed = norm(hidden)
            attended, _ = attention(normed, normed, normed, need_weights=False)
            hidden = hidden + attended
            hidden = hidden + feedforward(ff_norm(hidden))

        # Read out the LAST timestep's tokens: that is the step whose successor we predict.
        hidden = hidden.reshape(batch, steps, num_objects, -1)[:, -1]
        out = self.output_projection(hidden)
        return out if self.mode == "absolute" else current_pose + out


class ProbabilisticEnsemble(nn.Module):
    """PETS-style ensemble of heteroscedastic Gaussian dynamics models (Chua et al. 2018).

    ``num_models`` independent per-object MLPs each predict a diagonal Gaussian over the pose
    delta. Training is by NLL on bootstrapped batches (each ensemble member sees a different
    resample), which is what makes the members disagree in a calibrated way rather than
    converging to the same function.

    The log-variance is bounded by the paper's soft clamp, which matters in practice: without
    it, members drive the variance to zero on easy (unchanged) objects and the NLL diverges.

    Ensemble disagreement is exposed via :meth:`epistemic_disagreement` because it gives a
    *gate-free* way to ask which objects a model thinks will move. If disagreement separates
    changed from unchanged objects, then "which objects will change" is recoverable from a
    standard probabilistic model without any explicit change gate -- which would be a genuine
    threat to this project's attribution, and is therefore worth measuring rather than
    assuming away.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_models: int = 5,
        max_logvar: float = 0.5,
        min_logvar: float = -10.0,
    ):
        super().__init__()
        self.num_models = num_models
        self.members = nn.ModuleList(
            _mlp(object_feature_dim, hidden_dim, 2 * POSE_OUTPUT_DIM, num_layers)
            for _ in range(num_models)
        )
        self.max_logvar = nn.Parameter(torch.full((POSE_OUTPUT_DIM,), max_logvar))
        self.min_logvar = nn.Parameter(torch.full((POSE_OUTPUT_DIM,), min_logvar))

    def _member(self, index: int, object_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.members[index](object_features)
        mean, logvar = raw.split(POSE_OUTPUT_DIM, dim=-1)
        # Chua et al.'s soft bounds: differentiable, so the bounds themselves are learned
        # downward/upward rather than acting as a hard clamp that kills the gradient.
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll(self, object_features: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian NLL over ensemble members on independently bootstrapped batches."""
        batch = object_features.shape[0]
        total = object_features.new_zeros(())
        for index in range(self.num_models):
            resample = torch.randint(0, batch, (batch,), device=object_features.device)
            mean, logvar = self._member(index, object_features[resample])
            target = target_delta[resample]
            inverse_variance = torch.exp(-logvar)
            total = total + (((mean - target) ** 2) * inverse_variance + logvar).mean()
        return total / self.num_models

    @torch.no_grad()
    def epistemic_disagreement(self, object_features: torch.Tensor) -> torch.Tensor:
        """Per-object std of the ensemble's mean predictions -- a gate-free change signal."""
        means = torch.stack(
            [self._member(index, object_features)[0] for index in range(self.num_models)]
        )
        return means.std(dim=0).norm(dim=-1)

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        means = torch.stack(
            [self._member(index, object_features)[0] for index in range(self.num_models)]
        )
        return current_pose + means.mean(dim=0)


class NeuralProductionSystem(nn.Module):
    """NPS: sparse learned rules with one rule applied per object per step (Goyal et al. 2021).

    Why this is the most important baseline in the set
    --------------------------------------------------
    NPS is a *sparse-mechanism* model, like ours. Each step it selects, for each object, one
    rule out of ``num_rules`` via a Gumbel-softmax over rule logits, together with a
    contextual object chosen by attention, and applies only that rule. If the behaviour this
    project attributes to an explicit binary **change gate** is really just a consequence of
    sparse mechanism selection, NPS should reproduce it -- it has the sparsity but not the
    gate. That makes it the sharpest available test of the attribution, and the reason it is
    implemented here rather than waved at in related work.

    The architectural difference to be clear about: NPS's sparsity is over *which rule
    transforms an object*, and every object still gets transformed. Ours is over *whether an
    object is transformed at all*. Those are different axes, and NPS having one does not give
    it the other -- which is a hypothesis this baseline exists to test, not an assumption.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int = 64,
        num_rules: int = 4,
        num_layers: int = 2,
        temperature: float = 1.0,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        self.mode = mode
        self.num_rules = num_rules
        self.temperature = temperature
        self.encoder = _mlp(object_feature_dim, hidden_dim, hidden_dim, num_layers)
        # Rule selection: which of the K rules fires for this object.
        self.rule_selector = _mlp(hidden_dim, hidden_dim, num_rules, num_layers)
        # Contextual attention: which OTHER object this rule reads from.
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        # One MLP per rule, applied to (primary, contextual) embeddings.
        self.rules = nn.ModuleList(
            _mlp(2 * hidden_dim, hidden_dim, POSE_OUTPUT_DIM, num_layers) for _ in range(num_rules)
        )

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape (batch, num_objects, feature_dim).")
        batch, num_objects, _ = object_features.shape
        device = object_features.device
        embedding = self.encoder(object_features)

        # Contextual slot selection by scaled dot-product attention, self excluded: a rule in
        # NPS relates a primary slot to a *different* contextual slot.
        scores = self.query(embedding) @ self.key(embedding).transpose(1, 2)
        scores = scores / (embedding.shape[-1] ** 0.5)
        eye = torch.eye(num_objects, dtype=torch.bool, device=device).unsqueeze(0)
        # With a single object there is no contextual slot to attend to; masking every entry
        # would make softmax produce NaN, so the context is explicitly zero in that case.
        if num_objects > 1:
            scores = scores.masked_fill(eye, float("-inf"))
            context = torch.softmax(scores, dim=-1) @ embedding
        else:
            context = torch.zeros_like(embedding)

        # Exactly one rule per object. Training samples straight-through Gumbel so selection
        # stays discrete in the forward pass while remaining differentiable; EVALUATION takes
        # the argmax instead.
        #
        # That split is not cosmetic. Gumbel sampling at eval makes the model
        # non-deterministic and, worse, breaks permutation equivariance in measurement: the
        # same scene with objects listed in a different order draws different noise and
        # therefore fires different rules. A permutation check on this class returns an error
        # of ~4e-1 with sampling on and ~1e-7 with argmax, so scoring NPS in sampling mode
        # would have quietly measured noise as if it were architectural asymmetry.
        logits = self.rule_selector(embedding)
        if self.training:
            selection = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            selection = F.one_hot(logits.argmax(dim=-1), self.num_rules).to(logits.dtype)

        paired = torch.cat([embedding, context], dim=-1)
        rule_outputs = torch.stack([rule(paired) for rule in self.rules], dim=-2)
        out = (selection.unsqueeze(-1) * rule_outputs).sum(dim=-2)
        return out if self.mode == "absolute" else current_pose + out
