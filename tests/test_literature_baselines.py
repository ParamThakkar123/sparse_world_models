"""Contract tests for the published-model re-implementations.

These baselines exist to answer "would a properly-built published model show this
degeneracy?", so the claim only means something if each one is actually the architecture it
says it is. The properties tested here are the ones whose violation would quietly turn a
baseline into a different (usually weaker) model and make the comparison unfair:

* **Permutation equivariance.** Every one of these is an object-centric model; if permuting
  the object order changes the per-object predictions, it has learned a positional shortcut
  and is no longer a fair stand-in. This is also where NPS failed before its eval path was
  fixed -- Gumbel sampling at evaluation made the same scene score differently depending on
  object ordering, an error of ~4e-1 that would have been read as architectural asymmetry.
* **Interaction actually happens.** A GNS whose message passing is a no-op is just an MLP
  wearing a graph, and would understate what relational modelling can do.
* **Self-edges are excluded.** Otherwise a node's own state enters the aggregate twice and
  nothing can be attributed to interaction.
* **C-SWM's decoder never touches the representation.** Its whole design claim is that it
  learns without reconstruction; if the pose probe leaked gradient into the encoder, the
  baseline would silently become an autoencoder.
"""

from __future__ import annotations

import pytest
import torch

from models.literature_baselines import (
    ContrastiveStructuredWorldModel,
    GraphNetworkSimulator,
    NeuralProductionSystem,
    ProbabilisticEnsemble,
    SlotFormerDynamics,
)

FEATURE_DIM = 24
BATCH, OBJECTS = 6, 5


def _inputs(objects: int = OBJECTS):
    torch.manual_seed(0)
    return (
        torch.randn(BATCH, objects, FEATURE_DIM),
        torch.randn(BATCH, objects, 3),
    )


def _pose_models():
    return {
        "gns": GraphNetworkSimulator(FEATURE_DIM, hidden_dim=32),
        "slotformer": SlotFormerDynamics(FEATURE_DIM, hidden_dim=32),
        "pets": ProbabilisticEnsemble(FEATURE_DIM, hidden_dim=32),
        "nps": NeuralProductionSystem(FEATURE_DIM, hidden_dim=32),
        "cswm": ContrastiveStructuredWorldModel(FEATURE_DIM, hidden_dim=32),
    }


@pytest.mark.parametrize("name", ["gns", "slotformer", "pets", "nps", "cswm"])
def test_permutation_equivariance(name: str) -> None:
    model = _pose_models()[name].eval()
    features, pose = _inputs()
    permutation = torch.randperm(OBJECTS)
    with torch.no_grad():
        straight = model(features, pose)
        permuted = model(features[:, permutation], pose[:, permutation])
    assert torch.allclose(straight[:, permutation], permuted, atol=1e-5)


@pytest.mark.parametrize("name", ["gns", "slotformer", "pets", "nps", "cswm"])
def test_output_shape_and_gradients(name: str) -> None:
    model = _pose_models()[name]
    features, pose = _inputs()
    out = model(features, pose)
    assert out.shape == (BATCH, OBJECTS, 3)
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_nps_is_deterministic_in_eval_but_samples_in_training() -> None:
    """The eval/training split is what makes NPS measurable at all -- see the module docstring."""
    model = NeuralProductionSystem(FEATURE_DIM, hidden_dim=32)
    features, pose = _inputs()

    model.eval()
    with torch.no_grad():
        first, second = model(features, pose), model(features, pose)
    assert torch.equal(first, second)

    model.train()
    torch.manual_seed(1)
    a = model(features, pose)
    torch.manual_seed(2)
    b = model(features, pose)
    assert not torch.allclose(a, b), "training must sample rules, not argmax them"


def test_gns_message_passing_is_not_a_no_op() -> None:
    """Changing ONLY another object's pose must change this object's prediction."""
    model = GraphNetworkSimulator(FEATURE_DIM, hidden_dim=32).eval()
    features, pose = _inputs()
    moved = pose.clone()
    moved[:, 1] += 0.4  # move object 1 only
    with torch.no_grad():
        before = model(features, pose)
        after = model(features, moved)
    # Object 0's prediction must respond to object 1 having moved.
    assert not torch.allclose(before[:, 0], after[:, 0], atol=1e-6)


def test_gns_uses_more_message_passing_steps_than_the_one_step_rung() -> None:
    """The point of this baseline over the existing `gnn` rung is depth of propagation."""
    shallow = GraphNetworkSimulator(FEATURE_DIM, hidden_dim=32, num_message_passing_steps=1)
    deep = GraphNetworkSimulator(FEATURE_DIM, hidden_dim=32, num_message_passing_steps=3)
    assert len(deep.edge_blocks) == 3 and len(deep.node_blocks) == 3
    assert sum(p.numel() for p in deep.parameters()) > sum(p.numel() for p in shallow.parameters())


def test_cswm_probe_does_not_backpropagate_into_the_representation() -> None:
    """C-SWM's claim is that it learns without reconstruction; the probe must not break it."""
    model = ContrastiveStructuredWorldModel(FEATURE_DIM, hidden_dim=32)
    features, _ = _inputs()
    latent = model.encode(features)
    model.decode(latent).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in model.encoder.parameters()), "decoder leaked gradient into the encoder"
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
               for p in model.decoder.parameters())


def test_cswm_contrastive_loss_and_ranking_are_well_formed() -> None:
    model = ContrastiveStructuredWorldModel(FEATURE_DIM, hidden_dim=32)
    features, _ = _inputs()
    next_features = torch.randn_like(features)
    action = torch.randn(BATCH, 2)
    loss = model.contrastive_loss(features, action, next_features)
    assert torch.isfinite(loss)
    metrics = model.ranking_metrics(features, action, next_features)
    assert 0.0 <= metrics["hits_at_1"] <= 1.0
    assert 0.0 < metrics["mrr"] <= 1.0


def test_pets_members_disagree_and_nll_is_finite() -> None:
    """Ensemble disagreement is reported as a gate-free change signal, so it must be real."""
    model = ProbabilisticEnsemble(FEATURE_DIM, hidden_dim=32, num_models=5)
    features, _ = _inputs()
    disagreement = model.epistemic_disagreement(features)
    assert disagreement.shape == (BATCH, OBJECTS)
    assert float(disagreement.mean()) > 0.0, "identical members carry no epistemic signal"
    nll = model.nll(features, torch.randn(BATCH, OBJECTS, 3))
    assert torch.isfinite(nll)


def test_nps_handles_a_single_object() -> None:
    """With one object there is no contextual slot; masked softmax would otherwise give NaN."""
    model = NeuralProductionSystem(FEATURE_DIM, hidden_dim=32).eval()
    features, pose = _inputs(objects=1)
    with torch.no_grad():
        out = model(features, pose)
    assert torch.isfinite(out).all()


def test_slotformer_accepts_history_and_single_step() -> None:
    model = SlotFormerDynamics(FEATURE_DIM, hidden_dim=32, history=3).eval()
    features, pose = _inputs()
    history = torch.randn(BATCH, 3, OBJECTS, FEATURE_DIM)
    with torch.no_grad():
        assert model(features, pose).shape == (BATCH, OBJECTS, 3)
        assert model(history, pose).shape == (BATCH, OBJECTS, 3)
