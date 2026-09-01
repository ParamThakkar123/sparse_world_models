from __future__ import annotations

import torch

from experiments.train_sparse_model import compute_gate_metrics
from models.sparse_residual import masked_delta_l2_loss, sparse_residual_loss


def test_masked_delta_l2_loss_respects_gate_weights() -> None:
    pred_delta = torch.tensor([[[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    target_delta = torch.zeros_like(pred_delta)
    changed_mask = torch.tensor([[1.0, 1.0]])
    gate_weights = torch.tensor([[1.0, 0.25]])

    loss = masked_delta_l2_loss(
        pred_delta,
        target_delta,
        changed_mask,
        gate_weights=gate_weights,
    )

    expected = torch.tensor((4.0 + 25.0) / (3.0 * 1.25))
    assert torch.isclose(loss, expected)


def test_sparse_residual_loss_uses_delta_gate_weighting() -> None:
    gate_logits = torch.zeros((1, 2))
    gate_probs = torch.tensor([[1.0, 0.0]])
    pred_delta = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]])
    target_changed_mask = torch.tensor([[1.0, 1.0]])
    target_delta = torch.zeros_like(pred_delta)

    losses = sparse_residual_loss(
        gate_logits,
        gate_probs,
        pred_delta,
        target_changed_mask,
        target_delta,
        delta_gate=gate_probs,
    )

    assert torch.isclose(losses.delta_l2, torch.tensor(1.0))


def test_compute_gate_metrics_reports_collapse_diagnostics() -> None:
    pred_mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    target_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    metrics = compute_gate_metrics(pred_mask, target_mask)

    assert metrics["gate_precision"] == 2 / 3
    assert metrics["gate_recall"] == 1.0
    assert metrics["gate_all_changed_fraction"] == 0.5
    assert metrics["gate_all_unchanged_fraction"] == 0.5
