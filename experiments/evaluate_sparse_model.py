from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.train_sparse_model import (
    build_object_features,
    compute_gate_metrics,
    compute_pose_metrics,
    resolve_gate_estimator,
    select_delta_supervision,
)
from models import SparseResidualHead, TransitionDataset
from models.sparse_residual import sparse_residual_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sparse residual model on test metrics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = SparseResidualHead(
        object_feature_dim=config["feature_dim"],
        gate_hidden_dim=config["gate_hidden_dim"],
        gate_num_layers=config["gate_num_layers"],
        delta_hidden_dim=config["delta_hidden_dim"],
        delta_num_layers=config["delta_num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TransitionDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total = {
        "loss": 0.0,
        "gate_bce": 0.0,
        "delta_l2": 0.0,
        "sparsity_penalty": 0.0,
        "pose_l2": 0.0,
        "changed_pose_l2": 0.0,
        "unchanged_pose_l2": 0.0,
    }
    pred_masks = []
    target_masks = []
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            object_features = build_object_features(batch["state"], batch["action"])
            out = model(
                object_features,
                estimator=config.get("eval_estimator", resolve_gate_estimator(False, config["estimator"])),
                temperature=config["temperature"],
                hard=True,
            )
            losses = sparse_residual_loss(
                out.gate.logits,
                out.gate.probs,
                out.delta,
                batch["object_change_mask"],
                batch["object_delta"],
                gate_loss_weight=config["gate_loss_weight"],
                delta_loss_weight=config["delta_loss_weight"],
                sparsity_weight=config["sparsity_weight"],
                positive_class_weight=config.get("positive_class_weight"),
                delta_gate=select_delta_supervision(
                    config.get("delta_supervision", "predicted_probs"),
                    batch,
                    out.gate.probs,
                    out.gate.gates,
                ),
            )
            batch_size = batch["state"].shape[0]
            pose_metrics = compute_pose_metrics(out.masked_delta, batch)
            total["loss"] += float(losses.total.item()) * batch_size
            total["gate_bce"] += float(losses.gate_bce.item()) * batch_size
            total["delta_l2"] += float(losses.delta_l2.item()) * batch_size
            total["sparsity_penalty"] += float(losses.sparsity_penalty.item()) * batch_size
            total["pose_l2"] += pose_metrics["pose_l2"] * batch_size
            total["changed_pose_l2"] += pose_metrics["changed_pose_l2"] * batch_size
            total["unchanged_pose_l2"] += pose_metrics["unchanged_pose_l2"] * batch_size
            total_count += batch_size
            pred_masks.append((out.gate.probs >= 0.5).float().cpu())
            target_masks.append(batch["object_change_mask"].cpu())

    summary = {key: value / max(total_count, 1) for key, value in total.items()}
    summary.update(compute_gate_metrics(torch.cat(pred_masks, dim=0), torch.cat(target_masks, dim=0)))
    summary["num_transitions"] = len(dataset)
    summary["checkpoint"] = str(args.checkpoint)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
