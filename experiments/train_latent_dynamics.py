"""Train the latent dynamics baseline (W5 control).

See ``models/latent_dynamics.py`` for why this exists: without a *learned model* baseline,
"sparse beats scripted and random" cannot distinguish object-centric structure from any
learned model at all. This trains the TD-MPC2/Dreamer-style latent core on exactly the data
the sparse and dense models see, so the planner and the data are held fixed and only the
representation varies.

Usage::

    python -m experiments.train_latent_dynamics \\
      --train data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_train.npz \\
      --val   data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_val.npz \\
      --run-name latent_plan_3obj_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments import ExperimentLogger
from models import POSE_DIM, StateLayout, TransitionDataset, infer_num_objects_from_state_dim
from models.latent_dynamics import LatentDynamicsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent dynamics control baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="latent_dynamics")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer, args, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"total": 0.0, "reconstruction": 0.0, "consistency": 0.0, "prediction": 0.0}
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            num_objects = int(batch["object_change_mask"].shape[1])
            current_pose = batch["current_object_pose"].reshape(-1, num_objects, POSE_DIM)
            next_pose = batch["next_object_pose"].reshape(-1, num_objects, POSE_DIM)
            losses = model.losses(
                batch["state"], batch["action"], batch["next_state"],
                current_pose, next_pose,
                consistency_weight=args.consistency_weight,
                reconstruction_weight=args.reconstruction_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                optimizer.step()
            size = batch["state"].shape[0]
            totals["total"] += float(losses.total.item()) * size
            totals["reconstruction"] += float(losses.reconstruction.item()) * size
            totals["consistency"] += float(losses.consistency.item()) * size
            totals["prediction"] += float(losses.prediction.item()) * size
            count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_dataset = TransitionDataset(args.train)
    val_dataset = TransitionDataset(args.val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0]
    state_dim = int(sample["state"].shape[0])
    action_dim = int(sample["action"].shape[0])
    num_objects = infer_num_objects_from_state_dim(state_dim)

    model = LatentDynamicsModel(
        state_dim=state_dim, action_dim=action_dim, num_objects=num_objects,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_parameters = sum(p.numel() for p in model.parameters())

    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config({
        "model": "LatentDynamicsModel", "train_split": str(args.train),
        "val_split": str(args.val), "num_objects": num_objects,
        "state_dim": state_dim, "action_dim": action_dim,
        "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "num_parameters": num_parameters,
        "epochs": args.epochs, "lr": args.lr,
        "consistency_weight": args.consistency_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "seed": args.seed,
    })

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"{args.run_name}.pt"
    best_val, best_epoch, best_metrics = float("inf"), -1, None

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, args, train=False)
        logger.log_metrics(
            epoch,
            train_total=train_metrics["total"], val_total=val_metrics["total"],
            train_prediction=train_metrics["prediction"], val_prediction=val_metrics["prediction"],
            train_consistency=train_metrics["consistency"], val_consistency=val_metrics["consistency"],
            train_reconstruction=train_metrics["reconstruction"],
            val_reconstruction=val_metrics["reconstruction"],
        )
        # Selected on the prediction term, not the total: that is the quantity the planner
        # consumes, and the other two are auxiliaries that shape the latent.
        if val_metrics["prediction"] < best_val:
            best_val, best_epoch, best_metrics = val_metrics["prediction"], epoch, val_metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "state_dim": state_dim, "action_dim": action_dim,
                    "num_objects": num_objects, "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
                    "num_parameters": num_parameters,
                },
            }, checkpoint_path)

    summary = {
        "best_val_prediction": best_val, "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path), "num_parameters": num_parameters,
        "best_val_metrics": best_metrics,
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
