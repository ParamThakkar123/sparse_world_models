from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments import ExperimentLogger
from models.datasets import TransitionDataset
from models.dense_predictor import DenseStatePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense object-pose baseline predictor.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="dense_baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predict-delta", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--overfit-batch-size", type=int, default=None)
    return parser.parse_args()


def evaluate(
    model: DenseStatePredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            target = batch["target"].to(device)
            prediction = model(state, action)
            loss = criterion(prediction, target)
            batch_size = state.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    max_train_samples = args.max_train_samples
    max_val_samples = args.max_val_samples
    if args.overfit_batch_size is not None:
        max_train_samples = args.overfit_batch_size
        max_val_samples = args.overfit_batch_size

    train_dataset = TransitionDataset(
        args.train,
        predict_delta=args.predict_delta,
        max_samples=max_train_samples,
    )
    val_dataset = TransitionDataset(
        args.val if args.overfit_batch_size is None else args.train,
        predict_delta=args.predict_delta,
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False)

    state_dim = int(train_dataset.state.shape[1])
    action_dim = int(train_dataset.action.shape[1])
    target_dim = int(train_dataset[0]["target"].shape[0])

    model = DenseStatePredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        output_dim=target_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config(
        {
            "model": "DenseStatePredictor",
            "num_objects": train_dataset.num_objects,
            "target": "object_pose_t_plus_1",
            "loss": "L2/MSE over flattened object planar pose vector at t+1",
            "target_slice": [train_dataset.target_slice.start, train_dataset.target_slice.stop],
            "train_split": str(args.train),
            "val_split": str(args.val if args.overfit_batch_size is None else args.train),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "target_dim": target_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "predict_delta": args.predict_delta,
            "seed": args.seed,
            "device": args.device,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "overfit_batch_size": args.overfit_batch_size,
        }
    )

    best_val_loss = float("inf")
    best_epoch = -1
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"{args.run_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_count = 0
        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(state, action)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            batch_size = state.shape[0]
            total_train_loss += float(loss.item()) * batch_size
            total_count += batch_size

        train_loss = total_train_loss / max(total_count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.log_metrics(epoch, train_loss=train_loss, val_loss=val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "num_objects": train_dataset.num_objects,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "target_dim": target_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "predict_delta": args.predict_delta,
                    },
                },
                checkpoint_path,
            )

    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "overfit_mode": args.overfit_batch_size is not None,
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
