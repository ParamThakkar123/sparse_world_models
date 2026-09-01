from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import DenseStatePredictor, POSE_DIM, StateLayout, infer_num_objects_from_state_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense baseline on pose prediction metrics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = DenseStatePredictor(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        output_dim=config["target_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(args.data)
    state = torch.from_numpy(data["s_t"]).float().to(device)
    action = torch.from_numpy(data["a_t"]).float().to(device)
    num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    target_pose = torch.from_numpy(data["s_t1"][:, layout.object_pose_slice]).float().to(device)
    change_mask = data["object_change_mask"].astype(bool)

    with torch.no_grad():
        prediction = model(state, action).cpu().numpy().reshape(-1, num_objects, POSE_DIM)
    target = target_pose.cpu().numpy().reshape(-1, num_objects, POSE_DIM)

    error = prediction - target
    abs_error = np.abs(error)
    l2_error = np.linalg.norm(error, axis=2)

    changed = change_mask
    unchanged = ~change_mask

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        count = int(mask.sum())
        if count == 0:
            return 0.0
        return float(values[mask].mean())

    summary = {
        "num_transitions": int(target.shape[0]),
        "num_objects_per_transition": int(target.shape[1]),
        "overall_pose_mse": float(np.mean(error ** 2)),
        "overall_pose_mae": float(np.mean(abs_error)),
        "overall_per_object_l2": float(np.mean(l2_error)),
        "changed_object_fraction": float(changed.mean()),
        "changed_pose_mae": masked_mean(abs_error, np.repeat(changed[:, :, None], POSE_DIM, axis=2)),
        "unchanged_pose_mae": masked_mean(abs_error, np.repeat(unchanged[:, :, None], POSE_DIM, axis=2)),
        "changed_object_l2": masked_mean(l2_error, changed),
        "unchanged_object_l2": masked_mean(l2_error, unchanged),
        "per_object_mae": {
            f"object_{idx}": float(abs_error[:, idx, :].mean()) for idx in range(target.shape[1])
        },
        "per_object_l2": {
            f"object_{idx}": float(l2_error[:, idx].mean()) for idx in range(target.shape[1])
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
