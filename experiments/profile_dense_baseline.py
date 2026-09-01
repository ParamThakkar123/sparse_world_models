from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from thop import profile

from models.dense_predictor import DenseStatePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile dense baseline compute stats.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--timing-iters", type=int, default=200)
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

    state = torch.randn(args.batch_size, config["state_dim"], device=device)
    action = torch.randn(args.batch_size, config["action_dim"], device=device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(state, action), verbose=False)

        for _ in range(args.warmup_iters):
            _ = model(state, action)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(args.timing_iters):
            _ = model(state, action)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / args.timing_iters) * 1000.0
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_parameters": int(params),
        "macs_per_forward": float(macs),
        "flops_per_forward_estimate": float(macs * 2.0),
        "avg_inference_latency_ms": avg_latency_ms,
        "timing_iters": args.timing_iters,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
