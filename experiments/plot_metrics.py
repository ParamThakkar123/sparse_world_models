from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from an experiment metrics CSV.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = []
    train_loss = []
    val_loss = []

    with args.metrics.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax.plot(epochs, val_loss, label="val_loss", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Dense Baseline Loss Curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
