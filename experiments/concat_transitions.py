"""Concatenate multiple transition ``.npz`` files into one dataset.

Used to build a *mixed-policy* training set for planning: scripted rollouts (clean
goal-directed pushes) plus random rollouts (diverse pusher approach angles and
near-rest contact states the planner actually queries). All inputs must share the
same object count and key set; row-aligned arrays are concatenated in order, so
episode boundaries (``done``) are preserved end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def concat_datasets(inputs: list[Path], output: Path) -> dict[str, int]:
    if not inputs:
        raise SystemExit("Provide at least one --input.")
    per_file = [dict(np.load(path)) for path in inputs]
    keys = set(per_file[0].keys())
    for path, data in zip(inputs[1:], per_file[1:]):
        if set(data.keys()) != keys:
            raise ValueError(f"Key mismatch: {path} has {set(data.keys())} vs {keys}.")

    merged = {key: np.concatenate([data[key] for data in per_file], axis=0) for key in keys}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)  # type: ignore[arg-type]  # keys are data arrays, not savez flags
    return {str(path): int(data["s_t"].shape[0]) for path, data in zip(inputs, per_file)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate transition .npz datasets.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = concat_datasets(args.inputs, args.output)
    total = sum(counts.values())
    print(f"Wrote {total} transitions to {args.output}")
    for path, count in counts.items():
        print(f"  {count:>7} from {path}")


if __name__ == "__main__":
    main()
