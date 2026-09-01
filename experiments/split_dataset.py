from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models import StateLayout, infer_num_objects_from_state_dim


@dataclass
class Episode:
    start: int
    end: int
    config_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a transition dataset into train/val/test at the episode level.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round-decimals", type=int, default=4)
    return parser.parse_args()


def extract_episodes(done: np.ndarray, states: np.ndarray, round_decimals: int) -> list[Episode]:
    done_indices = np.flatnonzero(done)
    if done_indices.size == 0 or done_indices[-1] != len(done) - 1:
        raise ValueError("Dataset must contain complete episodes ending with done=True.")

    episodes: list[Episode] = []
    start = 0
    for end in done_indices:
        config_id = fingerprint_initial_configuration(states[start], round_decimals)
        episodes.append(Episode(start=start, end=int(end) + 1, config_id=config_id))
        start = int(end) + 1
    return episodes


def fingerprint_initial_configuration(state: np.ndarray, round_decimals: int) -> str:
    num_objects = infer_num_objects_from_state_dim(int(state.shape[0]))
    layout = StateLayout(num_objects=num_objects)
    object_pose = state[layout.object_pose_slice]
    rounded = np.round(object_pose, decimals=round_decimals)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def assign_groups(
    config_ids: list[str], train_frac: float, val_frac: float, seed: int
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    unique_ids = list(dict.fromkeys(config_ids))
    rng.shuffle(unique_ids)

    num_groups = len(unique_ids)
    train_target = int(round(num_groups * train_frac))
    val_target = int(round(num_groups * val_frac))
    if train_target + val_target > num_groups:
        val_target = max(0, num_groups - train_target)

    assignments: dict[str, str] = {}
    for idx, config_id in enumerate(unique_ids):
        if idx < train_target:
            split = "train"
        elif idx < train_target + val_target:
            split = "val"
        else:
            split = "test"
        assignments[config_id] = split
    return assignments


def save_split(output_path: Path, dataset: dict[str, np.ndarray], indices: np.ndarray) -> None:
    payload = {key: value[indices] for key, value in dataset.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    raw = np.load(args.input)
    dataset = {key: raw[key] for key in raw.files}
    episodes = extract_episodes(dataset["done"], dataset["s_t"], args.round_decimals)

    assignments = assign_groups(
        [episode.config_id for episode in episodes],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    split_episode_counts = {"train": 0, "val": 0, "test": 0}
    split_indices = {"train": [], "val": [], "test": []}
    split_config_ids = {"train": set(), "val": set(), "test": set()}

    for episode in episodes:
        split = assignments[episode.config_id]
        split_episode_counts[split] += 1
        split_indices[split].append(np.arange(episode.start, episode.end))
        split_config_ids[split].add(episode.config_id)

    for split_name, groups in split_config_ids.items():
        other_groups = set().union(
            *(split_config_ids[name] for name in split_config_ids if name != split_name)
        )
        overlap = groups & other_groups
        if overlap:
            raise RuntimeError(f"Configuration leakage detected in split '{split_name}'.")

    split_transition_counts: dict[str, int] = {}
    for split_name, chunks in split_indices.items():
        if chunks:
            indices = np.concatenate(chunks)
        else:
            indices = np.array([], dtype=np.int64)
        split_transition_counts[split_name] = int(indices.size)
        save_split(args.output_dir / f"{args.input.stem}_{split_name}.npz", dataset, indices)

    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "round_decimals": args.round_decimals,
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "episodes_total": len(episodes),
        "transitions_total": int(dataset["done"].shape[0]),
        "split_episode_counts": split_episode_counts,
        "split_transition_counts": split_transition_counts,
        "unique_config_counts": {key: len(value) for key, value in split_config_ids.items()},
        "configuration_leakage": False,
    }

    metadata_path = args.output_dir / f"{args.input.stem}_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
