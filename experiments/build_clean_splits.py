"""Build episode-disjoint train/val/test splits, with the hard subset derived *from* them.

Fixes a leak in the existing pipeline. The current order of operations is
``create_hard_subset`` then ``split_dataset``, and ``create_hard_subset`` sets
``done=True`` at the end of **every kept chunk**, not only at true episode ends. So when
``split_dataset`` later calls ``extract_episodes`` on the filtered file it sees each chunk
as an independent episode and fingerprints it by that chunk's *first row* -- mid-trajectory
object poses, not the episode's initial configuration. Two chunks carved out of one
trajectory therefore get different fingerprints and are assigned to splits independently.

The ``configuration_leakage: false`` guard does not catch this: it only checks that a given
fingerprint never appears in two splits, and these fingerprints genuinely differ. Measured
on the 3-object seed-0 data, 765 chunks come from 247 source episodes and **62 of those
episodes (25%) have chunks in both the train and the test split**, with 107 spanning more
than one split. Train and test then hold states from the same rollout a few simulator
steps apart, which inflates test metrics.

This script reverses the order: split the *unfiltered* data first, at the level of whole
episodes keyed by true initial configuration, then apply the motion filter independently
within each split. The hard splits are subsets of the full splits by construction, so both
families are episode-disjoint and mutually consistent -- and the full splits additionally
retain the long contiguous runs that multi-step rollout training needs (the pre-existing
hard splits offer only 56 windows of length 5 at 3 objects).

Emits, per (count, seed), into ``data/transitions/splits_clean_{N}obj_s{S}/``::

    scale_{N}obj_s{S}_full_{train,val,test}.npz   # rollout training
    scale_{N}obj_s{S}_hard_{train,val,test}.npz   # one-step training / headline metrics
    split_metadata.json

The old ``splits_{N}obj_s{S}`` directories are left untouched so previously published
numbers remain reproducible; results computed on these clean splits are not directly
comparable to them and must be regenerated for every model being compared.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.create_hard_subset import (
    compute_interaction_keep_mask,
    build_filtered_dataset,
    compute_keep_mask,
    compute_onset_keep_mask,
)
from experiments.split_dataset import assign_groups, extract_episodes

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build episode-disjoint clean splits.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--input-template", type=str, default="data/transitions/scale_{n}obj_s{seed}.npz")
    parser.add_argument("--output-template", type=str, default="data/transitions/splits_clean_{n}obj_s{seed}")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--round-decimals", type=int, default=4)
    parser.add_argument("--min-max-xy-delta", type=float, default=0.02,
                        help="Motion threshold for the hard subset (matches create_hard_subset).")
    parser.add_argument("--min-changed-objects", type=int, default=1)
    parser.add_argument(
        "--min-onset-objects", type=int, default=1,
        help=(
            "Onset filter only: how many objects must start moving in the same step. 1 is the "
            "published onset benchmark. Higher values were intended to force contact chains, "
            "but chains are not simultaneous (the second object starts moving after the first "
            "is already in motion), so >=2 retains 0.00-0.05%% of steps in every domain and "
            "density measured. Use --filter-mode interaction instead."
        ),
    )
    parser.add_argument(
        "--filter-mode",
        choices=["motion", "onset", "interaction"],
        default="motion",
        help=(
            "'motion' keeps any step with real movement (the original filter, whose metric "
            "a one-line velocity rule wins). 'onset' keeps only steps where a stationary "
            "object starts moving, removing the momentum shortcut from the training signal."
        ),
    )
    return parser.parse_args()


def build_one(count: int, seed: int, args: argparse.Namespace) -> dict:
    input_path = Path(args.input_template.format(n=count, seed=seed))
    output_dir = Path(args.output_template.format(n=count, seed=seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    raw = np.load(input_path)
    dataset = {key: raw[key] for key in raw.files}

    # Episodes of the UNFILTERED data: each is a whole trajectory, fingerprinted by its
    # genuine initial configuration.
    episodes = extract_episodes(dataset["done"], dataset["s_t"], args.round_decimals)

    # Tag every row with the trajectory it came from and carry the tag through both the
    # split and the filter. This is what makes the disjointness check below exact: matching
    # rows by value would fail, because the unfiltered data repeats states (one resting
    # configuration recurs 71 times at 3 objects) and would tie unrelated episodes together.
    source_episode = np.empty(dataset["s_t"].shape[0], dtype=np.int64)
    for episode_index, episode in enumerate(episodes):
        source_episode[episode.start : episode.end] = episode_index
    dataset["source_episode"] = source_episode
    assignments = assign_groups(
        [episode.config_id for episode in episodes],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.split_seed,
    )

    split_episodes: dict[str, list] = {s: [] for s in SPLITS}
    for episode in episodes:
        split_episodes[assignments[episode.config_id]].append(episode)

    config_ids = {s: {e.config_id for e in split_episodes[s]} for s in SPLITS}
    for split in SPLITS:
        others = set().union(*(config_ids[o] for o in SPLITS if o != split))
        if config_ids[split] & others:
            raise RuntimeError(f"Configuration leakage in split '{split}'.")

    summary: dict[str, object] = {
        "input": str(input_path),
        "episodes_total": len(episodes),
        "split_episode_counts": {s: len(split_episodes[s]) for s in SPLITS},
        "full_transition_counts": {},
        "hard_transition_counts": {},
        "hard_window_note": "hard splits are subsets of the full splits of the same name",
        "filter_mode": args.filter_mode,
    }

    for split in SPLITS:
        indices = (
            np.concatenate([np.arange(e.start, e.end) for e in split_episodes[split]])
            if split_episodes[split]
            else np.array([], dtype=np.int64)
        )
        full_split = {key: value[indices] for key, value in dataset.items()}
        np.savez_compressed(output_dir / f"{stem}_full_{split}.npz", **full_split)
        summary["full_transition_counts"][split] = int(indices.size)  # type: ignore[index]

        # Filter *within* the split, so the hard rows can only come from this split's
        # episodes. build_filtered_dataset re-marks done at chunk ends, which is fine here
        # because nothing downstream re-splits this file.
        if args.filter_mode == "onset":
            keep_mask = compute_onset_keep_mask(
                full_split, args.min_max_xy_delta, min_onset_objects=args.min_onset_objects
            )
        elif args.filter_mode == "interaction":
            keep_mask = compute_interaction_keep_mask(full_split, args.min_max_xy_delta)
        else:
            keep_mask = compute_keep_mask(full_split, args.min_max_xy_delta, args.min_changed_objects)
        hard_split, _ = build_filtered_dataset(full_split, keep_mask, None, args.split_seed)
        np.savez_compressed(output_dir / f"{stem}_hard_{split}.npz", **hard_split)
        summary["hard_transition_counts"][split] = int(hard_split["s_t"].shape[0])  # type: ignore[index]

    verify_episode_disjoint(output_dir, stem, summary)
    (output_dir / "split_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify_episode_disjoint(output_dir: Path, stem: str, summary: dict) -> None:
    """Read back every emitted file and assert no source episode spans two splits.

    This is the check that the original pipeline lacked. Run against the old
    ``splits_{N}obj_s{S}`` directories it would have failed: 62 of 247 source episodes had
    chunks in both train and test.
    """
    episode_splits: dict[int, set[str]] = defaultdict(set)
    hard_within_full: dict[str, bool] = {}
    for split in SPLITS:
        full_episodes = set(
            np.load(output_dir / f"{stem}_full_{split}.npz")["source_episode"].tolist()
        )
        hard_episodes = set(
            np.load(output_dir / f"{stem}_hard_{split}.npz")["source_episode"].tolist()
        )
        hard_within_full[split] = hard_episodes.issubset(full_episodes)
        for episode_index in full_episodes | hard_episodes:
            episode_splits[int(episode_index)].add(split)

    spanning = sorted(index for index, splits in episode_splits.items() if len(splits) > 1)
    if spanning:
        raise RuntimeError(
            f"{len(spanning)} source episodes span multiple splits (e.g. {spanning[:5]})."
        )
    if not all(hard_within_full.values()):
        raise RuntimeError(f"Hard split is not a subset of its full split: {hard_within_full}")

    summary["source_episodes_spanning_multiple_splits"] = 0
    summary["hard_is_subset_of_full"] = hard_within_full
    summary["episodes_verified"] = len(episode_splits)


def main() -> None:
    args = parse_args()
    results = []
    for count in args.counts:
        for seed in args.seeds:
            summary = build_one(count, seed, args)
            results.append(summary)
            print(
                f"{count}obj s{seed}: episodes {summary['split_episode_counts']} "
                f"full {summary['full_transition_counts']} hard {summary['hard_transition_counts']}",
                flush=True,
            )
    print(json.dumps({"built": len(results)}, indent=2))


if __name__ == "__main__":
    main()
