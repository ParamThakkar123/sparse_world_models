"""Build the INTERACTION benchmark: change caused by object-object contact, not by the pusher.

Why this benchmark
------------------
Two benchmarks in this project have now been shown to be solvable without learning:

* the motion-filtered benchmark, by "predict change iff the object is already moving"
  (F1 0.866, beating every learned model and all five published ones);
* the onset-filtered benchmark, by "predict change for the object nearest the pusher"
  (F1 0.925, **zero parameters**, against a learned gate's 0.694).

Both are symptoms of one thing: with a single point-like pusher and separated objects, the
label is recoverable from a single scalar -- speed, or distance. The task never poses a
genuine attribution problem.

This benchmark keeps only transitions where a stationary object starts moving **while some
other object is closer to the pusher**. Such an object was not moved by the pusher; the
impulse reached it through another object. Distance to the pusher cannot identify it even in
principle, because the quantity that decides the label is the transferred impulse.

Why it needs its own builder
----------------------------
Qualifying events are 0.04-0.18% of steps (measured: billiards 0.180%, clutter 0.072%,
planar 0.042%), so a usable split needs roughly 10,000-40,000 episodes per cell against the
onset benchmark's 1,500. Generating that in one file would produce a multi-hundred-megabyte
array that the split machinery loads whole -- and running several such jobs concurrently is
exactly what killed the cross-domain split stage with a bare ``MemoryError``.

So generation is **chunked and streamed**: generate a chunk, assign its episodes to splits,
filter within each split, keep only the surviving rows, delete the chunk, repeat. Peak memory
is one chunk rather than the whole corpus.

Episode-disjointness across chunks
----------------------------------
The usual splitter shuffles the episode list and cuts it by fraction, which is not
chunk-independent: the same episode would land in different splits depending on which chunk
it arrived in. Here the split is a deterministic function of the episode's own configuration
fingerprint -- the SHA1 of its rounded initial object poses, mapped to [0, 1) and cut by the
same fractions. Two identical initial configurations therefore always land in the same split
no matter which chunk they appear in, which is the property the leak audit
(``build_clean_splits``) established as necessary and which a naive streaming split would
silently break.

Usage::

    python -m experiments.build_interaction_benchmark --env billiards --counts 5 8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments import ExperimentLogger
from experiments.create_hard_subset import compute_interaction_keep_mask
from experiments.split_dataset import extract_episodes

SPLITS = ("train", "val", "test")

# Per-domain motion thresholds from experiments/domain_characterization.py.
THRESHOLDS = {"tabletop": 0.020, "planar": 0.010, "billiards": 0.031, "clutter": 0.029}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the interaction benchmark.")
    parser.add_argument("--env", default="billiards",
                        choices=["tabletop", "planar", "billiards", "clutter"])
    parser.add_argument("--counts", type=int, nargs="+", default=[5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes-per-chunk", type=int, default=2500)
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--object-bound", type=float, default=0.26)
    parser.add_argument("--min-object-separation", type=float, default=0.07)
    parser.add_argument("--min-max-xy-delta", type=float, default=None,
                        help="Defaults to the domain's derived threshold.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--round-decimals", type=int, default=4)
    parser.add_argument("--output-template", type=str,
                        default="data/transitions/splits_interaction_{env}_{n}obj_s{seed}")
    parser.add_argument("--scratch-dir", type=Path, default=Path("data/transitions/_interaction_tmp"))
    parser.add_argument("--run-name", type=str, default="build_interaction_benchmark")
    return parser.parse_args()


def split_for(config_id: str, train_frac: float, val_frac: float) -> str:
    """Deterministic split assignment from the configuration fingerprint alone.

    Chunk-independent by construction -- see the module docstring for why that matters. The
    first 8 hex digits give ~4 billion buckets, far more than the episode count, so the
    induced split proportions are within a fraction of a percent of the targets.
    """
    position = int(config_id[:8], 16) / float(0x100000000)
    if position < train_frac:
        return "train"
    if position < train_frac + val_frac:
        return "val"
    return "test"


def generate_chunk(path: Path, count: int, generation_seed: int, args: argparse.Namespace) -> bool:
    """Generate one chunk in a SUBPROCESS so its memory is returned to the OS on exit."""
    if path.exists():
        return True
    result = subprocess.run(
        [
            sys.executable, "-m", "experiments.generate_transitions",
            "--policy", "scripted",
            "--episodes", str(args.episodes_per_chunk),
            "--max-steps", str(args.max_steps),
            "--num-objects", str(count),
            "--seed", str(generation_seed),
            "--object-bound", str(args.object_bound),
            "--min-object-separation", str(args.min_object_separation),
            "--env", args.env,
            "--output", str(path),
            "--run-name", f"_interaction_chunk_{args.env}_{count}_{generation_seed}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"    chunk generation failed: {result.stderr.decode(errors='replace')[-400:]}")
        return False
    return True


def build_cell(count: int, seed: int, args: argparse.Namespace, threshold: float) -> dict:
    accumulated: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in SPLITS}
    config_ids: dict[str, set[str]] = {s: set() for s in SPLITS}
    total_steps = 0

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for chunk in range(args.num_chunks):
        # Distinct generation seed per (seed, chunk) so chunks are different episodes rather
        # than the same ones regenerated.
        generation_seed = seed * 1000 + chunk
        path = args.scratch_dir / f"{args.env}_{count}obj_s{seed}_c{chunk}.npz"
        if not generate_chunk(path, count, generation_seed, args):
            continue

        raw = np.load(path)
        dataset = {key: raw[key] for key in raw.files}
        raw.close()
        total_steps += dataset["s_t"].shape[0]

        episodes = extract_episodes(dataset["done"], dataset["s_t"], args.round_decimals)
        for episode in episodes:
            split = split_for(episode.config_id, args.train_frac, args.val_frac)
            config_ids[split].add(episode.config_id)
            indices = np.arange(episode.start, episode.end)
            rows = {key: value[indices] for key, value in dataset.items()}
            keep = compute_interaction_keep_mask(rows, threshold)
            if not keep.any():
                continue
            accumulated[split].append({key: value[keep] for key, value in rows.items()})

        del dataset
        path.unlink(missing_ok=True)
        kept = {s: sum(part["s_t"].shape[0] for part in accumulated[s]) for s in SPLITS}
        print(f"    chunk {chunk + 1}/{args.num_chunks}: steps={total_steps} kept={kept}", flush=True)

    output_dir = Path(args.output_template.format(env=args.env, n=count, seed=seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"interaction_{args.env}_{count}obj_s{seed}"

    counts: dict[str, int] = {}
    for split in SPLITS:
        parts = accumulated[split]
        if parts:
            merged = {
                key: np.concatenate([part[key] for part in parts], axis=0)
                for key in parts[0]
            }
            # The filtered rows are no longer contiguous episodes, so mark every row done.
            # Nothing downstream re-splits this file -- the split already happened above --
            # and leaving stale done flags is precisely the bug that caused the original leak.
            merged["done"] = np.ones(merged["s_t"].shape[0], dtype=merged["done"].dtype)
        else:
            merged = {}
        counts[split] = int(merged["s_t"].shape[0]) if merged else 0
        if merged:
            np.savez_compressed(output_dir / f"{stem}_hard_{split}.npz", **merged)

    overlap = {
        f"{a}&{b}": len(config_ids[a] & config_ids[b])
        for a in SPLITS for b in SPLITS if a < b
    }
    summary = {
        "env": args.env, "object_count": count, "seed": seed,
        "threshold": threshold,
        "total_generated_steps": total_steps,
        "kept": counts,
        "keep_rate": (sum(counts.values()) / total_steps) if total_steps else 0.0,
        "config_id_overlap_between_splits": overlap,
        "episode_disjoint": all(value == 0 for value in overlap.values()),
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    threshold = args.min_max_xy_delta or THRESHOLDS[args.env]
    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config({**{k: str(v) for k, v in vars(args).items()}, "threshold": threshold})

    summaries = []
    for count in args.counts:
        for seed in args.seeds:
            print(f"== {args.env} N={count} seed={seed} (threshold {threshold}) ==", flush=True)
            summary = build_cell(count, seed, args, threshold)
            summaries.append(summary)
            print(f"   -> kept {summary['kept']} from {summary['total_generated_steps']} steps "
                  f"({summary['keep_rate']:.5f}), disjoint={summary['episode_disjoint']}",
                  flush=True)

    logger.log_summary({
        "cells": summaries,
        "all_episode_disjoint": all(s["episode_disjoint"] for s in summaries),
        "min_train_rows": min((s["kept"]["train"] for s in summaries), default=0),
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
