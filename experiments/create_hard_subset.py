from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models import StateLayout


@dataclass
class Episode:
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a harder transition subset by filtering for meaningful object motion.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-max-xy-delta", type=float, default=0.02)
    parser.add_argument("--min-changed-objects", type=int, default=1)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def extract_episodes(done: np.ndarray) -> list[Episode]:
    done_indices = np.flatnonzero(done)
    if done_indices.size == 0 or done_indices[-1] != len(done) - 1:
        raise ValueError("Dataset must contain complete episodes ending with done=True.")

    episodes: list[Episode] = []
    start = 0
    for end in done_indices:
        episodes.append(Episode(start=start, end=int(end) + 1))
        start = int(end) + 1
    return episodes


# An object is "at rest" below this planar speed. Shared with
# experiments/momentum_shortcut.py, which is where the number is justified.
REST_SPEED = 2.55e-05


def compute_keep_mask(dataset: dict[str, np.ndarray], min_max_xy_delta: float, min_changed_objects: int) -> np.ndarray:
    delta = dataset["object_delta"]
    changed_mask = dataset["object_change_mask"]
    xy_norm = np.linalg.norm(delta[:, :, :2], axis=2)
    max_xy_delta = xy_norm.max(axis=1)
    changed_count = changed_mask.sum(axis=1)
    return (max_xy_delta >= min_max_xy_delta) & (changed_count >= min_changed_objects)


def compute_onset_keep_mask(
    dataset: dict[str, np.ndarray],
    min_max_xy_delta: float,
    rest_speed: float = REST_SPEED,
    min_onset_objects: int = 1,
) -> np.ndarray:
    """Keep transitions in which at least one **stationary** object starts moving.

    The motion filter above ("any object moved by at least ``min_max_xy_delta``") produces a
    subset where change detection is solved by reading velocity: an object already in motion
    almost certainly keeps moving, so P(changed | moving) is 0.94-0.99 and a one-line
    "already moving" rule beats every learned model on the resulting metric, in both
    environments (see experiments/momentum_shortcut.py).

    Selecting *onset* events instead removes the shortcut from the training signal. A kept
    transition must contain an object that was at rest at time ``t`` and moved by ``t+1``,
    which can only happen through contact. Velocity is still observable, and objects already
    in motion still appear in the scene -- what changes is that the positives the model must
    catch are no longer predictable from momentum.

    Onset events are rare (5-6% of at-rest object slots), so this filter keeps far fewer
    transitions than the motion filter. That is the point: the retained data is the part of
    the original dataset that actually required prediction.

    ``min_onset_objects``: the CHAIN requirement, and the reason this parameter exists
    -----------------------------------------------------------------------------------
    Removing the velocity shortcut is not sufficient. ``experiments/onset_shortcut_audit.py``
    found that with ``min_onset_objects=1`` the resulting benchmark is won by a *proximity*
    rule -- "predict change for the object nearest the pusher", zero parameters, F1 0.925 and
    onset F1 0.980 against a learned gate's 0.694. The mechanism is structural: with one
    point-like pusher and well-separated objects, exactly one object is contacted per step and
    it is always the nearest, so the label is recoverable from a single scalar.

    Requiring **two or more simultaneous onsets** breaks that by construction. Two stationary
    objects can only start moving in the same step through a contact *chain*: the pusher moves
    A, and A moves B. B is then not the nearest object to the pusher and may be outside any
    fixed contact radius, so no rule over pusher-to-object distance can identify it. Predicting
    which objects a chain reaches requires modelling the transferred impulse, which is the
    thing this whole project claims to be about.

    Set to 1 for the original onset filter (backwards compatible; the published onset numbers
    used that). Set to 2 for the chain benchmark.
    """
    delta = dataset["object_delta"]
    changed = dataset["object_change_mask"] > 0.5
    num_objects = changed.shape[1]

    layout = StateLayout(num_objects=num_objects)
    velocity = dataset["s_t"][:, layout.object_velocity_slice].reshape(-1, num_objects, 6)
    at_rest = np.linalg.norm(velocity[:, :, 3:5], axis=2) <= rest_speed

    onset = at_rest & changed
    # Require the onset displacement itself to clear the motion threshold, so a barely
    # detectable jitter does not qualify as an onset event.
    xy_norm = np.linalg.norm(delta[:, :, :2], axis=2)
    qualifying = onset & (xy_norm >= min_max_xy_delta)
    return qualifying.sum(axis=1) >= max(1, int(min_onset_objects))


# Must match PUSHER_ACTION_SCALE / PUSHER_BOUND in experiments.train_sparse_model, so the
# post-action pusher position used here is the same one the contact featurisation exposes to
# models and the same one the shortcut audit's proximity rules use.
PUSHER_ACTION_SCALE = 0.04
PUSHER_BOUND = 0.26


def compute_interaction_keep_mask(
    dataset: dict[str, np.ndarray],
    min_max_xy_delta: float,
    rest_speed: float = REST_SPEED,
) -> np.ndarray:
    """Keep onset events caused by OBJECT-OBJECT interaction, not by direct pusher contact.

    Why this filter exists
    ----------------------
    The onset filter removes the velocity shortcut but leaves a stronger one. Measured by
    ``experiments/onset_shortcut_audit.py``: on the onset benchmark a zero-parameter rule --
    "predict change for the object nearest the pusher" -- reaches F1 0.925 and onset F1 0.980,
    against a learned gate's 0.694. The cause is structural, not a bug in the filter. With one
    point-like pusher and objects placed 0.09 m apart, exactly one object is contacted per
    step and it is always the nearest, so the label is recoverable from a single scalar.

    A step passes this filter when a **stationary object starts moving while some OTHER object
    is closer to the pusher**. Such an object cannot have been moved by the pusher directly;
    the impulse reached it through another object. That is precisely the phenomenon
    object-centric relational world models claim to capture, and no rule over
    pusher-to-object distance can identify it, because the quantity that determines the label
    is the transferred impulse rather than the distance.

    Rate, measured before building anything (250 episodes, 8 objects): 37-44% of onset events
    qualify across the clutter and billiards domains, and 24-35% involve an object more than
    0.09 m from the pusher. Since onset events are themselves 0.2-0.44% of steps, the
    qualifying population is roughly 0.1-0.16% of steps, so this filter needs about an order
    of magnitude more episodes than the onset benchmark.

    **The methodological hazard, stated plainly.** This filter is defined using a quantity --
    distance to the pusher -- that a known-winning trivial rule also uses, so it could be
    read as gerrymandering the benchmark against that specific baseline. Two things make it
    defensible, and neither is optional: the selection criterion is *physical* (change caused
    indirectly) rather than "wherever rule X fails", and the resulting benchmark must be run
    against the **whole** battery of trivial rules, including new ones invented against it.
    A benchmark that defeats one one-liner and is won by the next is no better than the one it
    replaced.
    """
    delta = dataset["object_delta"]
    changed = dataset["object_change_mask"] > 0.5
    num_objects = changed.shape[1]
    layout = StateLayout(num_objects=num_objects)

    state = dataset["s_t"]
    velocity = state[:, layout.object_velocity_slice].reshape(-1, num_objects, 6)
    at_rest = np.linalg.norm(velocity[:, :, 3:5], axis=2) <= rest_speed
    xy_norm = np.linalg.norm(delta[:, :, :2], axis=2)
    onset = at_rest & changed & (xy_norm >= min_max_xy_delta)

    pusher = state[:, 0:2]
    action = np.clip(dataset["a_t"], -1.0, 1.0)
    pusher_next = np.clip(pusher + action * PUSHER_ACTION_SCALE, -PUSHER_BOUND, PUSHER_BOUND)
    object_xy = state[:, layout.object_pose_slice].reshape(-1, num_objects, 3)[:, :, :2]
    distance = np.linalg.norm(object_xy - pusher_next[:, None, :], axis=2)
    nearest = distance.argmin(axis=1)

    # An onset object that is not the nearest one to the pusher was reached indirectly.
    not_nearest = np.ones_like(onset)
    not_nearest[np.arange(onset.shape[0]), nearest] = False
    return (onset & not_nearest).any(axis=1)


def filter_episode_indices(episode: Episode, keep_mask: np.ndarray) -> list[np.ndarray]:
    episode_keep = keep_mask[episode.start:episode.end]
    kept_local = np.flatnonzero(episode_keep)
    if kept_local.size == 0:
        return []

    chunks: list[np.ndarray] = []
    chunk_start = kept_local[0]
    previous = kept_local[0]
    for idx in kept_local[1:]:
        if idx != previous + 1:
            chunks.append(np.arange(episode.start + chunk_start, episode.start + previous + 1))
            chunk_start = idx
        previous = idx
    chunks.append(np.arange(episode.start + chunk_start, episode.start + previous + 1))
    return chunks


def build_filtered_dataset(dataset: dict[str, np.ndarray], keep_mask: np.ndarray, max_transitions: int | None, seed: int) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    episodes = extract_episodes(dataset["done"])
    chunks: list[np.ndarray] = []
    kept_episode_count = 0
    for episode in episodes:
        episode_chunks = filter_episode_indices(episode, keep_mask)
        if episode_chunks:
            kept_episode_count += 1
            chunks.extend(episode_chunks)

    if not chunks:
        raise ValueError("Filtering removed all transitions.")

    if max_transitions is not None:
        rng = np.random.default_rng(seed)
        chunk_order = np.arange(len(chunks))
        rng.shuffle(chunk_order)
        selected_chunks: list[np.ndarray] = []
        total = 0
        for chunk_idx in chunk_order:
            chunk = chunks[int(chunk_idx)]
            if total >= max_transitions:
                break
            if total + len(chunk) <= max_transitions:
                selected_chunks.append(chunk)
                total += len(chunk)
        if not selected_chunks:
            raise ValueError("max_transitions is too small to keep any chunk.")
        chunks = sorted(selected_chunks, key=lambda chunk: int(chunk[0]))

    indices = np.concatenate(chunks)
    filtered = {key: value[indices].copy() for key, value in dataset.items()}
    filtered["done"] = np.zeros_like(filtered["done"], dtype=bool)

    offset = 0
    chunk_lengths: list[int] = []
    for chunk in chunks:
        chunk_length = len(chunk)
        offset += chunk_length
        filtered["done"][offset - 1] = True
        chunk_lengths.append(chunk_length)

    changed_mask = filtered["object_change_mask"]
    delta = filtered["object_delta"]
    xy_norm = np.linalg.norm(delta[:, :, :2], axis=2)
    max_xy_delta = xy_norm.max(axis=1)

    metadata = {
        "episodes_total": len(episodes),
        "episodes_with_kept_transitions": kept_episode_count,
        "chunks_total": len(chunks),
        "transitions_total": int(dataset["done"].shape[0]),
        "transitions_kept": int(indices.size),
        "changed_object_fraction": float(changed_mask.mean()),
        "transition_any_changed_fraction": float((changed_mask.sum(axis=1) > 0).mean()),
        "transition_multi_changed_fraction": float((changed_mask.sum(axis=1) >= 2).mean()),
        "max_xy_delta_median": float(np.median(max_xy_delta)),
        "max_xy_delta_p90": float(np.quantile(max_xy_delta, 0.9)),
        "chunk_length_median": float(np.median(np.asarray(chunk_lengths, dtype=np.float32))),
    }
    return filtered, metadata


def main() -> None:
    args = parse_args()
    raw = np.load(args.input)
    dataset = {key: raw[key] for key in raw.files}

    keep_mask = compute_keep_mask(
        dataset,
        min_max_xy_delta=args.min_max_xy_delta,
        min_changed_objects=args.min_changed_objects,
    )
    filtered, metadata = build_filtered_dataset(
        dataset,
        keep_mask,
        max_transitions=args.max_transitions,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **filtered)

    full_metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "min_max_xy_delta": args.min_max_xy_delta,
        "min_changed_objects": args.min_changed_objects,
        "max_transitions": args.max_transitions,
        "seed": args.seed,
        **metadata,
    }
    metadata_path = args.output.with_suffix("")
    metadata_path = metadata_path.parent / f"{metadata_path.name}_metadata.json"
    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
    print(json.dumps(full_metadata, indent=2))


if __name__ == "__main__":
    main()
