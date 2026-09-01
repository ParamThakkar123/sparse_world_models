"""The change-detection study, run from PIXELS instead of privileged state.

Why this is the experiment the scope limitation demands
-------------------------------------------------------
Everything else here consumes structured per-object state: poses and velocities are handed
to the model, which never has to find the objects. For the momentum-shortcut finding that is
not an incidental limitation -- it is load-bearing. The shortcut is available *because*
velocity is served as an input feature. So the sharpest possible objection is: your finding
is an artefact of feeding models a velocity channel, and it would evaporate in a system that
had to see.

This runs the whole pipeline from 96x96 RGB renderings:

    frames -> Slot Attention (unsupervised) -> per-object slots -> change gate + delta head

and re-asks both questions there. Three conditions span the featurisation axis that the
state-space study found decisive:

  ``slot_twoframe``   slots at t and t-1 concatenated. This is the pixel analogue of the
                      velocity-using ``global`` featurisation: velocity is not given, but it
                      is *recoverable* by differencing the two frames. If the shortcut is
                      real rather than an artefact of the velocity channel, this condition
                      should learn it anyway.
  ``slot_oneframe``   slots at t only. The pixel analogue of the velocity-free ``contact``
                      featurisation: a single frame contains no motion information at all
                      (the renderer draws no blur and no trails), so the shortcut is
                      physically unavailable.
  ``trivial_pixel``   the one-line rule, computed from pixels: an object "is already moving"
                      if its slot's mask centroid shifted between t-1 and t. No learning.

**Pre-registered predictions**, recorded before the run:

  a. ``trivial_pixel`` beats both learned conditions on the motion benchmark, reproducing the
     state-space result. The shortcut is a property of the evaluation population, not of the
     input representation, so hiding velocity behind a perception step should not remove it.
  b. ``slot_twoframe`` beats ``slot_oneframe`` on the motion benchmark and LOSES to it on
     onset, mirroring ``global`` vs ``contact`` exactly.
  c. All pixel numbers sit below their state-space counterparts, because perception error
     adds on top of dynamics error.

What ground truth is used for, and what that costs
---------------------------------------------------
Slots come out unordered and their order is not stable between frames, so slot ``k`` at
``t`` is not slot ``k`` at ``t+1``. Correspondence is established by Hungarian matching
between slot mask centroids and ground-truth object pixel positions -- the protocol the Slot
Attention paper itself uses for property prediction.

Ground truth therefore orders the slots; it never produces their contents and is never a
model input. The honest consequence is that these numbers are an **upper bound**: a fully
self-contained system would also have to solve tracking, and errors there would subtract
from these results. Slot-matching quality is reported (``match_distance_px``) so a reader
can see how much of any gap is perception rather than dynamics.

Usage::

    python -m experiments.pixel_benchmark --counts 3 --seeds 0 1 2 --filter-mode motion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from experiments import ExperimentLogger
from experiments.compare_phase4_models import compute_mask_metrics
from experiments.create_hard_subset import compute_keep_mask, compute_onset_keep_mask, extract_episodes
from models import POSE_DIM, SparseResidualHead, StateLayout
from models.envs.renderer import object_pixel_positions, render_dataset
from models.keypoint_encoder import KeypointAutoencoder, positions_to_pixels
from models.slot_attention import SlotAutoencoder, slot_centroids

REST_SPEED = 2.55e-05


class Perception:
    """Adapter over the two perception front ends.

    Both are trained unsupervised by reconstruction and both expose "give me ``K`` located
    entities with a feature vector each", but they name and shape things differently. This
    wraps them so the rest of the experiment never branches on which one is in use, and so
    the pixel result can be reported as robust to the choice rather than resting on one
    module -- which matters here because Slot Attention is known to be fragile when objects
    cover a small fraction of the frame, as they do in these sparse scenes.
    """

    def __init__(self, kind: str, model: nn.Module, resolution: int):
        self.kind = kind
        self.model = model
        self.resolution = resolution

    @classmethod
    def build(cls, kind: str, resolution: int, num_entities: int, device: torch.device) -> "Perception":
        if kind == "slot":
            model = SlotAutoencoder(
                resolution=resolution, num_slots=num_entities,
                decoder_resolution=resolution // 8,
            ).to(device)
        elif kind == "keypoint":
            model = KeypointAutoencoder(
                resolution=resolution, num_keypoints=num_entities,
                decoder_resolution=resolution // 8,
            ).to(device)
        else:
            raise ValueError(f"Unknown perception front end '{kind}'.")
        return cls(kind, model, resolution)

    def loss(self, batch: torch.Tensor, foreground_weight: float = 0.0) -> torch.Tensor:
        """Reconstruction loss, optionally reweighted toward the foreground.

        Only the keypoint autoencoder takes the weight: it is the front end the weighting was
        designed for, and passing it silently to Slot Attention would make the two conditions
        differ on two things at once.
        """
        if self.kind == "keypoint" and foreground_weight > 0.0:
            return self.model.loss(batch, foreground_weight=foreground_weight)
        return self.model.loss(batch)

    @torch.no_grad()
    def entities(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(features, pixel_positions)`` for a batch of images.

        Slot Attention has no explicit position output, so its entity position is the centre
        of mass of the decoder mask; the keypoint encoder produces a position by
        construction. Both are returned in the renderer's pixel frame so the Hungarian
        matching and the pixel trivial rule are identical for the two.
        """
        if self.kind == "slot":
            output = self.model(batch)
            return output["slots"], slot_centroids(output["masks"])
        output = self.model(batch)
        return output["features"], positions_to_pixels(output["positions"], self.resolution)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Change detection from pixels.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--filter-mode", choices=["motion", "onset"], default="motion")
    parser.add_argument("--min-max-xy-delta", type=float, default=0.020)
    parser.add_argument(
        "--input-template", type=str, default="data/transitions/onset_{n}obj_s{seed}.npz",
        help="RAW contiguous episode data. Contiguity is required: the two-frame condition "
             "and the pixel trivial rule both need each step's predecessor.",
    )
    parser.add_argument(
        "--perception", choices=["slot", "keypoint"], default="keypoint",
        help=(
            "Unsupervised perception front end. 'slot' is Slot Attention (Locatello 2020); "
            "'keypoint' is the spatial-softmax autoencoder (Finn 2016), the manipulation "
            "literature's equivalent. Default is 'keypoint' because Slot Attention collapses "
            "on these scenes -- objects cover ~0.7%% of the frame, so reconstruction MSE is "
            "dominated by background and every slot mask drifts to the image centre. Run both "
            "and report both: agreement makes the pixel result robust to the choice."
        ),
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument(
        "--foreground-weight", type=float, default=0.0,
        help=(
            "Upweight each pixel by its deviation from the image's own median colour, "
            "computed per batch from the pixels themselves -- no labels, no masks, no use of "
            "the known background constant, so the front end stays unsupervised. This exists "
            "because a 5 cm object on a 60 cm table covers ~0.7%% of the frame: a decoder that "
            "reproduces the dark background and nothing else already attains most of the "
            "achievable reconstruction loss, so plain MSE supplies almost no gradient pressure "
            "to represent the objects. Keypoint front end only. 0 reproduces the published "
            "(failing) runs; 10 is the value the unit test pins as clearly foreground-dominated."
        ),
    )
    parser.add_argument("--extra-slots", type=int, default=2,
                        help="Entities beyond the object count, for the pusher and background.")
    parser.add_argument("--slot-epochs", type=int, default=30)
    parser.add_argument("--slot-batch-size", type=int, default=32)
    parser.add_argument("--slot-lr", type=float, default=4e-4)
    parser.add_argument("--gate-epochs", type=int, default=30)
    parser.add_argument("--gate-batch-size", type=int, default=128)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--sparsity-weight", type=float, default=0.2)
    parser.add_argument("--centroid-move-threshold", type=float, default=0.35,
                        help="Pixel shift above which the trivial rule calls a slot 'moving'.")
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", type=str, default="pixel_benchmark")
    return parser.parse_args()


# --------------------------------------------------------------------------- data

def build_split_indices(
    dataset: dict[str, np.ndarray], keep: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Episode-disjoint 80/10/10 split of the KEPT step indices.

    Whole episodes are assigned first and the filter applied within them, which is the
    ordering ``build_clean_splits`` established after the reverse order was found to leak 25%
    of episodes across train and test.
    """
    episodes = extract_episodes(dataset["done"])
    order = rng.permutation(len(episodes))
    train_end = int(0.8 * len(episodes))
    val_end = int(0.9 * len(episodes))
    assignment = {
        "train": order[:train_end], "val": order[train_end:val_end], "test": order[val_end:],
    }
    result: dict[str, np.ndarray] = {}
    for split, episode_ids in assignment.items():
        indices: list[int] = []
        for episode_index in episode_ids:
            episode = episodes[episode_index]
            for step in range(episode.start, episode.end):
                # The first step of an episode has no predecessor frame, and the two-frame
                # condition and the pixel trivial rule both need one. Dropping it costs ~1
                # row per episode and avoids fabricating a predecessor.
                if keep[step] and step > episode.start:
                    indices.append(step)
        result[split] = np.asarray(sorted(indices), dtype=np.int64)
    return result


def render_pairs(
    dataset: dict[str, np.ndarray], indices: np.ndarray, count: int, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(frames_t, frames_prev)`` for the given step indices."""
    frames = render_dataset(dataset["state"][indices], count, resolution)
    previous = render_dataset(dataset["state"][indices - 1], count, resolution)
    return frames, previous


# --------------------------------------------------------------------- perception

def train_perception(
    frames: np.ndarray, args: argparse.Namespace, num_entities: int, device: torch.device
) -> Perception:
    """Unsupervised reconstruction training. No object labels enter here."""
    perception = Perception.build(args.perception, args.resolution, num_entities, device)
    model = perception.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.slot_lr)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)

    steps_per_epoch = max(1, (tensor.shape[0] + args.slot_batch_size - 1) // args.slot_batch_size)
    total_steps = steps_per_epoch * args.slot_epochs
    # Linear warmup then exponential decay, as in the reference implementation. This is not
    # optional tuning: without warmup the attention logits saturate in the first few hundred
    # steps and every slot collapses onto the background, which shows up downstream as a
    # slot-to-object match distance around a third of the image and makes the whole pixel
    # experiment measure nothing.
    warmup_steps = max(1, int(0.05 * total_steps))
    decay_steps = max(1, total_steps)

    def learning_rate(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup_steps)
        return warm * (0.5 ** (step / decay_steps))

    step = 0
    for epoch in range(args.slot_epochs):
        model.train()
        permutation = torch.randperm(tensor.shape[0])
        total, batches = 0.0, 0
        for start in range(0, tensor.shape[0], args.slot_batch_size):
            for group in optimizer.param_groups:
                group["lr"] = args.slot_lr * learning_rate(step)
            batch = tensor[permutation[start : start + args.slot_batch_size]].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = perception.loss(batch, foreground_weight=args.foreground_weight)
            loss.backward()
            # Slot attention diverges early without gradient clipping.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
            batches += 1
            step += 1
        if epoch % 10 == 0 or epoch == args.slot_epochs - 1:
            print(f"    {args.perception} epoch {epoch:3d} "
                  f"recon_mse={total / max(batches, 1):.5f}", flush=True)
    return perception


@torch.no_grad()
def extract_matched_entities(
    perception: Perception,
    frames: np.ndarray,
    states: np.ndarray,
    count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode frames and reorder entities into ground-truth object order.

    Returns ``(features, positions_px, mean_match_distance_px)``. Features and positions come
    from the same encode pass, which matters for more than speed: computing them separately
    would run the encoder twice, and Slot Attention samples its initial slots from a learned
    Gaussian, so the two passes would not even return the same decomposition.

    Ground truth is used ONLY to order the entities. Their contents never see it.
    """
    perception.model.eval()
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    matched_features: list[np.ndarray] = []
    matched_positions: list[np.ndarray] = []
    distances: list[float] = []
    for start in range(0, tensor.shape[0], args.slot_batch_size):
        stop = min(start + args.slot_batch_size, tensor.shape[0])
        features_batch, positions_batch = perception.entities(tensor[start:stop].to(device))
        features = features_batch.cpu().numpy()   # (b, K, D)
        positions = positions_batch.cpu().numpy()  # (b, K, 2) in pixels
        for row in range(stop - start):
            truth = object_pixel_positions(states[start + row], count, args.resolution)
            cost = np.linalg.norm(
                truth[:, None, :] - positions[row][None, :, :], axis=-1
            )  # (count, K)
            object_index, entity_index = linear_sum_assignment(cost)
            ordered_features = np.zeros((count, features.shape[-1]), dtype=np.float32)
            ordered_positions = np.zeros((count, 2), dtype=np.float32)
            ordered_features[object_index] = features[row][entity_index]
            ordered_positions[object_index] = positions[row][entity_index]
            matched_features.append(ordered_features)
            matched_positions.append(ordered_positions)
            distances.append(float(cost[object_index, entity_index].mean()))
    return np.stack(matched_features), np.stack(matched_positions), float(np.mean(distances))


# ------------------------------------------------------------------ change models

def build_features(
    slots_now: np.ndarray, slots_prev: np.ndarray, action: np.ndarray, condition: str
) -> np.ndarray:
    """Per-object features for one condition, with the action broadcast to every object."""
    count = slots_now.shape[1]
    broadcast_action = np.repeat(action[:, None, :], count, axis=1)
    if condition == "slot_oneframe":
        return np.concatenate([slots_now, broadcast_action], axis=-1).astype(np.float32)
    if condition == "slot_twoframe":
        return np.concatenate([slots_now, slots_prev, broadcast_action], axis=-1).astype(np.float32)
    raise ValueError(f"Unknown condition '{condition}'.")


def train_gate(
    features: np.ndarray,
    target_mask: np.ndarray,
    target_delta: np.ndarray,
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> SparseResidualHead:
    """Train the project's own gate + residual delta head on slot features."""
    model = SparseResidualHead(object_feature_dim=features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gate_lr)
    x = torch.from_numpy(features).to(device)
    mask = torch.from_numpy(target_mask).float().to(device)
    delta = torch.from_numpy(target_delta).float().to(device)
    val_x = torch.from_numpy(val[0]).to(device)
    val_mask = torch.from_numpy(val[1]).float().to(device)

    positive = float(mask.mean().clamp(1e-6, 1 - 1e-6))
    # Same auto-balanced BCE the state-space runs use, so the two are comparable.
    pos_weight = torch.tensor((1.0 - positive) / positive, device=device)

    best_state, best_f1 = None, -1.0
    for _ in range(args.gate_epochs):
        model.train()
        permutation = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], args.gate_batch_size):
            index = permutation[start : start + args.gate_batch_size]
            output = model(x[index], estimator="gumbel_st", temperature=1.0, hard=True)
            gate_loss = nn.functional.binary_cross_entropy_with_logits(
                output.gate.logits, mask[index], pos_weight=pos_weight
            )
            selected = mask[index].unsqueeze(-1)
            denominator = selected.sum().clamp_min(1.0) * POSE_DIM
            delta_loss = (((output.delta - delta[index]) ** 2) * selected).sum() / denominator
            loss = gate_loss + delta_loss + args.sparsity_weight * output.gate.probs.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = model(val_x, estimator="sigmoid", temperature=1.0, hard=True).gate.probs
        f1 = compute_mask_metrics(
            (probs >= 0.5).float().cpu().numpy(), val_mask.cpu().numpy()
        )["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_mask(model: SparseResidualHead, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(features).to(device)
    probs = model(x, estimator="sigmoid", temperature=1.0, hard=True).gate.probs
    return (probs >= 0.5).float().cpu().numpy()


def trivial_pixel_mask(
    slots_now: np.ndarray, slots_prev: np.ndarray, centroids: tuple[np.ndarray, np.ndarray],
    threshold: float,
) -> np.ndarray:
    """The one-line rule computed from pixels: predict change iff the slot centroid moved.

    Takes matched centroids rather than slot vectors: a slot embedding shift is not
    interpretable as motion, but a mask-centroid shift in pixels is exactly the quantity a
    velocity rule needs, and it is measurable without any labels.
    """
    now, previous = centroids
    return (np.linalg.norm(now - previous, axis=-1) > threshold).astype(np.float32)


# ------------------------------------------------------------------------- driver

def at_rest_from_state(states: np.ndarray, count: int) -> np.ndarray:
    """Ground-truth at-rest mask. Used only to DEFINE the onset metric, never as input."""
    layout = StateLayout(num_objects=count)
    velocity = states[:, layout.object_velocity_slice].reshape(-1, count, 6)
    return np.linalg.norm(velocity[:, :, 3:5], axis=2) <= REST_SPEED


def score(prediction: np.ndarray, target: np.ndarray, at_rest: np.ndarray) -> dict[str, float]:
    return {
        "f1": compute_mask_metrics(prediction, target)["f1"],
        "onset_f1": (
            compute_mask_metrics(prediction[at_rest], target[at_rest])["f1"]
            if at_rest.any() else float("nan")
        ),
        "precision": compute_mask_metrics(prediction, target)["precision"],
        "recall": compute_mask_metrics(prediction, target)["recall"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config(vars(args))
    rows: list[dict] = []

    for count in args.counts:
        for seed in args.seeds:
            path = Path(args.input_template.format(n=count, seed=seed))
            if not path.exists():
                print(f"  skip missing {path}")
                continue
            print(f"== N={count} seed={seed} ({args.filter_mode}) ==", flush=True)
            raw = dict(np.load(path))
            raw.setdefault("state", raw["s_t"])
            keep = (
                compute_keep_mask(raw, args.min_max_xy_delta, 1)
                if args.filter_mode == "motion"
                else compute_onset_keep_mask(raw, args.min_max_xy_delta, 1)
            )
            indices = build_split_indices(raw, keep, np.random.default_rng(seed))
            if min(len(v) for v in indices.values()) < 10:
                print(f"  skip: split too small {[len(v) for v in indices.values()]}")
                continue
            if len(indices["train"]) > args.max_train_rows:
                indices["train"] = indices["train"][: args.max_train_rows]
            print(f"  rows train/val/test = "
                  f"{len(indices['train'])}/{len(indices['val'])}/{len(indices['test'])}", flush=True)

            rendered = {
                split: render_pairs(raw, index, count, args.resolution)
                for split, index in indices.items()
            }

            torch.manual_seed(seed)
            perception = train_perception(
                np.concatenate([rendered["train"][0], rendered["train"][1]], axis=0),
                args, count + args.extra_slots, device,
            )

            data: dict[str, dict] = {}
            for split, index in indices.items():
                frames, previous = rendered[split]
                states = raw["state"][index]
                previous_states = raw["state"][index - 1]
                slots_now, centroids_now, distance = extract_matched_entities(
                    perception, frames, states, count, args, device
                )
                slots_prev, centroids_prev, _ = extract_matched_entities(
                    perception, previous, previous_states, count, args, device
                )
                data[split] = {
                    "slots_now": slots_now, "slots_prev": slots_prev,
                    "action": raw["a_t"][index],
                    "target_mask": raw["changed_mask"][index].astype(np.float32),
                    "target_delta": raw["delta_vector"][index].astype(np.float32),
                    "at_rest": at_rest_from_state(states, count),
                    "match_distance_px": distance,
                    "centroids_now": centroids_now,
                    "centroids_prev": centroids_prev,
                }
            print(f"  {args.perception} match distance (test) = "
                  f"{data['test']['match_distance_px']:.2f} px", flush=True)

            test = data["test"]
            trivial = trivial_pixel_mask(
                test["slots_now"], test["slots_prev"],
                (test["centroids_now"], test["centroids_prev"]), args.centroid_move_threshold,
            )
            rows.append({
                "object_count": count, "seed": seed, "condition": "trivial_pixel",
                "match_distance_px": test["match_distance_px"],
                **score(trivial, test["target_mask"], test["at_rest"]),
            })

            for condition in ("slot_twoframe", "slot_oneframe"):
                features = {
                    split: build_features(
                        values["slots_now"], values["slots_prev"], values["action"], condition
                    )
                    for split, values in data.items()
                }
                gate = train_gate(
                    features["train"], data["train"]["target_mask"], data["train"]["target_delta"],
                    (features["val"], data["val"]["target_mask"], data["val"]["target_delta"]),
                    args, device,
                )
                prediction = predict_mask(gate, features["test"], device)
                rows.append({
                    "object_count": count, "seed": seed, "condition": condition,
                    "match_distance_px": test["match_distance_px"],
                    **score(prediction, test["target_mask"], test["at_rest"]),
                })

            for row in rows[-3:]:
                print(f"  {row['condition']:16s} F1={row['f1']:.4f} onset={row['onset_f1']:.4f} "
                      f"recall={row['recall']:.4f}", flush=True)

    write_outputs(rows, args, logger.run_dir)
    summary = build_summary(rows, args)
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


COLUMNS = ["object_count", "seed", "condition", "f1", "onset_f1", "precision", "recall",
           "match_distance_px"]


def build_summary(rows: list[dict], args: argparse.Namespace) -> dict:
    def mean(condition: str, metric: str) -> float | None:
        values = [r[metric] for r in rows if r["condition"] == condition and r[metric] == r[metric]]
        return float(np.mean(values)) if values else None

    trivial_f1 = mean("trivial_pixel", "f1")
    two_f1, one_f1 = mean("slot_twoframe", "f1"), mean("slot_oneframe", "f1")
    two_onset, one_onset = mean("slot_twoframe", "onset_f1"), mean("slot_oneframe", "onset_f1")
    return {
        "filter_mode": args.filter_mode,
        "per_condition": {
            condition: {metric: mean(condition, metric)
                        for metric in ("f1", "onset_f1", "recall", "match_distance_px")}
            for condition in sorted({r["condition"] for r in rows})
        },
        # The pre-registered checks from the module docstring.
        "trivial_beats_learned_on_this_benchmark": (
            bool(trivial_f1 > max(two_f1, one_f1))
            if None not in (trivial_f1, two_f1, one_f1) else None
        ),
        "twoframe_beats_oneframe_on_f1": (
            bool(two_f1 > one_f1) if None not in (two_f1, one_f1) else None
        ),
        "oneframe_beats_twoframe_on_onset": (
            bool(one_onset > two_onset) if None not in (one_onset, two_onset) else None
        ),
    }


def write_outputs(rows: list[dict], args: argparse.Namespace, output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "pixel_benchmark.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        f"# Change detection from pixels ({args.filter_mode} benchmark)",
        "",
        "Slot Attention trained unsupervised on 96x96 renderings; the change gate then runs on",
        "slots rather than on privileged state. `trivial_pixel` is the one-line velocity rule",
        "computed from slot mask-centroid displacement, with no learning and no labels.",
        "`match_distance_px` is how far a matched slot's centroid sits from its object, i.e.",
        "how much of any gap is perception rather than dynamics.",
        "",
        "| condition | F1 | onset F1 | recall | match dist (px) |",
        "|---|---|---|---|---|",
    ]
    for condition in ("trivial_pixel", "slot_twoframe", "slot_oneframe"):
        matching = [r for r in rows if r["condition"] == condition]
        if not matching:
            continue
        md.append(
            f"| {condition} | {np.mean([r['f1'] for r in matching]):.4f} | "
            f"{np.nanmean([r['onset_f1'] for r in matching]):.4f} | "
            f"{np.mean([r['recall'] for r in matching]):.4f} | "
            f"{np.mean([r['match_distance_px'] for r in matching]):.2f} |"
        )
    (output_dir / "pixel_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
