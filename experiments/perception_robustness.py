"""W6 (partial): does the gate survive detector-grade observation noise?

The single objection most likely to sink this work is that the model consumes *structured*
state -- ground-truth object poses -- rather than pixels. The full answer is a slot-attention
front end operating on rendered images, which this repository has no pipeline for (both
environments emit state, not frames) and which needs sustained GPU time.

This is the cheaper partial answer, and it is worth stating exactly what it does and does not
buy. It keeps structured state but corrupts it the way a real perception stack would, then
asks whether the change gate degrades gracefully:

  * ``noise``      -- Gaussian jitter on every observed pose (position and yaw). A detector
                      with finite precision.
  * ``occlusion``  -- with some probability an object's pose is *stale*: replaced by its pose
                      from the previous step, as if the detector lost and re-acquired it.
  * ``dropout``    -- an object's pose is replaced by a plausible but wrong value, modelling
                      an identity swap between two tracked objects.
  * ``combined``   -- all three at once.

**What this does not address.** It grants the model a fixed, correctly-sized set of object
slots with stable identity, which a slot encoder must *discover*. So it tests robustness to
*measurement* error, not to representation error, and a reviewer is entitled to say so. It is
evidence that the mechanism is not brittle to imperfect input, not evidence that the pipeline
works from pixels.

Corruption is applied to the model's *input* only. The supervision targets stay clean,
because the question is whether the gate can still identify what changed given a degraded
view -- not whether it can learn from degraded labels.

Usage::

    python -m experiments.perception_robustness --counts 3 5 8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    compute_mask_metrics,
    compute_pose_metrics,
    load_dataset,
    load_sparse_model,
)
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout

# Sweep chosen around plausible detector error for 5 cm objects: 1 mm is a good pose
# estimator, 1 cm is a poor one, and 2 cm equals the displacement threshold that defines a
# "changed" object -- at which point noise and signal are the same size by construction.
NOISE_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02)
CORRUPTION_RATES = (0.0, 0.05, 0.15, 0.30)
# Velocity is not observed by a perception stack; it is finite-differenced from two noisy
# pose estimates, so its error is ~sqrt(2)/dt times the pose error. dt = 0.05 s here.
VELOCITY_AMPLIFICATION = float(np.sqrt(2.0) / 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W6-partial: perception-noise robustness.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["noise", "occlusion", "dropout", "combined", "velocity_ablation"])
    parser.add_argument("--feature-modes", type=str, nargs="+", default=["global"],
                        help=(
                            "Which trained featurisations to score. 'contact' is velocity-free "
                            "and should be far more robust, since the coupled velocity error "
                            "dominates this sweep."
                        ))
    parser.add_argument("--no-velocity-coupling", action="store_true",
                        help="Corrupt pose only (reproduces the vacuous earlier sweep).")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--corruption-rates", type=float, nargs="+", default=list(CORRUPTION_RATES))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="perception_robustness")
    parser.add_argument("--split-template", type=str,
                        default="data/transitions/splits_clean_{n}obj_s{seed}")
    parser.add_argument("--sparse-template", type=str,
                        default="models/checkpoints/sparse_clean_{n}obj_s{seed}.pt")
    parser.add_argument("--contact-template", type=str,
                        default="models/checkpoints/sparse_contact_clean_{n}obj_s{seed}.pt",
                        help="Velocity-free contact-featurised checkpoints, for the comparison.")
    return parser.parse_args()


def corrupt_state(
    state: np.ndarray,
    num_objects: int,
    mode: str,
    noise_std: float,
    rate: float,
    rng: np.random.Generator,
    corrupt_velocity: bool = True,
) -> np.ndarray:
    """Return a corrupted copy of the observed state.

    The pusher is a proprioceptive reading (the robot knows where its own end effector is)
    and the goal is commanded, so neither is corrupted -- that would model a different and
    less realistic failure.

    **Velocity must be corrupted too, and harder than pose.** An earlier version perturbed
    only the pose slice and found F1 completely flat across every corruption level, including
    2 cm noise -- the same size as the displacement that defines a "changed" object. The
    reason is that this gate reads change almost entirely from *velocity*: zeroing the
    velocity slice collapses F1 from 0.875 to 0.292, while 2 cm of pose noise moves it only
    to 0.865. Corrupting pose alone therefore measures nothing.

    A real perception stack does not observe velocity; it finite-differences successive noisy
    pose estimates, which amplifies the error by roughly ``sqrt(2)/dt``. At the 0.05 s control
    step that is a ~28x amplification, so pose noise of 5 mm implies velocity noise of about
    0.14 m/s. That coupling is what ``VELOCITY_AMPLIFICATION`` encodes, and it is the
    difference between a decorative robustness sweep and an informative one.
    """
    layout = StateLayout(num_objects=num_objects)
    corrupted = state.copy()
    poses = corrupted[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)

    if mode in {"noise", "combined"} and noise_std > 0:
        # Yaw noise is scaled to radians at roughly the same relative severity as position.
        jitter = rng.normal(0.0, noise_std, size=poses.shape)
        jitter[:, :, 2] *= np.pi / 0.05
        poses += jitter

    if mode in {"occlusion", "combined"} and rate > 0:
        # Stale pose: the detector lost the object and is reporting its last known location.
        # Emulated by carrying the previous row's pose forward, which is what a tracker with
        # a dropped frame would output.
        stale = rng.random((poses.shape[0], num_objects)) < rate
        stale[0] = False  # nothing to carry forward on the first row
        previous = np.roll(poses, 1, axis=0)
        poses[stale] = previous[stale]

    if mode in {"dropout", "combined"} and rate > 0:
        # Identity swap: two tracked objects exchange poses, the classic association failure.
        swap = rng.random(poses.shape[0]) < rate
        if num_objects >= 2 and np.any(swap):
            rows = np.flatnonzero(swap)
            first = rng.integers(0, num_objects, size=rows.size)
            second = (first + 1 + rng.integers(0, num_objects - 1, size=rows.size)) % num_objects
            temp = poses[rows, first].copy()
            poses[rows, first] = poses[rows, second]
            poses[rows, second] = temp

    corrupted[:, layout.object_pose_slice] = poses.reshape(poses.shape[0], -1)

    if mode == "velocity_ablation":
        # Diagnostic rather than a perception model: how much of the gate's decision rests on
        # velocity at all? Answering this is what revealed the pose-only sweep was vacuous.
        corrupted[:, layout.object_velocity_slice] = 0.0
        return corrupted

    if corrupt_velocity and mode in {"noise", "combined"} and noise_std > 0:
        velocities = corrupted[:, layout.object_velocity_slice]
        corrupted[:, layout.object_velocity_slice] = velocities + rng.normal(
            0.0, noise_std * VELOCITY_AMPLIFICATION, size=velocities.shape
        )
    return corrupted


def evaluate(
    checkpoint: Path, dataset: dict[str, np.ndarray], corrupted_state: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Score the model on corrupted inputs against clean targets."""
    model, config = load_sparse_model(checkpoint, device)
    state = torch.from_numpy(corrupted_state).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    feature_mode = str(config.get("feature_mode", "global"))

    deltas, gates = [], []
    with torch.no_grad():
        for start in range(0, state.shape[0], 256):
            stop = min(start + 256, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            out = model(features, estimator=estimator, temperature=float(config["temperature"]), hard=True)
            deltas.append(out.delta.cpu().numpy())
            gates.append((out.gate.probs >= 0.5).float().cpu().numpy())
    delta = np.concatenate(deltas, axis=0)
    gate = np.concatenate(gates, axis=0)

    # Poses are reconstructed from the CORRUPTED current pose, because that is what a
    # deployed system would have; scoring against the clean next pose therefore charges the
    # model for the observation error as well as its own, which is the honest accounting.
    num_objects = int(dataset["num_objects"])
    layout = StateLayout(num_objects=num_objects)
    observed_pose = corrupted_state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
    predicted = observed_pose + gate[:, :, None] * delta

    pose = compute_pose_metrics(predicted, dataset["current_pose"], dataset["next_pose"], dataset["target_mask"])
    mask_metrics = compute_mask_metrics(gate, dataset["target_mask"])
    return {
        "f1": mask_metrics["f1"], "precision": mask_metrics["precision"],
        "recall": mask_metrics["recall"],
        "overall_l2": pose["overall_per_object_l2"],
        "changed_l2": pose["changed_object_l2"],
        "unchanged_l2": pose["unchanged_object_l2"],
    }


def run_cells(args, rows, dataset, checkpoint, count, seed, feature_mode, device) -> None:
    """Score every corruption cell for one (count, seed, featurisation)."""
    for mode in args.modes:
        levels = args.noise_levels if mode in {"noise", "combined"} else [0.0]
        rates = args.corruption_rates if mode not in {"noise", "velocity_ablation"} else [0.0]
        for noise_std in levels:
            for rate in rates:
                if mode == "combined" and (noise_std == 0.0) != (rate == 0.0):
                    continue  # keep the combined sweep on its diagonal
                rng = np.random.default_rng(seed * 97 + int(noise_std * 1e5) + int(rate * 1e3))
                corrupted = corrupt_state(
                    dataset["state"], count, mode, noise_std, rate, rng,
                    corrupt_velocity=not args.no_velocity_coupling,
                )
                metrics = evaluate(checkpoint, dataset, corrupted, device)
                rows.append({
                    "object_count": count, "seed": seed, "feature_mode": feature_mode,
                    "mode": mode, "noise_std": noise_std, "corruption_rate": rate, **metrics,
                })


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "perception_robustness", "counts": args.counts,
                       "seeds": args.seeds, "modes": args.modes,
                       "noise_levels": args.noise_levels,
                       "corruption_rates": args.corruption_rates})

    rows, skipped = [], []
    for count in args.counts:
        for seed in args.seeds:
            directory = Path(args.split_template.format(n=count, seed=seed))
            data_path = directory / f"scale_{count}obj_s{seed}_hard_test.npz"
            dataset = None
            for feature_mode in args.feature_modes:
                template = args.contact_template if feature_mode == "contact" else args.sparse_template
                checkpoint = Path(template.format(n=count, seed=seed))
                if not data_path.exists() or not checkpoint.exists():
                    skipped.append(f"{count}obj s{seed} ({feature_mode})")
                    continue
                if dataset is None:
                    dataset = load_dataset(data_path)
                run_cells(args, rows, dataset, checkpoint, count, seed, feature_mode, device)
                print(f"  {count}obj s{seed} [{feature_mode}]: {len(rows)} cells so far", flush=True)


    write_outputs(rows, output_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def build_summary(rows: list[dict]) -> dict:
    def mean_f1(**filters) -> float | None:
        values = [r["f1"] for r in rows if all(r[k] == v for k, v in filters.items())]
        return float(np.mean(values)) if values else None

    clean = mean_f1(mode="noise", noise_std=0.0, corruption_rate=0.0)
    degradation: dict[str, dict[str, float | None]] = {}
    for mode in sorted({r["mode"] for r in rows}):
        entries: dict[str, float | None] = {}
        for row in rows:
            if row["mode"] != mode:
                continue
            key = f"noise{row['noise_std']}_rate{row['corruption_rate']}"
            entries.setdefault(key, mean_f1(mode=mode, noise_std=row["noise_std"],
                                            corruption_rate=row["corruption_rate"]))
        degradation[mode] = entries

    # The headline: F1 retained at a detector error of 5 mm, a fifth of the displacement
    # that defines a changed object.
    mild = mean_f1(mode="noise", noise_std=0.005, corruption_rate=0.0)
    return {
        "clean_f1": clean,
        "f1_at_5mm_noise": mild,
        "retention_at_5mm": (mild / clean) if (clean and mild) else None,
        "f1_by_mode": degradation,
    }


COLUMNS = ["object_count", "seed", "feature_mode", "mode", "noise_std", "corruption_rate",
           "f1", "precision", "recall", "overall_l2", "changed_l2", "unchanged_l2"]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "perception_robustness.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["# Perception-noise robustness (W6, partial)", "",
          "Structured state corrupted the way a detector would corrupt it, applied to the",
          "model's **input** only; targets stay clean. Mean over seeds.", "",
          "| mode | noise std | corruption rate | F1 | precision | recall | changed L2 |",
          "|---|---|---|---|---|---|---|"]
    seen = set()
    for row in sorted(rows, key=lambda r: (r["mode"], r["noise_std"], r["corruption_rate"])):
        key = (row["mode"], row["noise_std"], row["corruption_rate"])
        if key in seen:
            continue
        seen.add(key)
        matching = [r for r in rows if (r["mode"], r["noise_std"], r["corruption_rate"]) == key]
        md.append(
            f"| {row['mode']} | {row['noise_std']:.3f} | {row['corruption_rate']:.2f} | "
            f"{np.mean([r['f1'] for r in matching]):.3f} | "
            f"{np.mean([r['precision'] for r in matching]):.3f} | "
            f"{np.mean([r['recall'] for r in matching]):.3f} | "
            f"{np.mean([r['changed_l2'] for r in matching]):.4f} |"
        )
    (output_dir / "perception_robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
