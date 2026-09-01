"""The momentum shortcut: what is change detection actually measuring?

This is the experiment that should have existed from the start, and its absence let a
straw-man comparison stand for months.

Every change-detection number in this project compares the gated model (F1 ~0.87 at 3
objects) against ungated regressors (~0.53). But the ungated models are degenerate by
construction -- they flag everything -- so that gap measures very little. The baseline that
was missing is the *trivial* one:

    predict "this object will change" iff it is ALREADY MOVING.

On the hard subset that rule scores **F1 0.912 at 3 objects, beating the learned gate's
0.875**, and it wins at every object count. The reason is momentum: P(changed | already
moving) is 0.94-0.99, because the hard subset is filtered to steps with real motion and an
object in motion stays in motion. So the headline metric is dominated by *continuation*, not
by prediction, and the learned gate does not beat one line of code on it.

The genuinely predictive part of the task is **onset**: objects at rest that begin moving
because something contacts them. Velocity is useless there by definition, so onset F1
isolates whether a model has learned contact at all. Splitting the metric that way inverts
the ranking:

  * the velocity-using ``global`` featurisation collapses to F1 0.04-0.10 on onset -- it has
    largely learned to read momentum;
  * the velocity-free ``contact`` featurisation reaches ~0.33 at every count, 3-5x better,
    despite scoring *worse* on the momentum-dominated headline metric.

That single fact explains three failures previously treated as unrelated: planning collapsed
under CEM's teleported states (no momentum cue available), perception noise hurt the
velocity-based model far more (19% vs 4.8% F1 loss at 2 cm), and onset detection was never
measured. They are one weakness -- a shortcut -- observed from three directions.

Reported metrics per model:

  ``f1_all``     -- the headline metric, for continuity with earlier tables.
  ``f1_onset``   -- restricted to objects at rest, i.e. the part requiring prediction.
  ``f1_moving``  -- restricted to objects already in motion, i.e. the part momentum solves.

Usage::

    python -m experiments.momentum_shortcut --counts 3 5 8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import compute_mask_metrics, load_dataset, load_sparse_model
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import StateLayout

# An object is "at rest" below this planar speed. Set well under the smallest speed that
# survives the hard-subset motion filter, so the split is between genuinely stationary
# objects and genuinely moving ones rather than an arbitrary cut through a continuum.
REST_SPEED = 2.55e-05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Momentum shortcut / onset analysis.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rest-speed", type=float, default=REST_SPEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="momentum_shortcut")
    parser.add_argument("--split-template", type=str,
                        default="data/transitions/splits_clean_{n}obj_s{seed}")
    parser.add_argument(
        "--stem-template",
        type=str,
        default="scale_{n}obj_s{seed}",
        help=(
            "Filename stem inside the split directory. The planar series carries its tag in "
            "the stem, so pass scale_{n}obj_planar_s{seed} for it."
        ),
    )
    parser.add_argument("--global-template", type=str,
                        default="models/checkpoints/sparse_clean_{n}obj_s{seed}.pt")
    parser.add_argument("--contact-template", type=str,
                        default="models/checkpoints/sparse_contact_clean_{n}obj_s{seed}.pt")
    return parser.parse_args()


def planar_speed(state: np.ndarray, num_objects: int) -> np.ndarray:
    layout = StateLayout(num_objects=num_objects)
    velocity = state[:, layout.object_velocity_slice].reshape(-1, num_objects, 6)
    # Columns 3:5 are the planar linear components in the shared velocity layout.
    return np.linalg.norm(velocity[:, :, 3:5], axis=2)


def predicted_mask(checkpoint: Path, dataset: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model, config = load_sparse_model(checkpoint, device)
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    feature_mode = str(config.get("feature_mode", "global"))
    chunks = []
    with torch.no_grad():
        for start in range(0, state.shape[0], 256):
            stop = min(start + 256, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            out = model(features, estimator=estimator, temperature=float(config["temperature"]), hard=True)
            chunks.append((out.gate.probs >= 0.5).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def score(prediction: np.ndarray, target: np.ndarray, at_rest: np.ndarray) -> dict[str, float]:
    moving = ~at_rest
    return {
        "f1_all": compute_mask_metrics(prediction, target)["f1"],
        "f1_onset": compute_mask_metrics(prediction[at_rest], target[at_rest])["f1"],
        "f1_moving": (
            compute_mask_metrics(prediction[moving], target[moving])["f1"] if moving.any() else float("nan")
        ),
        "onset_positive_rate": float(target[at_rest].mean()),
        "at_rest_fraction": float(at_rest.mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "momentum_shortcut", "counts": args.counts,
                       "seeds": args.seeds, "rest_speed": args.rest_speed})

    rows, skipped = [], []
    for count in args.counts:
        for seed in args.seeds:
            directory = Path(args.split_template.format(n=count, seed=seed))
            stem = args.stem_template.format(n=count, seed=seed)
            data_path = directory / f"{stem}_hard_test.npz"
            if not data_path.exists():
                skipped.append(str(data_path))
                continue
            dataset = load_dataset(data_path)
            target = dataset["target_mask"]
            at_rest = planar_speed(dataset["state"], count) <= args.rest_speed

            conditions = {
                # One line of code. The baseline the project never ran.
                "trivial_already_moving": (~at_rest).astype(np.float32),
            }
            for name, template in (("gate_global", args.global_template),
                                   ("gate_contact", args.contact_template)):
                checkpoint = Path(template.format(n=count, seed=seed))
                if checkpoint.exists():
                    conditions[name] = predicted_mask(checkpoint, dataset, device)
                else:
                    skipped.append(str(checkpoint))

            for name, prediction in conditions.items():
                rows.append({"object_count": count, "seed": seed, "condition": name,
                             **score(prediction, target, at_rest)})

    write_outputs(rows, output_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def build_summary(rows: list[dict]) -> dict:
    def mean(condition: str, metric: str, count: int | None = None) -> float | None:
        values = [r[metric] for r in rows if r["condition"] == condition
                  and (count is None or r["object_count"] == count)
                  and r[metric] == r[metric]]
        return float(np.mean(values)) if values else None

    counts = sorted({r["object_count"] for r in rows})
    per_count = {}
    for count in counts:
        per_count[f"{count}obj"] = {
            condition: {
                "f1_all": mean(condition, "f1_all", count),
                "f1_onset": mean(condition, "f1_onset", count),
                "f1_moving": mean(condition, "f1_moving", count),
            }
            for condition in sorted({r["condition"] for r in rows})
        }

    trivial_all = mean("trivial_already_moving", "f1_all")
    global_all = mean("gate_global", "f1_all")
    contact_onset = mean("gate_contact", "f1_onset")
    global_onset = mean("gate_global", "f1_onset")
    return {
        "per_count": per_count,
        # The two facts that reframe the paper.
        "trivial_baseline_beats_learned_gate_on_headline_metric": (
            bool(trivial_all > global_all) if (trivial_all and global_all) else None
        ),
        "velocity_free_beats_velocity_based_on_onset": (
            bool(contact_onset > global_onset) if (contact_onset and global_onset) else None
        ),
        "onset_f1": {"gate_global": global_onset, "gate_contact": contact_onset,
                     "trivial_already_moving": mean("trivial_already_moving", "f1_onset")},
    }


COLUMNS = ["object_count", "seed", "condition", "f1_all", "f1_onset", "f1_moving",
           "onset_positive_rate", "at_rest_fraction"]


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "momentum_shortcut.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# The momentum shortcut: what is change detection measuring?",
        "",
        "`f1_all` is the headline metric used throughout this project. `f1_onset` restricts to",
        "objects **at rest**, i.e. the part of the task that requires predicting contact rather",
        "than continuing momentum. `trivial_already_moving` is a one-line rule with no learning.",
        "Mean over seeds.",
        "",
        "| N | condition | F1 (all) | F1 (onset) | F1 (already moving) |",
        "|---|---|---|---|---|",
    ]
    for count in sorted({r["object_count"] for r in rows}):
        for condition in ("trivial_already_moving", "gate_global", "gate_contact"):
            matching = [r for r in rows if r["object_count"] == count and r["condition"] == condition]
            if not matching:
                continue
            md.append(
                f"| {count} | {condition} | {np.mean([r['f1_all'] for r in matching]):.4f} | "
                f"{np.mean([r['f1_onset'] for r in matching]):.4f} | "
                f"{np.nanmean([r['f1_moving'] for r in matching]):.4f} |"
            )
    (output_dir / "momentum_shortcut.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
