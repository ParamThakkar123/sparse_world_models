"""Does the CORRECTED benchmark have a shortcut of its own?

This applies the project's own criticism to the project's own fix. The motion benchmark was
discredited by a one-line rule nobody had thought to run -- "predict change iff the object is
already moving" -- which beat every learned model. The onset benchmark was built to remove
that. It would be a poor correction if it merely swapped one one-liner for another, and the
only honest way to find out is to try hard to break it.

**Eleven** trivial rules are run on every benchmark. None has any learned parameters; the
thresholded ones have their radius fitted on the **validation** split and applied unchanged to
test, so they get the same courtesy a learned model gets and cannot be dismissed as untuned.

The first six are the original battery. The last five were added later, invented **against
the interaction benchmark** once it existed -- the commitment that keeps this from being
circular. A benchmark designed to defeat the rules used to build it, which is then won by the
next obvious one-liner, is no improvement on what it replaced.

  ``already_moving``      the original shortcut: predict change iff the object is in motion.
  ``pusher_near``         predict change iff the post-action pusher position is within a
                          fitted radius. This is the obvious onset-benchmark shortcut: onset
                          events are contact-driven by construction, so "something is about to
                          be touched" is exactly the signal the corrected task rewards.
  ``pusher_approaching``  ``pusher_near`` AND the action points from pusher toward the object,
                          which removes contacts that are separating rather than closing.
  ``nearest_to_pusher``   predict change for the single object closest to the pusher, nothing
                          else. Parameter-free, and a strong rule whenever contacts are
                          one-at-a-time.
  ``moving_or_near``      the disjunction of the first two, i.e. the best a reader could do by
                          combining both shortcuts by hand.
  ``always_change``       the ungated degeneracy, as a floor.

Added against the interaction benchmark:

  ``second_nearest_to_pusher`` / ``two_nearest_to_pusher``
                          the obvious follow-ups once "nearest" is defeated.
  ``near_a_mover``        predict change iff within a fitted radius of an object that is
                          ALREADY MOVING. This is the one that matters: in a contact chain the
                          object that starts moving is by definition adjacent to a mover, so
                          this is the natural shortcut for any chain-based benchmark and has
                          to be ruled out explicitly rather than assumed away.
  ``near_pusher_or_mover`` / ``moving_or_near_mover``
                          the hand-combined versions, i.e. the best a reader could do by
                          stacking the available shortcuts.

**What each outcome would mean, written before the run.**

* If ``pusher_near`` matches or beats the learned gate on the onset benchmark, then the
  correction is incomplete: the corrected task is solvable by contact geometry alone with no
  learning, and the paper must say so and propose a further correction rather than presenting
  the onset benchmark as fixed.
* If it lands well below the learned gate, the corrected benchmark is doing what it was built
  to do -- rewarding a model that predicts *which* contacts transfer enough momentum to move
  an object, which distance alone cannot say.
* Either way ``already_moving`` should collapse on the onset benchmark, since that is the
  shortcut the filter was designed to remove; if it does not, the filter is not working.

**The prediction recorded before running** is the middle one: ``pusher_near`` will be clearly
better than chance -- proximity genuinely predicts contact -- but clearly worse than the
learned gate, because whether a touched object actually moves depends on the contact angle,
the object's orientation and whether the pusher is closing or grazing. Being partially
predictable is expected and fine; being *fully* predictable would not be.

Usage::

    python -m experiments.onset_shortcut_audit --benchmarks motion onset --counts 3 5 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import compute_mask_metrics, load_dataset
from experiments.momentum_shortcut import planar_speed, predicted_mask
from experiments.statistics import bootstrap_ci
from models.layout import StateLayout

REST_SPEED = 2.55e-05
# Candidate contact radii, in metres, swept on validation. The range brackets the object
# half-extent (0.025) and the pusher radius (0.02) by a wide margin in both directions, so a
# rule that could win is not excluded by too narrow a sweep.
RADII = np.round(np.arange(0.02, 0.221, 0.005), 4)

# Matches PUSHER_ACTION_SCALE / PUSHER_BOUND in experiments.train_sparse_model, so the
# post-action pusher position computed here is the same one the contact featurisation sees.
PUSHER_ACTION_SCALE = 0.04
PUSHER_BOUND = 0.26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the corrected benchmark for shortcuts.")
    parser.add_argument("--benchmarks", nargs="+", default=["motion", "onset"])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--split-templates", nargs="+",
        default=[
            "motion=data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz",
            "onset=data/transitions/splits_onset_{n}obj_s{seed}/onset_{n}obj_s{seed}_hard_{split}.npz",
        ],
        help="benchmark=template pairs; template takes {n}, {seed}, {split}.",
    )
    parser.add_argument(
        "--gate-templates", nargs="+",
        default=[
            "motion=models/checkpoints/sparse_clean_{n}obj_s{seed}.pt",
            "onset=models/checkpoints/sparse_onset_{n}obj_s{seed}.pt",
        ],
        help="The learned gate each benchmark's trivial rules are compared against.",
    )
    parser.add_argument(
        "--contact-gate-templates", nargs="+",
        default=[
            "motion=models/checkpoints/sparse_contact_clean_{n}obj_s{seed}.pt",
            "onset=models/checkpoints/sparse_onset_contact_{n}obj_s{seed}.pt",
        ],
        help=(
            "The velocity-FREE gate, whose features already contain signed contact distance. "
            "This is the control that separates 'the benchmark is trivially solvable' from "
            "'the model was given the wrong features': a proximity rule beating a "
            "velocity-featurised gate says little, but beating a gate that can see contact "
            "distance says the task is genuinely degenerate."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="onset_shortcut_audit")
    return parser.parse_args()


def parse_mapping(pairs: list[str]) -> dict[str, str]:
    mapping = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        mapping[key] = value
    return mapping


def pusher_geometry(dataset: dict[str, np.ndarray], count: int) -> tuple[np.ndarray, np.ndarray]:
    """``(contact_distance, closing)`` per object.

    ``contact_distance`` is from the object to the pusher's position *after* this action --
    the same quantity the contact featurisation exposes. ``closing`` is whether the action
    moves the pusher toward the object rather than away from it.
    """
    layout = StateLayout(num_objects=count)
    state = dataset["state"]
    pusher = state[:, 0:2]
    action = np.clip(dataset["action"], -1.0, 1.0)
    pusher_next = np.clip(pusher + action * PUSHER_ACTION_SCALE, -PUSHER_BOUND, PUSHER_BOUND)
    obj_xy = state[:, layout.object_pose_slice].reshape(-1, count, 3)[:, :, :2]

    to_object_before = obj_xy - pusher[:, None, :]
    to_object_after = obj_xy - pusher_next[:, None, :]
    distance = np.linalg.norm(to_object_after, axis=-1)
    closing = np.linalg.norm(to_object_after, axis=-1) < np.linalg.norm(to_object_before, axis=-1)
    return distance, closing


def object_distance_matrix(dataset: dict[str, np.ndarray], count: int) -> np.ndarray:
    """Pairwise object-object distances, ``(rows, count, count)`` with an infinite diagonal."""
    layout = StateLayout(num_objects=count)
    obj_xy = dataset["state"][:, layout.object_pose_slice].reshape(-1, count, 3)[:, :, :2]
    distance = np.linalg.norm(obj_xy[:, :, None, :] - obj_xy[:, None, :, :], axis=-1)
    idx = np.arange(count)
    distance[:, idx, idx] = np.inf
    return distance


def trivial_rules(
    dataset: dict[str, np.ndarray], count: int, radius: float
) -> dict[str, np.ndarray]:
    """The battery.

    The first six are the original set. The last five were invented **against the interaction
    benchmark** after it was built, which is the commitment that keeps this honest: a
    benchmark that defeats the rules used to design it, and is then won by the next obvious
    one-liner, is no improvement. ``near_a_mover`` is the important one -- in a contact chain
    the object that starts moving is by definition adjacent to an object already in motion, so
    "predict change near a mover" is the natural shortcut for a chain benchmark and has to be
    ruled out explicitly.
    """
    at_rest = planar_speed(dataset["state"], count) <= REST_SPEED
    distance, closing = pusher_geometry(dataset, count)
    moving = (~at_rest).astype(np.float32)
    near = (distance <= radius).astype(np.float32)
    order = np.argsort(distance, axis=1)
    rows = np.arange(distance.shape[0])

    nearest = np.zeros_like(distance, dtype=np.float32)
    nearest[rows, order[:, 0]] = 1.0

    second = np.zeros_like(distance, dtype=np.float32)
    if count >= 2:
        second[rows, order[:, 1]] = 1.0

    two_nearest = np.copy(nearest)
    if count >= 2:
        two_nearest[rows, order[:, 1]] = 1.0

    # Chain-aware rules: distance to the nearest object that is CURRENTLY MOVING.
    pairwise = object_distance_matrix(dataset, count)
    moving_mask = (~at_rest)[:, None, :]
    masked = np.where(moving_mask, pairwise, np.inf)
    distance_to_mover = masked.min(axis=2)
    near_mover = (distance_to_mover <= radius).astype(np.float32)
    # Guard the all-static rows: min over an empty set is inf, which compares false, so those
    # rows correctly predict nothing rather than everything.

    return {
        "already_moving": moving,
        "pusher_near": near,
        "pusher_approaching": (near * closing.astype(np.float32)),
        "nearest_to_pusher": nearest,
        "moving_or_near": np.maximum(moving, near),
        "always_change": np.ones_like(moving),
        # --- invented against the interaction benchmark ---
        "second_nearest_to_pusher": second,
        "two_nearest_to_pusher": two_nearest,
        "near_a_mover": near_mover,
        "near_pusher_or_mover": np.maximum(near, near_mover),
        "moving_or_near_mover": np.maximum(moving, near_mover),
    }


def fit_radius(validation: dict[str, np.ndarray], count: int) -> dict[str, float]:
    """Pick each thresholded rule's radius on VALIDATION, by F1.

    Fitting on validation rather than test is what makes these rules a fair comparator: a
    trivial baseline tuned on the test split would be the mirror image of the mistake this
    whole line of work exists to expose.
    """
    best: dict[str, float] = {}
    for name in ("pusher_near", "pusher_approaching", "moving_or_near",
                 "near_a_mover", "near_pusher_or_mover", "moving_or_near_mover"):
        scores = []
        for radius in RADII:
            prediction = trivial_rules(validation, count, float(radius))[name]
            scores.append(compute_mask_metrics(prediction, validation["target_mask"])["f1"])
        best[name] = float(RADII[int(np.argmax(scores))])
    return best


def logistic_on_distance(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray], count: int
) -> np.ndarray:
    """Logistic regression on ONE feature: distance from the object to the post-action pusher.

    The control that separates two very different diagnoses, both consistent with a trivial
    rule beating the learned gate:

    * If this one-parameter model matches the hand-tuned proximity rule, then the benchmark
      really is solvable from a single scalar and the *task* is degenerate.
    * If it also underperforms the hand rule, then something is wrong with how models are
      being fitted here rather than with the benchmark, and the audit's conclusion would not
      follow.

    It is also the direct answer to "you just undertrained the gate": this is trained on the
    same rows, with a convex objective and no tuning to get wrong.
    """
    from sklearn.linear_model import LogisticRegression

    def features(dataset: dict[str, np.ndarray]) -> np.ndarray:
        distance, _ = pusher_geometry(dataset, count)
        return distance.reshape(-1, 1)

    x_train, y_train = features(train), train["target_mask"].reshape(-1)
    if len(np.unique(y_train)) < 2:
        return np.zeros_like(test["target_mask"])
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    predicted = model.predict(features(test)).astype(np.float32)
    return predicted.reshape(test["target_mask"].shape)


def score(prediction: np.ndarray, dataset: dict[str, np.ndarray], count: int) -> dict[str, float]:
    at_rest = planar_speed(dataset["state"], count) <= REST_SPEED
    target = dataset["target_mask"]
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
    split_templates = parse_mapping(args.split_templates)
    gate_templates = parse_mapping(args.gate_templates)
    contact_gate_templates = parse_mapping(args.contact_gate_templates)

    rows: list[dict] = []
    skipped: list[str] = []
    for benchmark in args.benchmarks:
        template = split_templates[benchmark]
        for count in args.counts:
            for seed in args.seeds:
                paths = {
                    split: Path(template.format(n=count, seed=seed, split=split))
                    for split in ("train", "val", "test")
                }
                if not all(path.exists() for path in paths.values()):
                    skipped.append(str(paths["test"]))
                    continue
                validation = load_dataset(paths["val"])
                test = load_dataset(paths["test"])
                train = load_dataset(paths["train"])
                radii = fit_radius(validation, count)

                # Each rule uses its own validation-fitted radius; the unthresholded ones
                # ignore it.
                predictions: dict[str, np.ndarray] = {}
                for name in ("already_moving", "nearest_to_pusher", "always_change",
                             "second_nearest_to_pusher", "two_nearest_to_pusher"):
                    predictions[name] = trivial_rules(test, count, 0.0)[name]
                for name, radius in radii.items():
                    predictions[name] = trivial_rules(test, count, radius)[name]
                # The "did you just undertrain it?" control -- see logistic_on_distance.
                predictions["logistic_on_distance"] = logistic_on_distance(train, test, count)

                for label, templates in (("learned_gate", gate_templates),
                                         ("learned_gate_contact", contact_gate_templates)):
                    gate_path = Path(templates[benchmark].format(n=count, seed=seed))
                    if gate_path.exists():
                        predictions[label] = predicted_mask(gate_path, test, device)
                    else:
                        skipped.append(str(gate_path))

                for name, prediction in predictions.items():
                    rows.append({
                        "benchmark": benchmark, "object_count": count, "seed": seed,
                        "rule": name,
                        "fitted_radius": radii.get(name, float("nan")),
                        **score(prediction, test, count),
                    })

    summary = build_summary(rows)
    summary["skipped"] = len(skipped)
    summary["skipped_examples"] = skipped[:5]
    write_outputs(rows, summary, logger.run_dir)
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


RULES = ("already_moving", "pusher_near", "pusher_approaching", "nearest_to_pusher",
         "moving_or_near", "always_change",
         "second_nearest_to_pusher", "two_nearest_to_pusher", "near_a_mover",
         "near_pusher_or_mover", "moving_or_near_mover", "logistic_on_distance",
         "learned_gate", "learned_gate_contact")
COLUMNS = ["benchmark", "object_count", "seed", "rule", "fitted_radius", "f1", "onset_f1",
           "precision", "recall"]


def build_summary(rows: list[dict]) -> dict:
    per_benchmark: dict[str, dict] = {}
    for benchmark in sorted({r["benchmark"] for r in rows}):
        entry: dict[str, object] = {}
        for rule in RULES:
            values = np.asarray(
                [r["f1"] for r in rows if r["benchmark"] == benchmark and r["rule"] == rule]
            )
            if values.size == 0:
                continue
            mean, low, high = bootstrap_ci(values)
            entry[rule] = {"f1": mean, "ci": [low, high], "n": int(values.size)}

        # The verdict compares against the BEST learned gate available, so a trivial rule
        # only "wins" if it beats the strongest model we have on that benchmark.
        gates = [entry[name] for name in ("learned_gate", "learned_gate_contact")
                 if isinstance(entry.get(name), dict)]
        gate = max(gates, key=lambda g: g["f1"]) if gates else None
        best_trivial_name, best_trivial = None, -1.0
        for rule in RULES:
            if rule.startswith("learned_gate") or rule not in entry:
                continue
            value = entry[rule]["f1"]  # type: ignore[index]
            if value > best_trivial:
                best_trivial_name, best_trivial = rule, value
        entry["best_trivial_rule"] = best_trivial_name
        if isinstance(gate, dict):
            entry["gate_minus_best_trivial"] = float(gate["f1"]) - best_trivial
            # The audit's verdict for this benchmark.
            entry["a_trivial_rule_beats_the_learned_gate"] = bool(best_trivial > float(gate["f1"]))
        per_benchmark[benchmark] = entry

    return {
        "per_benchmark": per_benchmark,
        "verdict": {
            benchmark: entry.get("a_trivial_rule_beats_the_learned_gate")
            for benchmark, entry in per_benchmark.items()
        },
    }


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS
        ))
    (output_dir / "onset_shortcut_audit.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Auditing the corrected benchmark for shortcuts of its own",
        "",
        "Six parameter-free (or validation-fitted) rules against the learned gate, on both",
        "benchmarks. `pusher_near` is the rule that would discredit the onset benchmark the way",
        "`already_moving` discredited the motion benchmark: onset events are contact-driven by",
        "construction, so proximity is the obvious thing to try. Thresholds are fitted on",
        "validation and applied unchanged to test.",
        "",
        "| benchmark | rule | F1 | onset F1 | recall |",
        "|---|---|---|---|---|",
    ]
    for benchmark in sorted({r["benchmark"] for r in rows}):
        for rule in RULES:
            matching = [r for r in rows if r["benchmark"] == benchmark and r["rule"] == rule]
            if not matching:
                continue
            md.append(
                f"| {benchmark} | {rule} | {np.mean([r['f1'] for r in matching]):.4f} | "
                f"{np.nanmean([r['onset_f1'] for r in matching]):.4f} | "
                f"{np.mean([r['recall'] for r in matching]):.4f} |"
            )
    md += ["", "**Verdict** (does any trivial rule beat the learned gate?):", ""]
    for benchmark, verdict in summary["verdict"].items():
        md.append(f"- `{benchmark}`: **{verdict}**")
    (output_dir / "onset_shortcut_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
