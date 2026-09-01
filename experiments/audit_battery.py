"""Run the trivial-rule battery against **any** change-detection benchmark.

`onset_shortcut_audit.py` is the version of this that runs against our own splits, our own
`StateLayout`, and our own checkpoints. It is the experiment. This file is the *deliverable*:
the same battery, with the project dependencies stripped out, so that somebody with a
different benchmark can point it at their data and find out in one command whether a
one-liner beats their model.

That distinction matters, because the whole finding is that this check is cheap and nobody
runs it. A battery that only runs on our data does not fix that.

## The claim this exists to test

Across three successive benchmarks -- motion-filtered, onset-filtered, interaction-filtered
-- the best parameter-free rule scored F1 0.93-0.96 every time, beating five published
object-centric world models and our own. Each filter removed its predecessor's shortcut and
was then won by a different one. So the question to ask of a new benchmark is not "does it
defeat the previously known shortcut" but "does it defeat the best one-liner anybody can
invent *against it*".

**Passing this battery is necessary, not sufficient.** The rules here are the ones that
happened to break our benchmarks. If your benchmark survives them, the honest conclusion is
"it survives these eleven rules", and the next step is to invent rules against *it* -- which
is exactly the step that would have saved us two benchmarks. `--extra-rules` exists for that.

## Input format

One `.npz` per split, with whatever of these arrays you have. Rules whose inputs are missing
are reported as `skipped`, never silently dropped:

    target_mask     (R, N)      required. 1 where object i changes at row r.
    object_xy       (R, N, 2)   required. Object positions (any consistent unit).
    object_speed    (R, N)      optional. Enables the velocity rules.
                                Or give `object_vel` (R, N, 2) and it is normed for you.
    actuator_xy     (R, 2)      optional. Enables the proximity rules.
    actuator_xy_next(R, 2)      optional. Post-action actuator position; falls back to
                                `actuator_xy` if absent, which weakens the proximity rules
                                rather than disabling them.

Thresholded rules fit their radius on the **validation** split and apply it unchanged to
test, so they get the same courtesy a learned model gets and cannot be dismissed as untuned.
If you pass no validation split the battery fits on train; if you pass neither, it refuses to
run rather than fitting on test.

## Usage

    # audit a third-party benchmark, comparing against your model's test predictions
    python -m experiments.audit_battery \
        --test  mybench_test.npz --val mybench_val.npz --train mybench_train.npz \
        --model-predictions my_model_test_preds.npz \
        --report audit_report.md

    # audit one of THIS project's splits (dogfooding; reproduces onset_shortcut_audit)
    python -m experiments.audit_battery --layout project --count 5 \
        --test  data/transitions/splits_onset_5obj_s0/onset_5obj_s0_hard_test.npz \
        --val   data/transitions/splits_onset_5obj_s0/onset_5obj_s0_hard_val.npz \
        --train data/transitions/splits_onset_5obj_s0/onset_5obj_s0_hard_train.npz

Exit status is 0 when the battery ran, 1 on a usage error. It is deliberately **not** keyed
to the verdict: a benchmark being degenerate is a finding, not a crash, and wiring it to an
exit code invites people to silence it in CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Rules are grouped by what they need, so a benchmark that ships only positions still gets a
# real audit instead of an error.
NEEDS_SPEED = ("already_moving", "moving_or_near", "near_a_mover", "near_pusher_or_mover",
               "moving_or_near_mover")
NEEDS_ACTUATOR = ("pusher_near", "pusher_approaching", "nearest_to_pusher",
                  "second_nearest_to_pusher", "two_nearest_to_pusher", "moving_or_near",
                  "near_pusher_or_mover")
FITTED = ("pusher_near", "pusher_approaching", "moving_or_near", "near_a_mover",
          "near_pusher_or_mover", "moving_or_near_mover")

# Defaults for this project's units (metres). `--radii` overrides for anything else; the
# sweep brackets the object half-extent and the actuator radius by a wide margin so a rule
# that could win is not excluded by too narrow a range.
DEFAULT_RADII = np.round(np.arange(0.02, 0.221, 0.005), 4)
PROJECT_REST_SPEED = 2.55e-05
PROJECT_PUSHER_ACTION_SCALE = 0.04
PROJECT_PUSHER_BOUND = 0.26


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Binary P/R/F1 over a flattened (rows x objects) mask."""
    predicted = prediction.reshape(-1) > 0.5
    actual = target.reshape(-1) > 0.5
    true_positive = float(np.sum(predicted & actual))
    false_positive = float(np.sum(predicted & ~actual))
    false_negative = float(np.sum(~predicted & actual))
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if actual.any() else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score(prediction: np.ndarray, data: dict, rest_speed: float) -> dict[str, float]:
    """Overall metrics plus **onset F1**, restricted to objects that are currently at rest.

    Onset F1 is the number that matters. The overall F1 on a motion-filtered benchmark is
    dominated by continuation -- an object in motion stays in motion -- so it measures
    persistence rather than prediction. Objects at rest can only start moving through
    contact, so that subset is the part of the task that requires a model.
    """
    result = mask_metrics(prediction, data["target_mask"])
    speed = data.get("object_speed")
    if speed is None:
        result["onset_f1"] = float("nan")
        return result
    at_rest = speed <= rest_speed
    result["onset_f1"] = (
        mask_metrics(prediction[at_rest], data["target_mask"][at_rest])["f1"]
        if at_rest.any() else float("nan")
    )
    return result


# --------------------------------------------------------------------------------------
# the battery
# --------------------------------------------------------------------------------------

def pairwise_distance(object_xy: np.ndarray) -> np.ndarray:
    """(R, N, N) object-object distances with an infinite diagonal."""
    distance = np.linalg.norm(object_xy[:, :, None, :] - object_xy[:, None, :, :], axis=-1)
    index = np.arange(object_xy.shape[1])
    distance[:, index, index] = np.inf
    return distance


def trivial_rules(data: dict, radius: float, rest_speed: float) -> dict[str, np.ndarray]:
    """The eleven rules. Missing inputs mean a rule is absent from the result, not zeroed.

    The first six are the original battery, invented against the motion and onset benchmarks.
    The last five were invented against the interaction benchmark **before it was built** --
    the commitment that keeps this from being circular. `near_a_mover` is the load-bearing
    one: in a contact chain the object that starts moving is by definition adjacent to
    something already in motion, so it is the natural shortcut for any chain benchmark and
    has to be ruled out explicitly rather than assumed away.
    """
    rules: dict[str, np.ndarray] = {}
    object_xy = data["object_xy"]
    rows, count = object_xy.shape[0], object_xy.shape[1]
    rules["always_change"] = np.ones((rows, count), dtype=np.float32)

    speed = data.get("object_speed")
    moving = None
    if speed is not None:
        moving = (speed > rest_speed).astype(np.float32)
        rules["already_moving"] = moving

    near = None
    actuator = data.get("actuator_xy_next")
    if actuator is not None:
        offset = object_xy - actuator[:, None, :]
        distance = np.linalg.norm(offset, axis=-1)
        near = (distance <= radius).astype(np.float32)
        rules["pusher_near"] = near

        before = data.get("actuator_xy")
        if before is not None:
            closing = distance < np.linalg.norm(object_xy - before[:, None, :], axis=-1)
            rules["pusher_approaching"] = near * closing.astype(np.float32)

        order = np.argsort(distance, axis=1)
        row_index = np.arange(rows)
        nearest = np.zeros_like(distance, dtype=np.float32)
        nearest[row_index, order[:, 0]] = 1.0
        rules["nearest_to_pusher"] = nearest
        if count >= 2:
            second = np.zeros_like(distance, dtype=np.float32)
            second[row_index, order[:, 1]] = 1.0
            rules["second_nearest_to_pusher"] = second
            two = nearest.copy()
            two[row_index, order[:, 1]] = 1.0
            rules["two_nearest_to_pusher"] = two

    if moving is not None and near is not None:
        rules["moving_or_near"] = np.maximum(moving, near)

    if moving is not None:
        # Distance to the nearest object that is CURRENTLY MOVING. Rows where nothing moves
        # give a min over an empty set (inf), which compares false, so those rows predict
        # nothing rather than everything.
        masked = np.where((moving > 0)[:, None, :], pairwise_distance(object_xy), np.inf)
        near_mover = (masked.min(axis=2) <= radius).astype(np.float32)
        rules["near_a_mover"] = near_mover
        rules["moving_or_near_mover"] = np.maximum(moving, near_mover)
        if near is not None:
            rules["near_pusher_or_mover"] = np.maximum(near, near_mover)

    return rules


def fit_radii(validation: dict, radii: np.ndarray, rest_speed: float) -> dict[str, float]:
    """Pick each thresholded rule's radius on VALIDATION, by F1.

    Fitting on validation rather than test is what makes these rules a fair comparator. A
    trivial baseline tuned on test would be the mirror image of the mistake this whole line
    of work exists to expose.
    """
    best: dict[str, float] = {}
    for name in FITTED:
        scores = []
        for radius in radii:
            rules = trivial_rules(validation, float(radius), rest_speed)
            if name not in rules:
                scores = []
                break
            scores.append(mask_metrics(rules[name], validation["target_mask"])["f1"])
        if scores:
            best[name] = float(radii[int(np.argmax(scores))])
    return best


def logistic_on_distance(train: dict, test: dict) -> np.ndarray | None:
    """Logistic regression on ONE feature: object-to-actuator distance.

    This is the control that separates two very different diagnoses, both consistent with a
    trivial rule beating a learned model: if a one-parameter convex model matches the hand
    rule, the label is recoverable from a single scalar and the *task* is degenerate; if it
    also underperforms, something is wrong with how models are being fitted rather than with
    the benchmark. It is also the direct answer to "you undertrained the gate".
    """
    if "actuator_xy_next" not in train or "actuator_xy_next" not in test:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    def features(data: dict) -> np.ndarray:
        offset = data["object_xy"] - data["actuator_xy_next"][:, None, :]
        return np.linalg.norm(offset, axis=-1).reshape(-1, 1)

    y_train = train["target_mask"].reshape(-1)
    if len(np.unique(y_train)) < 2:
        return None
    model = LogisticRegression(max_iter=1000).fit(features(train), y_train)
    return model.predict(features(test)).astype(np.float32).reshape(test["target_mask"].shape)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_generic(path: Path) -> dict:
    """Read the documented array format, normalising the optional velocity representation."""
    raw = dict(np.load(path, allow_pickle=False))
    data: dict[str, np.ndarray] = {}
    for key in ("target_mask", "object_xy", "object_speed", "actuator_xy", "actuator_xy_next"):
        if key in raw:
            data[key] = np.asarray(raw[key], dtype=np.float64 if key != "target_mask" else raw[key].dtype)
    if "object_speed" not in data and "object_vel" in raw:
        data["object_speed"] = np.linalg.norm(np.asarray(raw["object_vel"]), axis=-1)
    if "actuator_xy_next" not in data and "actuator_xy" in data:
        data["actuator_xy_next"] = data["actuator_xy"]
    missing = {"target_mask", "object_xy"} - set(data)
    if missing:
        raise SystemExit(f"{path}: missing required array(s) {sorted(missing)}")
    return data


def load_project(path: Path, count: int) -> dict:
    """Read one of THIS project's transition `.npz` files into the generic format.

    Kept here rather than in the caller so the standalone battery can be run against our own
    splits and checked against `onset_shortcut_audit.py`. A deliverable nobody dogfoods is a
    deliverable nobody has tested.
    """
    from models.layout import StateLayout

    raw = np.load(path, allow_pickle=False)
    layout = StateLayout(num_objects=count)
    state = raw["s_t"]
    poses = state[:, layout.object_pose_slice].reshape(-1, count, 3)
    # Columns 3:5 of the 6-dim per-object velocity block are the planar linear components,
    # matching `momentum_shortcut.planar_speed` so the two agree by construction.
    velocities = state[:, layout.object_velocity_slice].reshape(-1, count, 6)
    pusher = state[:, 0:2]
    action = np.clip(raw["a_t"], -1.0, 1.0)
    pusher_next = np.clip(
        pusher + action * PROJECT_PUSHER_ACTION_SCALE,
        -PROJECT_PUSHER_BOUND, PROJECT_PUSHER_BOUND,
    )
    return {
        "target_mask": raw["object_change_mask"],
        "object_xy": poses[:, :, :2],
        "object_speed": np.linalg.norm(velocities[:, :, 3:5], axis=-1),
        "actuator_xy": pusher,
        "actuator_xy_next": pusher_next,
    }


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def build_report(rows: list[dict], model_row: dict | None, skipped: dict[str, str],
                 context: dict) -> str:
    lines = ["# Trivial-rule audit", ""]
    lines.append(f"Rows: {context['rows']} | objects: {context['count']} | "
                 f"positive rate: {context['positive_rate']:.3f}")
    lines.append("")
    lines.append("| rule | params | F1 | onset F1 | precision | recall | radius |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: -r["f1"]):
        radius = f"{row['radius']:.3f}" if row.get("radius") is not None else "--"
        onset = "--" if np.isnan(row["onset_f1"]) else f"{row['onset_f1']:.4f}"
        lines.append(f"| `{row['rule']}` | {row['params']} | **{row['f1']:.4f}** | {onset} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {radius} |")
    if model_row is not None:
        onset = "--" if np.isnan(model_row["onset_f1"]) else f"{model_row['onset_f1']:.4f}"
        lines.append(f"| **your model** | -- | **{model_row['f1']:.4f}** | {onset} | "
                     f"{model_row['precision']:.4f} | {model_row['recall']:.4f} | -- |")
    lines.append("")

    if skipped:
        lines.append("**Skipped rules** (inputs not supplied -- these are *unrun*, not passed):")
        for name, reason in sorted(skipped.items()):
            lines.append(f"- `{name}`: {reason}")
        lines.append("")

    best = max(rows, key=lambda r: r["f1"])
    lines.append("## Verdict")
    lines.append("")
    plural = "parameter" if best["params"] == 1 else "parameters"
    lines.append(f"Best trivial rule: **`{best['rule']}`** at F1 **{best['f1']:.4f}** "
                 f"({best['params']} {plural}).")
    if model_row is None:
        lines.append("")
        lines.append("No model predictions supplied, so there is no verdict on whether a model "
                     "beats the battery. Pass `--model-predictions` to get one.")
    else:
        margin = model_row["f1"] - best["f1"]
        if margin <= 0:
            lines.append("")
            lines.append(f"**DEGENERATE.** The model loses to `{best['rule']}` by "
                         f"{-margin:.4f} F1. On this benchmark, any claim that the model has "
                         "learned which objects change is unsupported: a rule with "
                         f"{best['params']} {plural} does it better.")
        elif margin < 0.05:
            lines.append("")
            lines.append(f"**MARGINAL.** The model leads `{best['rule']}` by only {margin:.4f} "
                         "F1. A margin this small over a parameter-free rule does not support "
                         "a claim about learned change prediction; report the battery "
                         "alongside the model rather than instead of it.")
        else:
            lines.append("")
            lines.append(f"**SURVIVES THESE RULES.** The model leads `{best['rule']}` by "
                         f"{margin:.4f} F1. That is necessary, not sufficient -- these are the "
                         "rules that broke *our* benchmarks. Invent rules against yours and "
                         "add them with `--extra-rules` before claiming non-degeneracy.")
    if not np.isnan(best["onset_f1"]):
        lines.append("")
        lines.append(f"Onset F1 is the number to lead with: the best trivial rule reaches "
                     f"{best['onset_f1']:.4f} there. Overall F1 on a motion-filtered benchmark "
                     "measures continuation, not prediction.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--val", type=Path, help="Where thresholded rules fit their radius.")
    parser.add_argument("--train", type=Path,
                        help="Radius fallback if --val is absent, and the fit set for the "
                             "one-feature logistic control.")
    parser.add_argument("--model-predictions", type=Path,
                        help="`.npz` with a `prediction` array shaped like target_mask.")
    parser.add_argument("--layout", choices=["generic", "project"], default="generic")
    parser.add_argument("--count", type=int, help="Object count; required for --layout project.")
    parser.add_argument("--rest-speed", type=float, default=None,
                        help="Speed at or below which an object counts as at rest. Defaults to "
                             f"{PROJECT_REST_SPEED} for --layout project and 0.0 otherwise.")
    parser.add_argument("--radii", type=float, nargs="+", default=None,
                        help="Candidate contact radii, in your position units.")
    parser.add_argument("--extra-rules", type=Path,
                        help="Python file defining `rules(data, radius, rest_speed) -> dict`, "
                             "merged into the battery. This is where rules invented against "
                             "YOUR benchmark go.")
    parser.add_argument("--report", type=Path, help="Write the markdown report here.")
    parser.add_argument("--json", type=Path, help="Write machine-readable results here.")
    args = parser.parse_args()

    if args.layout == "project" and args.count is None:
        raise SystemExit("--layout project requires --count")
    rest_speed = args.rest_speed
    if rest_speed is None:
        rest_speed = PROJECT_REST_SPEED if args.layout == "project" else 0.0
    radii = np.asarray(args.radii, dtype=float) if args.radii else DEFAULT_RADII

    def load(path: Path) -> dict:
        return load_project(path, args.count) if args.layout == "project" else load_generic(path)

    test = load(args.test)
    fit_split_path = args.val or args.train
    if fit_split_path is None:
        raise SystemExit(
            "Pass --val (or at least --train). Fitting the thresholded rules on test would "
            "make this battery commit the error it exists to detect."
        )
    fit_split = load(fit_split_path)
    if args.val is None:
        print("WARNING: no --val; fitting rule radii on train.")

    radius_by_rule = fit_radii(fit_split, radii, rest_speed)
    default_radius = float(np.median(list(radius_by_rule.values()))) if radius_by_rule else float(radii[0])

    available = trivial_rules(test, default_radius, rest_speed)
    skipped: dict[str, str] = {}
    for name in NEEDS_SPEED:
        if name not in available:
            skipped[name] = "needs `object_speed` (or `object_vel`)"
    for name in NEEDS_ACTUATOR:
        if name not in available and name not in skipped:
            skipped[name] = "needs `actuator_xy`"

    if args.extra_rules:
        namespace: dict = {}
        exec(compile(args.extra_rules.read_text(encoding="utf-8"), str(args.extra_rules), "exec"),
             namespace)
        if "rules" not in namespace:
            raise SystemExit(f"{args.extra_rules}: no `rules(data, radius, rest_speed)` function")
        available.update(namespace["rules"](test, default_radius, rest_speed))

    parameter_count = {name: (1 if name in FITTED else 0) for name in available}

    rows = []
    for name, prediction in available.items():
        radius = radius_by_rule.get(name)
        if radius is not None:
            prediction = trivial_rules(test, radius, rest_speed)[name]
        rows.append({"rule": name, "params": parameter_count.get(name, 0),
                     "radius": radius, **score(prediction, test, rest_speed)})

    if args.train is not None:
        logistic = logistic_on_distance(load(args.train), test)
        if logistic is not None:
            rows.append({"rule": "logistic_on_distance", "params": 2, "radius": None,
                         **score(logistic, test, rest_speed)})

    model_row = None
    if args.model_predictions:
        prediction = np.load(args.model_predictions, allow_pickle=False)["prediction"]
        if prediction.shape != test["target_mask"].shape:
            raise SystemExit(
                f"--model-predictions shape {prediction.shape} != target_mask "
                f"{test['target_mask'].shape}"
            )
        model_row = {"rule": "model", "params": None, "radius": None,
                     **score(prediction, test, rest_speed)}

    context = {"rows": int(test["target_mask"].shape[0]),
               "count": int(test["target_mask"].shape[1]),
               "positive_rate": float(np.mean(test["target_mask"] > 0.5))}
    report = build_report(rows, model_row, skipped, context)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"context": context, "rules": rows, "model": model_row, "skipped": skipped,
             "rest_speed": rest_speed}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
