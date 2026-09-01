"""Characterise every domain in the suite, and derive its motion threshold.

Two jobs, both prerequisites for the cross-domain shortcut study.

**1. Report the shortcut's raw ingredients per domain, before any model is trained.**
The momentum shortcut is not a fact about models; it is a fact about the *evaluation
population*. Its strength is fully determined by two conditional probabilities that can be
read straight off the simulator:

    P(changed | already moving)   -- how well "keep going" predicts change
    P(changed | at rest)          -- the onset rate, i.e. how much genuine prediction is left

The trivial rule's F1 is a deterministic function of those two numbers and the at-rest
fraction. Reporting them per domain says *why* the shortcut is strong or weak somewhere,
rather than only that it is, and it does so without a single trained model in the loop.

**2. Derive each domain's hard-subset motion threshold instead of guessing it.**
The 0.02 m threshold in ``create_hard_subset`` was tuned for the MuJoCo tabletop. Applying
it unchanged to a domain with smaller per-step displacements keeps almost nothing -- on the
planar domain it retained 0.13% of steps and left a 15-row training split, which produced
meaningless numbers before it was caught. Rather than repeat that discovery once per new
engine, each domain's threshold is derived so that the motion filter **retains the same
fraction of steps** it retains on the tabletop under the same settings -- a target that is
measured in this run rather than hardcoded, so the tabletop reproducing its own 0.020 is a
self-check on the calibration.

Equal retention, not equal threshold, is the right invariant. The threshold's only job is to
define "a step where something really moved", and a domain whose objects move half as far
per step needs half the threshold to mean the same thing. Calibrating on retention also
equalises the thing that actually broke the planar runs -- split size -- so the four
benchmarks differ in physics rather than in how much data survives their filter.

An earlier version of this script calibrated on the ratio between the threshold and the
median moved-object displacement, anchored to the tabletop's documented 0.0254. That was
circular and is recorded here so the mistake is not repeated: 0.0254 was measured on the
*already filtered* hard split, so it is conditioned on having passed the very threshold it
was being used to derive. Measured on raw data the tabletop's median is 0.0122, and the
"derived" threshold for the tabletop came out at 0.010 rather than reproducing its own
hand-tuned 0.020 -- which is the tell that the anchor was wrong.

The four domains and why they are the four::

    domain     engine       contact regime   post-contact motion   prediction
    ---------  -----------  ---------------  --------------------  ------------------
    tabletop   MuJoCo       impulsive 3D     short slide           (reference)
    planar     ours         quasi-static     stops immediately     weakest shortcut
    billiards  Box2D        near-elastic     very long             STRONGEST shortcut
    clutter    Chipmunk2D   high-friction    short, chained        weak shortcut

Two of the four engines are third-party and share no code with us or with each other, so
agreement across the suite cannot be an artefact of our implementation. The regimes bracket
manipulation from both ends, so agreement cannot be an artefact of one dynamical style.

**Pre-registered predictions, recorded before the run** (this project's convention; see
RESULTS.md for previous cases where writing the prediction down first is what made a
surprise legible):

  a. ``billiards`` has the highest P(changed | already moving) of the four -- near 1.0 --
     because near-elastic collisions keep objects moving for tens of steps.
  b. ``clutter`` has the lowest, because high friction stops objects within a step or two.
  c. Every domain has P(changed | already moving) >> P(changed | at rest), which is the
     shortcut's existence condition, and it holds regardless of engine.

Usage::

    python -m experiments.domain_characterization --domains tabletop planar billiards clutter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import ExperimentLogger
from experiments.generate_transitions import build_env, compute_diff_labels
from models.policies import RandomPolicy, ScriptedPushPolicy

# Matches experiments/momentum_shortcut.REST_SPEED so "at rest" means the same thing here as
# it does when the trained models are scored.
REST_SPEED = 2.55e-05

# The MuJoCo tabletop's hand-tuned threshold. It is the suite's anchor: the fraction of
# steps it retains ON THIS RUN'S SETTINGS is measured here and every other domain's
# threshold is set to reproduce that same retention.
#
# The target retention is measured rather than hardcoded because it depends on the geometry.
# RESULTS.md quotes 21.7% for the published tabletop series, but that series used the
# default placement bounds; the cross-domain suite runs at bounds +/-0.26, where contacts are
# rarer and the same 0.02 m threshold retains only ~6%. Hardcoding 21.7% would therefore have
# calibrated the three new domains against a number the reference domain does not itself hit
# under the settings actually in use -- and the tabletop's own derived threshold would not
# have come back as 0.020, which is the self-check that catches exactly this.
TABLETOP_THRESHOLD = 0.020
REFERENCE_DOMAIN = "tabletop"

DOMAINS = ("tabletop", "planar", "billiards", "clutter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Characterise each domain's change statistics.")
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--object-bound", type=float, default=0.26)
    parser.add_argument("--min-object-separation", type=float, default=0.09)
    parser.add_argument("--rest-speed", type=float, default=REST_SPEED)
    parser.add_argument(
        "--policy",
        choices=["scripted", "random", "mixed"],
        default="scripted",
        help=(
            "'mixed' alternates scripted and random episodes. The headline benchmarks use "
            "scripted data, so 'scripted' is the default and the one that characterises "
            "them; 'mixed' is available to check the shortcut is not an artefact of the "
            "scripted policy specifically."
        ),
    )
    parser.add_argument("--run-name", type=str, default="domain_characterization")
    return parser.parse_args()


def rollout(
    domain: str,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Collect one domain/count/seed probe and return per-object arrays."""
    overrides: dict[str, object] = {
        "object_bounds": (-args.object_bound, args.object_bound),
        "min_object_separation": args.min_object_separation,
    }
    env = build_env(domain, count, args.max_steps, seed, overrides)
    scripted = ScriptedPushPolicy()
    random_policy = RandomPolicy(seed=seed)

    at_rest, changed, displacement, step_max_displacement = [], [], [], []
    for episode in range(args.episodes):
        if args.policy == "scripted":
            policy = scripted
        elif args.policy == "random":
            policy = random_policy
        else:
            policy = scripted if episode % 2 == 0 else random_policy
        obs = env.reset()
        for _ in range(args.max_steps):
            action = policy.act(obs)
            next_obs, _, done, _ = env.step(action)
            mask, delta = compute_diff_labels(obs, next_obs)
            speed = np.linalg.norm(obs["object_velocities"][:, 3:5], axis=1)
            per_object = np.linalg.norm(delta[:, :2], axis=1)
            at_rest.append(speed <= args.rest_speed)
            changed.append(mask.astype(bool))
            displacement.append(per_object)
            # The hard-subset filter is a STEP-level decision -- keep the step if any object
            # cleared the threshold -- so retention has to be computed from the per-step max,
            # not from the per-object distribution.
            step_max_displacement.append(float(per_object.max()))
            obs = next_obs
            if done:
                break
    return {
        "at_rest": np.concatenate(at_rest),
        "changed": np.concatenate(changed),
        "displacement": np.concatenate(displacement),
        "step_max_displacement": np.asarray(step_max_displacement),
    }


def trivial_rule_f1(at_rest: np.ndarray, changed: np.ndarray) -> float:
    """F1 of "predict change iff already moving", computed directly.

    This is the whole baseline: one line, no parameters, no training. It is reported here on
    the *unfiltered* population; the benchmarks filter further, which is what pushes it as
    high as 0.90-0.99 in the motion-filtered case.
    """
    prediction = ~at_rest
    true_positive = float(np.sum(prediction & changed))
    if true_positive == 0.0:
        return 0.0
    precision = true_positive / float(np.sum(prediction))
    recall = true_positive / float(np.sum(changed))
    return 2.0 * precision * recall / (precision + recall)


def summarise(sample: dict[str, np.ndarray]) -> dict[str, float]:
    at_rest = sample["at_rest"]
    changed = sample["changed"]
    displacement = sample["displacement"]
    step_max = sample["step_max_displacement"]
    moving = ~at_rest
    moved_displacement = displacement[changed]
    return {
        "retention_at_tabletop_threshold": float((step_max >= TABLETOP_THRESHOLD).mean()),
        "n_object_steps": int(at_rest.size),
        "at_rest_fraction": float(at_rest.mean()),
        "changed_fraction": float(changed.mean()),
        # The two numbers that determine the shortcut's strength.
        "p_changed_given_moving": float(changed[moving].mean()) if moving.any() else float("nan"),
        "p_changed_given_rest": float(changed[at_rest].mean()) if at_rest.any() else float("nan"),
        "trivial_rule_f1_unfiltered": trivial_rule_f1(at_rest, changed),
        "median_moved_displacement": (
            float(np.median(moved_displacement)) if moved_displacement.size else float("nan")
        ),
        "p99_moved_displacement": (
            float(np.percentile(moved_displacement, 99)) if moved_displacement.size else float("nan")
        ),
    }


def aggregate(rows: list[dict], step_max: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per-domain means over counts and seeds, plus the derived motion threshold.

    ``step_max`` holds each domain's pooled per-step maximum displacement, which is what the
    retention calibration quantiles over. Pooling across counts and seeds before taking the
    quantile is deliberate: the threshold is one number per domain, so it should be fitted to
    that domain's whole population rather than averaged over per-cell quantile estimates.
    """
    per_domain: dict[str, dict] = {}
    if REFERENCE_DOMAIN in step_max:
        target_retention = float((step_max[REFERENCE_DOMAIN] >= TABLETOP_THRESHOLD).mean())
    else:
        # Without the reference domain there is nothing to anchor to. Fall back to the
        # published tabletop figure and say so in the summary rather than silently inventing
        # a target.
        target_retention = 0.217

    for domain in sorted({row["domain"] for row in rows}):
        matching = [row for row in rows if row["domain"] == domain]
        median_displacement = float(np.mean([row["median_moved_displacement"] for row in matching]))
        # Rounded to the nearest 0.001 m: the threshold is a task definition, and a value
        # quoted to four decimals would imply a precision the calibration does not have.
        threshold = round(float(np.quantile(step_max[domain], 1.0 - target_retention)), 3)
        per_domain[domain] = {
            "retention_at_tabletop_threshold": float(
                np.mean([r["retention_at_tabletop_threshold"] for r in matching])
            ),
            "target_step_retention": target_retention,
            "p_changed_given_moving": float(np.mean([r["p_changed_given_moving"] for r in matching])),
            "p_changed_given_rest": float(np.mean([r["p_changed_given_rest"] for r in matching])),
            "at_rest_fraction": float(np.mean([r["at_rest_fraction"] for r in matching])),
            "changed_fraction": float(np.mean([r["changed_fraction"] for r in matching])),
            "trivial_rule_f1_unfiltered": float(
                np.mean([r["trivial_rule_f1_unfiltered"] for r in matching])
            ),
            "median_moved_displacement": median_displacement,
            "derived_motion_threshold": threshold,
        }
    return per_domain


def check_predictions(per_domain: dict[str, dict]) -> dict[str, object]:
    """Score the pre-registered predictions in the module docstring."""
    available = set(per_domain)
    ranked = sorted(per_domain, key=lambda d: per_domain[d]["p_changed_given_moving"])
    result: dict[str, object] = {
        "ranking_by_p_changed_given_moving_ascending": ranked,
    }
    if {"billiards", "clutter"} <= available:
        result["billiards_has_strongest_shortcut"] = bool(ranked[-1] == "billiards")
        result["clutter_has_weakest_shortcut"] = bool(ranked[0] == "clutter")
    result["shortcut_exists_in_every_domain"] = bool(
        all(
            values["p_changed_given_moving"] > 5.0 * values["p_changed_given_rest"]
            for values in per_domain.values()
        )
    )
    return result


COLUMNS = [
    "domain", "object_count", "seed", "n_object_steps", "at_rest_fraction", "changed_fraction",
    "p_changed_given_moving", "p_changed_given_rest", "trivial_rule_f1_unfiltered",
    "median_moved_displacement", "p99_moved_displacement", "retention_at_tabletop_threshold",
]


def write_outputs(
    rows: list[dict],
    per_domain: dict[str, dict],
    target_retention: float,
    output_dir: Path,
) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(
                f"{row[column]:.6f}" if isinstance(row[column], float) else str(row[column])
                for column in COLUMNS
            )
        )
    (output_dir / "domain_characterization.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Domain suite characterisation",
        "",
        "Measured on the simulators directly, with no model trained. `P(chg | moving)` is what",
        "the trivial \"already moving\" rule exploits; `P(chg | rest)` is the onset rate, i.e.",
        "the part of the task that genuinely requires prediction. The motion threshold is",
        "derived, not guessed: it is the value that makes the hard-subset filter retain",
        f"{target_retention:.1%} of steps in this domain, which is what the tabletop's",
        f"hand-tuned {TABLETOP_THRESHOLD} m retains under these same settings. That target is",
        "measured here rather than hardcoded, so the tabletop's own derived threshold coming",
        "back as 0.020 is a self-check on the calibration. `retention @ 0.02` shows what the",
        "tabletop's threshold would have retained if applied unchanged -- the column that",
        "explains why a shared threshold silently destroyed the planar splits.",
        "",
        "| domain | engine | P(chg \\| moving) | P(chg \\| rest) | changed frac | median disp | retention @ 0.02 | motion threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    engines = {
        "tabletop": "MuJoCo",
        "planar": "ours",
        "billiards": "Box2D",
        "clutter": "Chipmunk2D",
    }
    for domain, values in per_domain.items():
        md.append(
            f"| {domain} | {engines.get(domain, '?')} | {values['p_changed_given_moving']:.4f} | "
            f"{values['p_changed_given_rest']:.4f} | {values['changed_fraction']:.3f} | "
            f"{values['median_moved_displacement']:.4f} | "
            f"{values['retention_at_tabletop_threshold']:.3f} | "
            f"{values['derived_motion_threshold']:.3f} |"
        )
    (output_dir / "domain_characterization.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config(vars(args))

    rows: list[dict] = []
    pooled_step_max: dict[str, list[np.ndarray]] = {}
    for domain in args.domains:
        for count in args.counts:
            for seed in args.seeds:
                sample = rollout(domain, count, seed, args)
                pooled_step_max.setdefault(domain, []).append(sample["step_max_displacement"])
                row = {"domain": domain, "object_count": count, "seed": seed, **summarise(sample)}
                rows.append(row)
                print(
                    f"{domain:10s} N={count:<3d} s={seed}  "
                    f"P(chg|mv)={row['p_changed_given_moving']:.4f}  "
                    f"P(chg|rest)={row['p_changed_given_rest']:.4f}  "
                    f"medDisp={row['median_moved_displacement']:.5f}",
                    flush=True,
                )

    step_max = {domain: np.concatenate(parts) for domain, parts in pooled_step_max.items()}
    per_domain = aggregate(rows, step_max)
    target_retention = next(iter(per_domain.values()))["target_step_retention"]
    write_outputs(rows, per_domain, target_retention, logger.run_dir)
    summary = {"per_domain": per_domain, "predictions": check_predictions(per_domain)}
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
