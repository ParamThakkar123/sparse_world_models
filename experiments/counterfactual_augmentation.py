"""W3: is the learned change gate a good enough local causal mask to generate data?

The gate is trained to detect change, but what it estimates is a **local causal mask**. If
that reading is right, the mask should support CoDA-style counterfactual splicing: relocate
the objects it marks causally inert, keep everything else, and the synthesized transition
should still obey the real dynamics. This experiment tests that claim twice over -- once
directly against the simulator, and once by whether the synthesized data is actually useful.

**Stage 1: validity, measured rather than argued.** Most counterfactual-augmentation work
cannot check its own output; here we can. Each synthetic transition is replayed through
MuJoCo -- ``set_planar_state`` into the synthesized configuration, apply the recorded
action, compare the true next state against the synthesized label. Four conditions:

  * ``oracle_mask``   -- splice using the ground-truth changed mask (upper bound).
  * ``learned_gate``  -- splice using the trained gate (the claim under test).
  * ``no_mask``       -- relocate objects chosen at random, ignoring causal structure.
  * ``real``          -- unmodified transitions, a floor for replay error itself.

``no_mask`` is the control that gives the experiment meaning. It produces the same *volume*
of extra data with the same machinery and none of the causal reasoning, so if it validates
as well as ``learned_gate``, the mask is doing nothing.

**Stage 2: does it help?** Self-bootstrapping, which is the only honest protocol: train the
gate on a data budget, use *that* gate to augment *that* budget, retrain from scratch on the
union, and compare against training on the budget alone. No ground-truth mask and no extra
real data enter anywhere. Run across the 10/25/50/100% budgets of the existing
sample-efficiency harness.

Usage::

    python -m experiments.counterfactual_augmentation --count 3 --seed 0
    python -m experiments.counterfactual_augmentation --count 3 --seed 0 --stages validity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from experiments.generate_transitions import POSITION_EPS, YAW_EPS, flatten_state
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout
from models.counterfactual import (
    generate_counterfactuals,
    placement_is_clear,
    recompute_labels,
)
from experiments.generate_transitions import build_env

CONDITIONS = ("real", "oracle_mask", "learned_gate", "no_mask")
BUDGETS = (0.10, 0.25, 0.50, 1.00)
# A synthetic transition counts as valid if every object lands within this of where the real
# simulator puts it. 5 mm is a quarter of the 2 cm displacement that defines "changed", so it
# is tight enough that a missed interaction cannot slip through.
VALIDITY_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W3 counterfactual augmentation study.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=str, nargs="+", default=["validity", "efficiency"],
                        choices=["validity", "efficiency"])
    parser.add_argument("--validity-episodes", type=int, default=20)
    parser.add_argument("--validity-max-steps", type=int, default=60)
    parser.add_argument("--augment-ratio", type=float, default=1.0,
                        help="Synthetic transitions to generate per real transition.")
    parser.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--sparsity-weight", type=float, default=0.2)
    parser.add_argument("--delta-head", type=str, default="mdn")
    parser.add_argument("--feature-mode", type=str, default="global")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--env", choices=["tabletop", "planar"], default="tabletop",
                        help="Simulator used for the stage-1 validity replay.")
    parser.add_argument("--object-bound", type=float, default=None,
                        help=(
                            "Placement half-extent for the validity env. MUST match the geometry "
                            "the evaluated data was generated with: the 8-object splits use "
                            "0.22/0.09, and leaving the env default (0.18/0.12) makes placement "
                            "of 8 boxes impossible and the run dies at reset."
                        ))
    parser.add_argument("--min-object-separation", type=float, default=None)
    parser.add_argument("--run-name", type=str, default="counterfactual_augmentation")
    parser.add_argument("--split-template", type=str,
                        default="data/transitions/splits_clean_{n}obj_s{seed}")
    return parser.parse_args()


def split_file(args: argparse.Namespace, kind: str, split: str) -> Path:
    directory = Path(args.split_template.format(n=args.count, seed=args.seed))
    return directory / f"scale_{args.count}obj_s{args.seed}_{kind}_{split}.npz"


def load_raw(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "s_t": data["s_t"].astype(np.float32),
        "a_t": data["a_t"].astype(np.float32),
        "s_t1": data["s_t1"].astype(np.float32),
        "object_change_mask": data["object_change_mask"].astype(np.float32),
        "object_delta": data["object_delta"].astype(np.float32),
    }


def gate_masks(checkpoint: Path, raw: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    """Predicted change mask from a trained gate, thresholded exactly as at evaluation."""
    model, config = load_sparse_model(checkpoint, device)
    state = torch.from_numpy(raw["s_t"]).to(device)
    action = torch.from_numpy(raw["a_t"]).to(device)
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


def verify_online(
    count: int, seed: int, episodes: int, max_steps: int, condition: str,
    rng: np.random.Generator, pose_pool: np.ndarray, gate_fn=None,
    env_name: str = "tabletop", overrides: dict | None = None,
) -> dict[str, float]:
    """Verify counterfactual splices against the simulator, exactly.

    An earlier version of this reconstructed the synthetic scene from the reduced planar
    state and replayed it. That does not work and the failure is instructive: the reduced
    state drops z, tilt, contact history and velocities, so re-entering a mid-contact
    configuration makes the solver eject objects — **unmodified** transitions replayed that
    way scored 0% valid with 6 m errors, i.e. the instrument had no resolution at all. Worse,
    the causally-blind ``no_mask`` condition scored *best*, because scattering objects apart
    reduced the interpenetration the reconstruction had introduced.

    This version never reconstructs. At each step it takes an exact
    :meth:`~models.envs.mujoco_tabletop.TabletopPushEnv.snapshot`, runs the real step to get
    the ground truth, restores the snapshot, relocates only the objects the condition selects
    (leaving every other degree of freedom bit-identical), and steps again with the same
    action. Any difference is then attributable to the relocation alone.

    Two things are scored:

    * ``relocated_stationary_fraction`` — did the relocated objects stay put, as the
      synthetic label asserts? This is the claim the augmentation actually makes.
    * ``others_match_fraction`` — did every *other* object move exactly as it did in the real
      step? This catches splices that silently perturb the dynamics they claim to preserve.

    A splice is ``valid`` only if both hold.
    """
    # Both simulators expose snapshot/restore/relocate_object, so this verification is
    # env-agnostic; only the construction differs.
    env = build_env(env_name, count, max_steps, seed, overrides or {})
    target_object = env.config.target_object
    from models.policies import ScriptedPushPolicy  # local import: only needed here

    relocated_ok = others_ok = both_ok = attempted = relocated_total = 0
    objects_stationary = 0
    for _ in range(episodes):
        obs = env.reset()
        policy = ScriptedPushPolicy(target_object=target_object)
        for _ in range(max_steps):
            action = policy.act(obs)
            before = env.get_state()["object_pose"].copy()
            snapshot = env.snapshot()

            next_obs, _, done, _ = env.step(action)
            after = env.get_state()["object_pose"].copy()
            moved = np.linalg.norm(after[:, :2] - before[:, :2], axis=1) > POSITION_EPS

            if condition == "no_mask":
                free = np.arange(count)
            elif condition == "learned_gate":
                if gate_fn is None:
                    raise ValueError("learned_gate needs gate_fn")
                # Query the gate on the state the agent was actually in, before the step.
                free = np.flatnonzero(~gate_fn(obs, action).astype(bool))
            else:
                free = np.flatnonzero(~moved)
            if free.size:
                env.restore(snapshot)
                chosen, placed = [], []
                for object_index in rng.permutation(free):
                    candidate = pose_pool[int(rng.integers(pose_pool.shape[0]))]
                    others = np.array([before[i, :2] for i in range(count) if i != object_index])
                    ok = condition == "no_mask" or placement_is_clear(
                        candidate[:2],
                        other_xy=others,
                        moved_xy=before[moved, :2],
                        pusher_xy=snapshot_pusher(env),
                        pusher_next_xy=snapshot_pusher(env),
                    )
                    if ok:
                        env.relocate_object(int(object_index), candidate[:2], float(candidate[2]))
                        chosen.append(int(object_index))
                        placed.append(candidate[:2].copy())
                if chosen:
                    attempted += 1
                    relocated_total += len(chosen)
                    env.step(action)
                    cf_after = env.get_state()["object_pose"]
                    per_object = [
                        np.linalg.norm(cf_after[i, :2] - p) <= VALIDITY_TOLERANCE
                        for i, p in zip(chosen, placed)
                    ]
                    objects_stationary += int(sum(per_object))
                    stationary = all(per_object)
                    untouched = [i for i in range(count) if i not in chosen]
                    others_match = all(
                        np.abs(cf_after[i] - after[i]).max() <= VALIDITY_TOLERANCE
                        for i in untouched
                    ) if untouched else True
                    relocated_ok += int(stationary)
                    others_ok += int(others_match)
                    both_ok += int(stationary and others_match)
                    # Put the simulator back on the real trajectory before continuing.
                    env.restore(snapshot)
                    env.step(action)
            obs = next_obs
            if done:
                break

    return {
        "splices": attempted,
        "objects_relocated": relocated_total,
        "relocated_stationary_fraction": relocated_ok / max(attempted, 1),
        # Per-object rate, which is the fair cross-condition comparison: `no_mask` relocates
        # every object while the masked conditions relocate only the inert ones, so a
        # per-splice "all stayed put" rate penalises whichever condition moves more.
        "per_object_stationary_fraction": objects_stationary / max(relocated_total, 1),
        "mean_objects_per_splice": relocated_total / max(attempted, 1),
        "others_match_fraction": others_ok / max(attempted, 1),
        "valid_fraction": both_ok / max(attempted, 1),
    }


def snapshot_pusher(env: TabletopPushEnv) -> np.ndarray:
    return env.get_observation()["pusher_xy"].copy()


def build_condition(
    condition: str, raw: dict[str, np.ndarray], learned: np.ndarray | None,
    rng: np.random.Generator, num_samples: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Produce the transitions for one condition, plus its augmentation statistics."""
    if condition == "real":
        return raw, {"note": "unmodified transitions"}

    mask = {
        "oracle_mask": raw["object_change_mask"],
        "learned_gate": learned,
        "no_mask": raw["object_change_mask"],  # ignored when respect_mask=False
    }[condition]
    if mask is None:
        raise ValueError("learned_gate condition needs a trained checkpoint")

    state, action, next_state, stats = generate_counterfactuals(
        raw["s_t"], raw["a_t"], raw["s_t1"], mask,
        rng=rng, num_samples=num_samples, respect_mask=(condition != "no_mask"),
    )
    change_mask, delta = recompute_labels(state, next_state, POSITION_EPS, YAW_EPS)
    return (
        {"s_t": state, "a_t": action, "s_t1": next_state,
         "object_change_mask": change_mask, "object_delta": delta},
        stats.as_dict(),
    )


def geometry_overrides(args: argparse.Namespace) -> dict:
    """Env config overrides matching the geometry the evaluated split was generated with."""
    overrides: dict = {}
    if args.object_bound is not None:
        overrides["object_bounds"] = (-abs(args.object_bound), abs(args.object_bound))
    if args.min_object_separation is not None:
        overrides["min_object_separation"] = args.min_object_separation
    return overrides


def build_gate_fn(checkpoint: Path, count: int, device: torch.device):
    """Return ``(obs, action) -> predicted change mask`` for a trained sparse checkpoint."""
    model, config = load_sparse_model(checkpoint, device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    feature_mode = str(config.get("feature_mode", "global"))
    temperature = float(config["temperature"])

    def gate_fn(obs: dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(flatten_state(obs).astype(np.float32)).unsqueeze(0).to(device)
        act = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = build_object_features_by_mode(state, act, feature_mode)
            out = model(features, estimator=estimator, temperature=temperature, hard=True)
            return (out.gate.probs >= 0.5).float().cpu().numpy()[0]

    return gate_fn


def run_validity(args: argparse.Namespace, device: torch.device, output_dir: Path) -> dict:
    """Stage 1: how often is a spliced transition actually consistent with the physics?"""
    raw = load_raw(split_file(args, "hard", "train"))
    layout = StateLayout(num_objects=args.count)
    pose_pool = raw["s_t"][:, layout.object_pose_slice].reshape(-1, POSE_DIM)

    checkpoint = Path(f"models/checkpoints/sparse_clean_{args.count}obj_s{args.seed}.pt")
    gate_fn = build_gate_fn(checkpoint, args.count, device) if checkpoint.exists() else None
    if gate_fn is None:
        print(f"[w3] {checkpoint} missing -- skipping the learned_gate condition", flush=True)

    results = {}
    for condition in ("oracle_mask", "learned_gate", "no_mask"):
        if condition == "learned_gate" and gate_fn is None:
            continue
        results[condition] = verify_online(
            args.count, args.seed, args.validity_episodes, args.validity_max_steps,
            condition, np.random.default_rng(args.seed), pose_pool, gate_fn, args.env,
            geometry_overrides(args),
        )
        r = results[condition]
        print(f"  {condition:14s} valid={r['valid_fraction']:.3f} "
              f"(relocated stayed put {r['relocated_stationary_fraction']:.3f}, "
              f"others unchanged {r['others_match_fraction']:.3f}) "
              f"over {r['splices']} splices", flush=True)

    (output_dir / "validity.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"validity": results}


def write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    # TransitionDataset reads only these keys, but `done` keeps the file loadable by the
    # rest of the tooling without special-casing.
    payload["done"] = np.zeros(data["s_t"].shape[0], dtype=bool)
    if payload["done"].size:
        payload["done"][-1] = True
    np.savez_compressed(path, **payload)


def train(run_name: str, train_path: Path, val_path: Path, args: argparse.Namespace) -> Path:
    checkpoint = Path("models/checkpoints") / f"{run_name}.pt"
    command = [
        sys.executable, "-m", "experiments.train_sparse_model",
        "--train", str(train_path), "--val", str(val_path),
        "--run-name", run_name,
        "--delta-head", args.delta_head,
        "--epochs", str(args.epochs),
        "--sparsity-weight", str(args.sparsity_weight),
        "--feature-mode", args.feature_mode,
        "--auto-balance-bce",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    subprocess.run(command, check=True)
    return checkpoint


def evaluate(checkpoint: Path, dataset: dict[str, np.ndarray], device: torch.device) -> dict:
    model, config = load_sparse_model(checkpoint, device)
    state = torch.from_numpy(dataset["state"]).to(device)
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

    current, following, target = dataset["current_pose"], dataset["next_pose"], dataset["target_mask"]
    pose = compute_pose_metrics(current + gate[:, :, None] * delta, current, following, target)
    mask_metrics = compute_mask_metrics(gate, target)
    noop = compute_pose_metrics(current.copy(), current, following, target)
    return {
        "f1": mask_metrics["f1"], "precision": mask_metrics["precision"],
        "recall": mask_metrics["recall"],
        "overall_l2": pose["overall_per_object_l2"],
        "changed_l2": pose["changed_object_l2"],
        "unchanged_l2": pose["unchanged_object_l2"],
        "changed_l2_margin_over_noop": noop["changed_object_l2"] - pose["changed_object_l2"],
    }


def run_efficiency(args: argparse.Namespace, device: torch.device, output_dir: Path) -> list[dict]:
    raw = load_raw(split_file(args, "hard", "train"))
    val_path = split_file(args, "hard", "val")
    test = load_dataset(split_file(args, "hard", "test"))
    scratch = output_dir / "data"
    rows: list[dict] = []

    total = raw["s_t"].shape[0]
    for budget in args.budgets:
        take = max(8, int(round(total * budget)))
        # A shuffled prefix, so every condition at a budget sees the same real subset.
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(total)[:take]
        subset = {key: value[order] for key, value in raw.items()}
        subset_path = scratch / f"budget{int(budget * 100)}_real.npz"
        write_npz(subset_path, subset)

        # Stage A -- the gate that will do the augmenting is trained on this budget alone.
        base_name = f"cfa_{args.count}obj_s{args.seed}_b{int(budget * 100)}_base"
        base_checkpoint = train(base_name, subset_path, val_path, args)
        rows.append({"budget": budget, "condition": "real_only", "num_train": take,
                     **evaluate(base_checkpoint, test, device)})

        learned = gate_masks(base_checkpoint, subset, device)
        num_synthetic = int(round(take * args.augment_ratio))

        for condition in ("learned_gate", "no_mask"):
            data, stats = build_condition(condition, subset, learned,
                                          np.random.default_rng(args.seed + 1), num_synthetic)
            if data["s_t"].shape[0] == 0:
                print(f"  [{condition}] produced nothing at budget {budget}", flush=True)
                continue
            merged = {key: np.concatenate([subset[key], data[key]], axis=0) for key in subset}
            merged_path = scratch / f"budget{int(budget * 100)}_{condition}.npz"
            write_npz(merged_path, merged)
            name = f"cfa_{args.count}obj_s{args.seed}_b{int(budget * 100)}_{condition}"
            checkpoint = train(name, merged_path, val_path, args)
            rows.append({
                "budget": budget, "condition": condition,
                "num_train": int(merged["s_t"].shape[0]), "num_synthetic": int(data["s_t"].shape[0]),
                "acceptance_rate": stats.get("acceptance_rate"),
                **evaluate(checkpoint, test, device),
            })
        for row in rows[-3:]:
            print(f"  budget={row['budget']:.2f} {row['condition']:12s} "
                  f"n={row['num_train']:5d} F1={row['f1']:.3f} margin={row['changed_l2_margin_over_noop']:+.4f}",
                  flush=True)
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({"task": "counterfactual_augmentation", "count": args.count, "env": args.env,
                       "seed": args.seed, "stages": args.stages,
                       "augment_ratio": args.augment_ratio, "budgets": args.budgets,
                       "delta_head": args.delta_head})

    summary: dict = {"count": args.count, "seed": args.seed}
    if "validity" in args.stages:
        print("[w3] stage 1: replaying synthetic transitions through the simulator", flush=True)
        summary.update(run_validity(args, device, output_dir))
    if "efficiency" in args.stages:
        print("[w3] stage 2: self-bootstrapped sample efficiency", flush=True)
        rows = run_efficiency(args, device, output_dir)
        write_efficiency(rows, output_dir)
        summary["efficiency"] = rows
        summary["augmentation_helps"] = {
            f"{row['budget']:.2f}": bool(
                row["f1"] > next(r["f1"] for r in rows
                                 if r["budget"] == row["budget"] and r["condition"] == "real_only")
            )
            for row in rows if row["condition"] == "learned_gate"
        }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2, default=str))


def write_efficiency(rows: list[dict], output_dir: Path) -> None:
    columns = ["budget", "condition", "num_train", "num_synthetic", "acceptance_rate",
               "f1", "precision", "recall", "overall_l2", "changed_l2",
               "unchanged_l2", "changed_l2_margin_over_noop"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(
            "" if row.get(c) is None else
            (f"{row[c]:.6f}" if isinstance(row.get(c), float) else str(row.get(c, "")))
            for c in columns
        ))
    (output_dir / "efficiency.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Counterfactual augmentation — self-bootstrapped sample efficiency",
        "",
        "`real_only` trains on the data budget alone. `learned_gate` adds counterfactuals",
        "spliced using the gate trained on that same budget. `no_mask` adds the same volume",
        "of splices with the causal mask ignored — the control that separates the mask from",
        "the extra data.",
        "",
        "| budget | condition | n train | n synth | accept | F1 | changed L2 | margin over no-op |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        accept = f"{row['acceptance_rate']:.3f}" if row.get("acceptance_rate") is not None else "—"
        md.append(
            f"| {row['budget']:.0%} | {row['condition']} | {row['num_train']} | "
            f"{row.get('num_synthetic', '—')} | {accept} | {row['f1']:.3f} | "
            f"{row['changed_l2']:.4f} | {row['changed_l2_margin_over_noop']:+.4f} |"
        )
    (output_dir / "efficiency.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
