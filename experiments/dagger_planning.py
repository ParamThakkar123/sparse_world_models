"""W5: DAgger on planner-visited states, with the W1 delta head.

The Phase-5 diagnosis was a distribution mismatch, not an architecture failure. A model
trained on scripted-policy transitions is queried by CEM on *teleported, near-rest states
from arbitrary approach angles* -- states no scripted episode ever produces -- so the gate
never fires and the delta head returns a near-constant drift. Contact-aware features plus
mixed-policy data lifted success from 0.00 to 0.23, but that only widens the training
distribution; it does not aim it at the states the planner actually reaches.

DAgger closes the loop directly. Each round:

  1. Run MPC episodes with the **current** model, recording every real state it visits.
  2. At each visited state, take an exact ``snapshot``, apply several candidate actions,
     read the **true** next state from the simulator, and ``restore``. Those are ground-truth
     labels for exactly the (state, action) pairs the planner queries.
  3. Aggregate with all previous data and retrain from scratch.
  4. Re-evaluate planning success.

Note what is and is not being imitated. Classic DAgger labels an *expert action*; here the
label is the *true transition*, because the object being corrected is the world model rather
than a policy. The oracle planner already in this repo confirms the planner itself is sound
(1.00 success with the true simulator), so model quality is the only thing left to fix.

Actions are drawn from a deliberate mixture: uniform samples cover what CEM explores early,
and perturbations of the executed plan cover what it converges to. Labelling only the
executed action would collect a single action per state and teach the model nothing about
the alternatives the planner must score.

Usage::

    python -m experiments.dagger_planning --rounds 3 --episodes-per-round 12
    python -m experiments.dagger_planning --rounds 3 --delta-head mdn --feature-mode contact
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import load_sparse_model
from experiments.generate_transitions import compute_diff_labels, flatten_state
from experiments.planning_mpc import (
    CEMConfig,
    SparseForward,
    cem_plan,
    model_evaluator,
)
from models import StateLayout
from models.envs.mujoco_tabletop import TabletopPushConfig, TabletopPushEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W5 DAgger loop for planner-visited states.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes-per-round", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--actions-per-state", type=int, default=4,
                        help="Candidate actions labelled by the simulator at each visited state.")
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-head", type=str, default="mdn",
                        help="W1 found the mixture head clears the no-op floor where MSE does not.")
    parser.add_argument("--mixture-components", type=int, default=3,
                        help="W1's component sweep plateaus at 3.")
    parser.add_argument("--feature-mode", type=str, default="contact",
                        help="Velocity-free contact features; 'global' is OOD under planning.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--sparsity-weight", type=float, default=0.05)
    parser.add_argument("--seed-data", type=Path,
                        default=Path("data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_train.npz"))
    parser.add_argument("--val-data", type=Path,
                        default=Path("data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_val.npz"))
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=3.0)
    parser.add_argument("--proximity-weight", type=float, default=0.3)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="dagger_planning")
    return parser.parse_args()


def make_env(num_objects: int, seed: int, max_steps: int) -> TabletopPushEnv:
    return TabletopPushEnv(
        TabletopPushConfig(num_objects=num_objects, max_steps=max_steps, seed=seed)
    )


def label_with_simulator(
    env: TabletopPushEnv, state_vec: np.ndarray, actions: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Ground-truth transitions for several actions from the *current* simulator state.

    Uses snapshot/restore so every action is evaluated from an identical starting point and
    the environment is left exactly where it began -- the episode continues on its real
    trajectory, unperturbed by the labelling.
    """
    snapshot = env.snapshot()
    before = env.get_observation()
    labelled = []
    for action in actions:
        env.restore(snapshot)
        after, _, _, _ = env.step(action.astype(np.float32))
        change_mask, delta = compute_diff_labels(before, after)
        labelled.append(
            (state_vec.copy(), action.astype(np.float32),
             flatten_state(after).astype(np.float32), change_mask, delta)
        )
    env.restore(snapshot)
    return labelled


def sample_candidate_actions(
    plan: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Half uniform, half jittered around the planned action.

    Uniform samples cover CEM's early exploration; the jittered ones cover the neighbourhood
    it converges to. Labelling only the executed action would give one action per state,
    which teaches nothing about the alternatives the planner has to score.
    """
    uniform_count = max(1, count // 2)
    uniform = rng.uniform(-1.0, 1.0, size=(uniform_count, 2))
    jittered = np.clip(
        plan[None, :] + rng.normal(0.0, 0.3, size=(count - uniform_count, 2)), -1.0, 1.0
    )
    return np.concatenate([uniform, jittered], axis=0).astype(np.float32)


def collect_round(
    checkpoint: Path | None, args: argparse.Namespace, round_index: int, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run episodes under the current model and label the states it visits."""
    layout = StateLayout(num_objects=args.num_objects)
    cem = CEMConfig(
        horizon=args.horizon, num_samples=args.num_samples, cem_iters=args.cem_iters,
        elite_frac=args.elite_frac, terminal_weight=args.terminal_weight,
        proximity_weight=args.proximity_weight,
    )
    rng = np.random.default_rng(args.seed * 1000 + round_index)
    generator = torch.Generator(device=device).manual_seed(args.seed + round_index)

    forward = None
    if checkpoint is not None:
        model, config = load_sparse_model(checkpoint, device)
        forward = SparseForward(model, config, args.num_objects, layout, device)

    states, actions, next_states, masks, deltas = [], [], [], [], []
    successes, distances = 0, []

    for episode in range(args.episodes_per_round):
        env = make_env(args.num_objects, args.base_seed + episode, args.max_steps)
        obs = env.reset()
        goal = torch.tensor(env.config.goal_xy, dtype=torch.float32, device=device)
        mean = torch.zeros(cem.horizon, 2, device=device)
        target_object = env.config.target_object
        final_distance = float(
            np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy))
        )

        for _ in range(args.max_steps):
            state_vec = flatten_state(obs).astype(np.float32)
            if forward is None:
                # Round 0 has no model yet: explore uniformly so the first labelled batch
                # is not biased by whatever the seed dataset happens to contain.
                planned = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                state = torch.from_numpy(state_vec).to(device)
                evaluate = model_evaluator(
                    forward, state, goal, layout, target_object,
                    env.config.action_scale, env.config.pusher_bounds, cem,
                )
                best, _ = cem_plan(evaluate, cem.horizon, device, cem, mean, generator)
                planned = best[0].cpu().numpy()
                mean = torch.cat([best[1:], torch.zeros(1, 2, device=device)], dim=0)

            candidates = sample_candidate_actions(planned, args.actions_per_state, rng)
            for row in label_with_simulator(env, state_vec, candidates):
                states.append(row[0]); actions.append(row[1]); next_states.append(row[2])
                masks.append(row[3]); deltas.append(row[4])

            obs, _, terminated, info = env.step(planned)
            final_distance = float(info["target_distance"])
            if info["success"]:
                successes += 1
                break
            if terminated:
                break
        distances.append(final_distance)

    data = {
        "s_t": np.stack(states).astype(np.float32),
        "a_t": np.stack(actions).astype(np.float32),
        "s_t1": np.stack(next_states).astype(np.float32),
        "object_change_mask": np.stack(masks).astype(np.float32),
        "object_delta": np.stack(deltas).astype(np.float32),
    }
    stats = {
        "episodes": args.episodes_per_round,
        "success_rate": successes / max(args.episodes_per_round, 1),
        "mean_final_distance": float(np.mean(distances)),
        "transitions_labelled": int(data["s_t"].shape[0]),
        "changed_object_fraction": float(data["object_change_mask"].mean()),
    }
    return data, stats


def write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["done"] = np.zeros(data["s_t"].shape[0], dtype=bool)
    if payload["done"].size:
        payload["done"][-1] = True
    np.savez_compressed(path, **payload)


def train(run_name: str, train_path: Path, args: argparse.Namespace) -> Path:
    command = [
        sys.executable, "-m", "experiments.train_sparse_model",
        "--train", str(train_path), "--val", str(args.val_data),
        "--run-name", run_name,
        "--delta-head", args.delta_head,
        "--mixture-components", str(args.mixture_components),
        "--feature-mode", args.feature_mode,
        "--epochs", str(args.epochs),
        "--sparsity-weight", str(args.sparsity_weight),
        "--auto-balance-bce",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    subprocess.run(command, check=True)
    return Path("models/checkpoints") / f"{run_name}.pt"


def load_seed_data(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "s_t": data["s_t"].astype(np.float32),
        "a_t": data["a_t"].astype(np.float32),
        "s_t1": data["s_t1"].astype(np.float32),
        "object_change_mask": data["object_change_mask"].astype(np.float32),
        "object_delta": data["object_delta"].astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config({
        "task": "dagger_planning", "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "actions_per_state": args.actions_per_state,
        "delta_head": args.delta_head, "mixture_components": args.mixture_components,
        "feature_mode": args.feature_mode, "seed_data": str(args.seed_data),
    })

    if not args.seed_data.exists():
        raise FileNotFoundError(
            f"{args.seed_data} missing -- generate the mixed planning dataset first "
            "(see the Phase-5 reproduce block in RESULTS.md)."
        )
    aggregate = load_seed_data(args.seed_data)
    print(f"[dagger] seed data: {aggregate['s_t'].shape[0]} transitions", flush=True)

    # Round 0 trains on the seed data alone: the baseline the loop has to beat.
    base_path = output_dir / "data" / "round0.npz"
    write_npz(base_path, aggregate)
    checkpoint = train(f"dagger_r0_s{args.seed}", base_path, args)

    history = []
    for round_index in range(1, args.rounds + 1):
        started = time.perf_counter()
        fresh, stats = collect_round(checkpoint, args, round_index, device)
        aggregate = {
            key: np.concatenate([aggregate[key], fresh[key]], axis=0) for key in aggregate
        }
        path = output_dir / "data" / f"round{round_index}.npz"
        write_npz(path, aggregate)
        checkpoint = train(f"dagger_r{round_index}_s{args.seed}", path, args)

        entry = {
            "round": round_index,
            "success_rate_during_collection": stats["success_rate"],
            "mean_final_distance": stats["mean_final_distance"],
            "transitions_added": stats["transitions_labelled"],
            "aggregate_transitions": int(aggregate["s_t"].shape[0]),
            "changed_object_fraction_of_new_data": stats["changed_object_fraction"],
            "checkpoint": str(checkpoint),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(entry)
        print(f"[dagger] round {round_index}: success during collection "
              f"{stats['success_rate']:.2f}, +{stats['transitions_labelled']} transitions "
              f"(total {entry['aggregate_transitions']})", flush=True)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    collection_seeds = [args.base_seed + i for i in range(args.episodes_per_round)]
    summary = {
        "rounds": history,
        "final_checkpoint": str(checkpoint),
        # Success measured during collection is optimistic as a headline: DAgger has trained
        # on transitions gathered from these very object configurations. The headline number
        # must come from planning_mpc on env seeds DISJOINT from this range -- note that
        # planning_mpc's --base-seed also defaults to 1000, so the default would silently
        # evaluate on the training configurations.
        "collection_env_seeds": [min(collection_seeds), max(collection_seeds)],
        "note": (
            "headline = experiments.planning_mpc --base-seed OUTSIDE "
            f"[{min(collection_seeds)}, {max(collection_seeds)}] with --sparse-checkpoint "
            f"{checkpoint}"
        ),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
