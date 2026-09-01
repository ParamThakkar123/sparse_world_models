"""Phase 5 -- downstream planning with the learned world models.

This is the *decision-making* test of the world models. Phases 2-4 score the
models as one-step (and multi-step rollout) *predictors*; here we close the loop
around a controller: we use each model as the **forward simulator inside a
sampling-based planner** (random-shooting / CEM Model-Predictive Control) and ask
whether it can actually *push the target object to the goal*.

The scientific question is the paper's core claim carried into control: does the
sparse/residual model -- which the rollout study showed compounds far less error
over a horizon -- make a *better forward model for planning* than the dense
monolith? We answer it by running the identical MPC loop with each model as the
only swapped component and comparing task success.

Planner
-------
Receding-horizon MPC. At every real environment step we optimise a length-``H``
action sequence with the Cross-Entropy Method (``cem_iters`` refit iterations over
``num_samples`` Gaussian samples, keeping the top ``elite_frac``), roll each
candidate forward *through the world model*, score it, execute only the first
action of the best sequence in the real env, then re-plan (warm-started from the
shifted previous solution). This is the standard MPC recipe; the model is the one
piece that differs between conditions.

Imagined-state reconstruction (identical for every model)
--------------------------------------------------------
The models predict only planar object poses ``(x, y, theta)``. To feed a predicted
state back in we must rebuild the full state vector. Two components are *not*
model outputs and are handled by the same closed form for sparse and dense so the
comparison stays fair:

  * **pusher_xy** -- deterministic. The pusher is a position-controlled actuator:
    ``env.step`` moves its command to ``clip(pusher + action * action_scale,
    bounds)`` and the high-gain actuator reaches it within a control step. We
    replicate exactly that kinematic update, so the planner knows where its own
    end-effector goes (this is controller state, not unmodelled dynamics).
  * **object velocities** -- held at their plan-time value across the imagined
    horizon (we have no velocity predictor). At planning time objects are usually
    at rest, so this is a mild approximation; crucially it is applied identically
    to both models.
  * **goal** -- static, copied.

  sparse : next_pose = pose + gate . delta   (masked residual)
  dense  : next_pose = f(state, action)      (absolute pose)

Cost
----
Lower is better. Primary term is the target object's distance to the goal,
averaged over the imagined horizon (dense progress signal) plus a terminal term.
An optional small pusher->target proximity term (``proximity_weight``) makes
contact discoverable by the sampler; it encodes *cost design*, not privileged
dynamics knowledge, and is applied identically to every model condition.

Baselines / anchors
-------------------
  * ``sparse`` / ``dense`` -- MPC with the respective world model (the comparison).
  * ``scripted`` -- the hand-written push controller used to generate data; a
    model-free *upper reference* answering "is the task solvable at all".
  * ``random`` -- uniform actions; the *lower reference*. If MPC does not beat this
    its model is not providing usable dynamics.

Metrics per condition (mean over episodes): success rate, steps-to-success on
solved episodes, final target->goal distance, and planning wall-clock per real
step (the efficiency angle -- the sparse model has far fewer parameters but
per-object heads, so we report measured planning time, not just FLOPs).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from experiments import ExperimentLogger
from experiments.compare_phase4_models import load_dense_model, load_sparse_model
from experiments.generate_transitions import flatten_state
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout
from models.envs import TabletopPushConfig, TabletopPushEnv
from models.policies import ScriptedPushPolicy

# "latent" is the learned-model control baseline (TD-MPC2/Dreamer-style latent dynamics);
# without it the only comparators are scripted and random, neither of which is a learned
# model, so the table could not separate object-centric structure from learning at all.
# The five published object-centric dynamics models (models/literature_baselines.py), added
# as planning conditions so the control result is not "our monolith and our latent model
# fail" but "no published world model in this class plans at all". A negative that covers
# GNS, C-SWM, SlotFormer, PETS and NPS is a substantially more informative statement about
# the field than one covering only baselines we wrote.
LITERATURE_CONDITIONS = ("gns", "cswm", "slotformer", "pets", "nps")
CONDITIONS = ("sparse", "dense", "latent", *LITERATURE_CONDITIONS, "oracle", "scripted", "random")
# Conditions that plan through a learned forward model (as opposed to the true simulator
# or a model-free policy). Kept as one list so adding a model does not require touching
# the dispatch in two places.
MODEL_CONDITIONS = ("sparse", "dense", "latent", *LITERATURE_CONDITIONS)


# --------------------------------------------------------------------------- #
# World-model forward wrappers: (states, actions) -> predicted next object pose
# --------------------------------------------------------------------------- #
class ModelForward:
    """Batched one-step object-pose predictor shared by the imagined rollout.

    ``__call__`` takes ``states (B, state_dim)`` and ``actions (B, 2)`` (both
    torch, on ``device``) and returns predicted next object poses
    ``(B, num_objects, 3)``. Subclasses wrap the sparse / dense checkpoints.
    """

    def __init__(self, num_objects: int, layout: StateLayout, device: torch.device):
        self.num_objects = num_objects
        self.layout = layout
        self.device = device

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError(f"{type(self).__name__} must implement __call__({states.shape}, {actions.shape}).")


class SparseForward(ModelForward):
    def __init__(self, model, config: dict, num_objects: int, layout: StateLayout, device: torch.device):
        super().__init__(num_objects, layout, device)
        self.model = model
        self.estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
        self.temperature = float(config["temperature"])
        self.feature_mode = str(config.get("feature_mode", "global"))

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = build_object_features_by_mode(states, actions, self.feature_mode)
        out = self.model(features, estimator=self.estimator, temperature=self.temperature, hard=True)
        current_pose = states[:, self.layout.object_pose_slice].reshape(-1, self.num_objects, POSE_DIM)
        return current_pose + out.masked_delta


class LatentForward(ModelForward):
    """Plan through the TD-MPC2/Dreamer-style latent dynamics baseline (W5).

    Re-encodes the reconstructed state each step rather than rolling purely in latent space.
    That matches how the sparse and dense conditions are driven -- each consumes the state
    the harness reconstructs -- so the comparison isolates the *representation* rather than
    rewarding whichever model gets a different rollout protocol.
    """

    def __init__(self, model, num_objects: int, layout: StateLayout, device: torch.device):
        super().__init__(num_objects, layout, device)
        self.model = model

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model(states, actions).reshape(-1, self.num_objects, POSE_DIM)


class DenseForward(ModelForward):
    def __init__(self, model, num_objects: int, layout: StateLayout, device: torch.device):
        super().__init__(num_objects, layout, device)
        self.model = model

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred = self.model(states, actions)
        return pred.reshape(-1, self.num_objects, POSE_DIM)


class LiteratureForward(ModelForward):
    """Plan through one of the published baselines (GNS / C-SWM / SlotFormer / PETS / NPS).

    All five share the ``(object_features, current_pose) -> next_pose`` signature, so one
    wrapper covers the set. They are driven exactly as the sparse condition is -- same
    featurisation entry point, same reconstructed state each step -- so the comparison
    isolates the model and not the rollout protocol.

    PETS is planned through its ensemble MEAN rather than by trajectory sampling. That is a
    deliberate simplification and it favours PETS on this task: TS-inf would inject
    per-particle model noise into an already-inaccurate rollout. If the mean fails, sampling
    around it does not rescue the condition, so the negative is not weakened by the choice.
    """

    def __init__(self, model, feature_mode: str, num_objects: int,
                 layout: StateLayout, device: torch.device):
        super().__init__(num_objects, layout, device)
        self.model = model
        self.feature_mode = feature_mode

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = build_object_features_by_mode(states, actions, self.feature_mode)
        current_pose = states[:, self.layout.object_pose_slice].reshape(
            -1, self.num_objects, POSE_DIM
        )
        return self.model(features, current_pose)


# --------------------------------------------------------------------------- #
# Imagined-state reconstruction and rollout
# --------------------------------------------------------------------------- #
def advance_pusher(pusher_xy: torch.Tensor, actions: torch.Tensor, action_scale: float, bounds: tuple[float, float]) -> torch.Tensor:
    """Deterministic position-controlled pusher update, matching ``env.step``."""
    clipped_action = torch.clamp(actions, -1.0, 1.0)
    target = pusher_xy + clipped_action * action_scale
    return torch.clamp(target, bounds[0], bounds[1])


def reconstruct_state(state: torch.Tensor, next_pose: torch.Tensor, next_pusher: torch.Tensor, layout: StateLayout) -> torch.Tensor:
    """Rebuild the full state from predicted poses + advanced pusher.

    Velocities and goal are copied from ``state`` (held constant); the pusher and
    object-pose slices are overwritten. ``next_pose`` is ``(B, N, 3)``.
    """
    new_state = state.clone()
    new_state[:, 0 : layout.object_pose_slice.start] = next_pusher
    new_state[:, layout.object_pose_slice] = next_pose.reshape(next_pose.shape[0], -1)
    return new_state


def imagined_target_trajectory(
    forward: ModelForward,
    init_state: torch.Tensor,
    action_seqs: torch.Tensor,
    layout: StateLayout,
    target_object: int,
    action_scale: float,
    pusher_bounds: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll ``action_seqs`` (B, H, 2) through the model from a single init state.

    Returns ``target_xy (B, H, 2)`` (imagined target-object planar position at each
    horizon) and ``pusher_xy (B, H, 2)`` (imagined end-effector position), both used
    by the cost. All ``B`` candidates share ``init_state``.
    """
    batch_size, horizon, _ = action_seqs.shape
    state = init_state.unsqueeze(0).expand(batch_size, -1).clone()
    pusher_start = layout.object_pose_slice.start
    target_xy = torch.empty(batch_size, horizon, 2, device=state.device, dtype=state.dtype)
    pusher_traj = torch.empty(batch_size, horizon, 2, device=state.device, dtype=state.dtype)

    with torch.no_grad():
        for h in range(horizon):
            action = action_seqs[:, h, :]
            next_pusher = advance_pusher(state[:, 0:pusher_start], action, action_scale, pusher_bounds)
            next_pose = forward(state, action)
            state = reconstruct_state(state, next_pose, next_pusher, layout)
            target_xy[:, h, :] = next_pose[:, target_object, :2]
            pusher_traj[:, h, :] = next_pusher
    return target_xy, pusher_traj


# --------------------------------------------------------------------------- #
# Cost + CEM optimiser
# --------------------------------------------------------------------------- #
@dataclass
class CEMConfig:
    horizon: int = 15
    num_samples: int = 256
    cem_iters: int = 3
    elite_frac: float = 0.1
    init_std: float = 0.6
    min_std: float = 0.05
    terminal_weight: float = 3.0
    proximity_weight: float = 0.3


def sequence_cost(
    target_xy: torch.Tensor,
    pusher_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    config: CEMConfig,
) -> torch.Tensor:
    """Per-candidate cost (B,) -- lower is better.

    Mean + weighted-terminal target->goal distance, plus a small pusher->target
    proximity shaping term that makes contact discoverable.
    """
    goal = goal_xy.reshape(1, 1, 2)
    target_goal_dist = torch.linalg.norm(target_xy - goal, dim=-1)  # (B, H)
    mean_dist = target_goal_dist.mean(dim=1)
    terminal_dist = target_goal_dist[:, -1]
    proximity = torch.linalg.norm(pusher_xy - target_xy, dim=-1).mean(dim=1)
    return mean_dist + config.terminal_weight * terminal_dist + config.proximity_weight * proximity


def cem_plan(
    evaluate_sequences,
    horizon: int,
    device: torch.device,
    config: CEMConfig,
    mean_init: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """Generic CEM optimiser. Returns ``(best_sequence (H,2), best_cost)``.

    ``evaluate_sequences(samples (B,H,2)) -> costs (B,)`` scores a batch of candidate
    action sequences; it encapsulates *how* the world is rolled forward (learned
    model or true simulator), so the optimiser itself is identical across
    conditions. ``mean_init`` (H, 2) warm-starts the mean (shifted prior solution).
    """
    num_elites = max(1, int(round(config.num_samples * config.elite_frac)))
    mean = mean_init.clone()
    std = torch.full((horizon, 2), config.init_std, device=device)

    best_sequence = mean.clone()
    best_cost = float("inf")

    for _ in range(config.cem_iters):
        noise = torch.randn(config.num_samples, horizon, 2, generator=generator, device=device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        samples = torch.clamp(samples, -1.0, 1.0)

        costs = evaluate_sequences(samples)

        elite_idx = torch.topk(costs, num_elites, largest=False).indices
        elites = samples[elite_idx]

        iter_best_idx = int(elite_idx[0].item())
        if float(costs[iter_best_idx].item()) < best_cost:
            best_cost = float(costs[iter_best_idx].item())
            best_sequence = samples[iter_best_idx].clone()

        mean = elites.mean(dim=0)
        std = torch.clamp(elites.std(dim=0), min=config.min_std)

    return best_sequence, best_cost


def model_evaluator(
    forward: ModelForward,
    init_state: torch.Tensor,
    goal_xy: torch.Tensor,
    layout: StateLayout,
    target_object: int,
    action_scale: float,
    pusher_bounds: tuple[float, float],
    config: CEMConfig,
):
    """Return an ``evaluate_sequences`` closure that rolls candidates through ``forward``."""

    def evaluate(samples: torch.Tensor) -> torch.Tensor:
        target_xy, pusher_xy = imagined_target_trajectory(
            forward, init_state, samples, layout, target_object, action_scale, pusher_bounds
        )
        return sequence_cost(target_xy, pusher_xy, goal_xy, config)

    return evaluate


def oracle_evaluator(
    env: TabletopPushEnv,
    snapshot: dict,
    goal_xy: torch.Tensor,
    target_object: int,
    config: CEMConfig,
):
    """Return an ``evaluate_sequences`` closure that rolls candidates through the *true*
    simulator from ``snapshot`` (restoring after each candidate). Exact dynamics, at the
    cost of ``num_samples * horizon`` env steps per CEM iteration."""

    def evaluate(samples: torch.Tensor) -> torch.Tensor:
        samples_np = samples.cpu().numpy()
        num_samples, horizon, _ = samples_np.shape
        target_xy = np.empty((num_samples, horizon, 2), dtype=np.float32)
        pusher_xy = np.empty((num_samples, horizon, 2), dtype=np.float32)
        for i in range(num_samples):
            env.restore(snapshot)
            for h in range(horizon):
                obs, _, terminated, _ = env.step(samples_np[i, h])
                target_xy[i, h] = obs["object_poses"][target_object, :2]
                pusher_xy[i, h] = obs["pusher_xy"]
                if terminated:
                    # Freeze the (successful/terminal) pose for the remainder of the horizon.
                    target_xy[i, h + 1 :] = target_xy[i, h]
                    pusher_xy[i, h + 1 :] = pusher_xy[i, h]
                    break
        env.restore(snapshot)
        return sequence_cost(
            torch.from_numpy(target_xy), torch.from_numpy(pusher_xy), goal_xy, config
        )

    return evaluate


# --------------------------------------------------------------------------- #
# Episode runners
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_distance: float
    plan_time_ms_per_step: float


@dataclass
class ConditionSummary:
    condition: str
    num_episodes: int
    success_rate: float
    mean_steps_on_success: float
    mean_final_distance: float
    mean_plan_time_ms_per_step: float
    per_episode: list[EpisodeResult] = field(default_factory=list)


def run_mpc_episode(
    env: TabletopPushEnv,
    forward: ModelForward,
    layout: StateLayout,
    config: CEMConfig,
    max_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> EpisodeResult:
    """One receding-horizon MPC episode using ``forward`` as the model."""
    obs = env.reset()
    target_object = env.config.target_object
    goal_xy = torch.tensor(env.config.goal_xy, dtype=torch.float32, device=device)
    mean = torch.zeros(config.horizon, 2, device=device)

    total_plan_time = 0.0
    num_plans = 0
    success = False
    steps = 0
    final_distance = float(np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy)))

    for step in range(max_steps):
        state = torch.from_numpy(flatten_state(obs).astype(np.float32)).to(device)
        evaluate = model_evaluator(
            forward, state, goal_xy, layout, target_object,
            env.config.action_scale, env.config.pusher_bounds, config,
        )

        start = time.perf_counter()
        best_sequence, _ = cem_plan(evaluate, config.horizon, device, config, mean, generator)
        total_plan_time += time.perf_counter() - start
        num_plans += 1

        action = best_sequence[0].cpu().numpy()
        obs, _, terminated, info = env.step(action)
        steps = step + 1
        final_distance = float(info["target_distance"])

        # Warm-start next plan: shift the solution forward one step, pad with zero.
        mean = torch.cat([best_sequence[1:], torch.zeros(1, 2, device=device)], dim=0)

        if info["success"]:
            success = True
            break
        if terminated:
            break

    plan_time_ms = (total_plan_time * 1000.0 / num_plans) if num_plans else 0.0
    return EpisodeResult(success=success, steps=steps, final_distance=final_distance, plan_time_ms_per_step=plan_time_ms)


def run_oracle_mpc_episode(
    env: TabletopPushEnv,
    config: CEMConfig,
    max_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> EpisodeResult:
    """MPC episode using the *true simulator* as the forward model (planner-soundness
    upper bound). Uses a second env instance for candidate rollouts so the real env is
    never perturbed."""
    obs = env.reset()
    target_object = env.config.target_object
    goal_xy = torch.tensor(env.config.goal_xy, dtype=torch.float32, device=device)
    mean = torch.zeros(config.horizon, 2, device=device)

    # Dedicated rollout env; kept in lock-step with the real env via snapshots.
    sim = TabletopPushEnv(env.config)
    sim.reset()

    total_plan_time = 0.0
    num_plans = 0
    success = False
    steps = 0
    final_distance = float(np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy)))

    for step in range(max_steps):
        sim.restore(env.snapshot())
        evaluate = oracle_evaluator(sim, env.snapshot(), goal_xy, target_object, config)

        start = time.perf_counter()
        best_sequence, _ = cem_plan(evaluate, config.horizon, device, config, mean, generator)
        total_plan_time += time.perf_counter() - start
        num_plans += 1

        action = best_sequence[0].cpu().numpy()
        obs, _, terminated, info = env.step(action)
        steps = step + 1
        final_distance = float(info["target_distance"])
        mean = torch.cat([best_sequence[1:], torch.zeros(1, 2, device=device)], dim=0)
        if info["success"]:
            success = True
            break
        if terminated:
            break

    plan_time_ms = (total_plan_time * 1000.0 / num_plans) if num_plans else 0.0
    return EpisodeResult(success=success, steps=steps, final_distance=final_distance, plan_time_ms_per_step=plan_time_ms)


def run_policy_episode(env: TabletopPushEnv, policy, max_steps: int) -> EpisodeResult:
    """One episode under a model-free policy (scripted / random anchors)."""
    obs = env.reset()
    target_object = env.config.target_object
    success = False
    steps = 0
    final_distance = float(np.linalg.norm(obs["object_poses"][target_object, :2] - np.asarray(env.config.goal_xy)))

    start = time.perf_counter()
    for step in range(max_steps):
        action = policy.act(obs)
        obs, _, terminated, info = env.step(action)
        steps = step + 1
        final_distance = float(info["target_distance"])
        if info["success"]:
            success = True
            break
        if terminated:
            break
    plan_time_ms = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
    return EpisodeResult(success=success, steps=steps, final_distance=final_distance, plan_time_ms_per_step=plan_time_ms)


def summarise(condition: str, results: list[EpisodeResult]) -> ConditionSummary:
    successes = [r for r in results if r.success]
    return ConditionSummary(
        condition=condition,
        num_episodes=len(results),
        success_rate=len(successes) / max(len(results), 1),
        mean_steps_on_success=float(np.mean([r.steps for r in successes])) if successes else float("nan"),
        mean_final_distance=float(np.mean([r.final_distance for r in results])) if results else float("nan"),
        mean_plan_time_ms_per_step=float(np.mean([r.plan_time_ms_per_step for r in results])) if results else float("nan"),
        per_episode=results,
    )


def build_forward(condition: str, checkpoints: dict[str, Path], num_objects: int, layout: StateLayout, device: torch.device) -> ModelForward:
    if condition == "sparse":
        model, config = load_sparse_model(checkpoints["sparse"], device)
        ckpt_num_objects = int(config["num_objects"])  # type: ignore[call-overload]
        if ckpt_num_objects != num_objects:
            raise ValueError(f"Sparse checkpoint is for {ckpt_num_objects} objects but env has {num_objects}.")
        return SparseForward(model, config, num_objects, layout, device)
    if condition == "dense":
        model, _ = load_dense_model(checkpoints["dense"], device)
        return DenseForward(model, num_objects, layout, device)
    if condition == "latent":
        from models.latent_dynamics import LatentDynamicsModel

        checkpoint = torch.load(checkpoints["latent"], map_location=device)
        config = checkpoint["config"]
        if int(config["num_objects"]) != num_objects:
            raise ValueError(
                f"Latent checkpoint is for {config['num_objects']} objects but env has {num_objects}."
            )
        model = LatentDynamicsModel(
            state_dim=config["state_dim"], action_dim=config["action_dim"],
            num_objects=config["num_objects"], latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return LatentForward(model, num_objects, layout, device)
    if condition in LITERATURE_CONDITIONS:
        from experiments.literature_baselines import load_baseline

        model, config = load_baseline(checkpoints[condition], device)
        if int(config["num_objects"]) != num_objects:
            raise ValueError(
                f"{condition} checkpoint is for {config['num_objects']} objects "
                f"but env has {num_objects}."
            )
        return LiteratureForward(
            model, str(config["feature_mode"]), num_objects, layout, device
        )
    raise ValueError(f"build_forward only handles model conditions, not '{condition}'.")


def make_env(args: argparse.Namespace, seed: int) -> TabletopPushEnv:
    config = TabletopPushConfig(num_objects=args.num_objects, max_steps=args.max_steps, seed=seed)
    if args.object_bound is not None:
        config.object_bounds = (-abs(args.object_bound), abs(args.object_bound))
    if args.min_object_separation is not None:
        config.min_object_separation = args.min_object_separation
    return TabletopPushEnv(config)


def run_condition(
    condition: str,
    args: argparse.Namespace,
    checkpoints: dict[str, Path],
    layout: StateLayout,
    cem_config: CEMConfig,
    device: torch.device,
) -> ConditionSummary:
    """Run ``num_episodes`` for one condition. Episode ``i`` uses env seed
    ``base_seed + i`` so every condition sees the identical object configurations."""
    results: list[EpisodeResult] = []
    forward = None
    if condition in MODEL_CONDITIONS:
        forward = build_forward(condition, checkpoints, args.num_objects, layout, device)

    for i in range(args.num_episodes):
        seed = args.base_seed + i
        env = make_env(args, seed)
        if condition in MODEL_CONDITIONS:
            assert forward is not None  # set above for model conditions
            # Deterministic per-episode action sampler, shared seed offset for fairness.
            generator = torch.Generator(device=device)
            generator.manual_seed(args.plan_seed + i)
            result = run_mpc_episode(env, forward, layout, cem_config, args.max_steps, generator, device)
        elif condition == "oracle":
            generator = torch.Generator(device=device)
            generator.manual_seed(args.plan_seed + i)
            result = run_oracle_mpc_episode(env, cem_config, args.max_steps, generator, device)
        elif condition == "scripted":
            result = run_policy_episode(env, ScriptedPushPolicy(target_object=env.config.target_object), args.max_steps)
        elif condition == "random":
            from models.policies import RandomPolicy

            result = run_policy_episode(env, RandomPolicy(seed=args.plan_seed + i), args.max_steps)
        else:
            raise ValueError(f"Unknown condition '{condition}'.")
        results.append(result)
        print(
            f"[{condition}] ep {i + 1}/{args.num_episodes} seed={seed} "
            f"success={result.success} steps={result.steps} final_dist={result.final_distance:.4f}"
        )
    return summarise(condition, results)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_results_table(summaries: list[ConditionSummary], md_path: Path, csv_path: Path, success_radius: float) -> None:
    columns = ["condition", "success_rate", "mean_steps_on_success", "mean_final_distance", "mean_plan_time_ms_per_step", "num_episodes"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [f"<!-- success radius = {success_radius} m -->", header, separator]
    for s in summaries:
        lines.append(
            f"| {s.condition} | {s.success_rate:.3f} | {s.mean_steps_on_success:.2f} | "
            f"{s.mean_final_distance:.4f} | {s.mean_plan_time_ms_per_step:.2f} | {s.num_episodes} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for s in summaries:
            writer.writerow([
                s.condition, f"{s.success_rate:.6f}", f"{s.mean_steps_on_success:.6f}",
                f"{s.mean_final_distance:.6f}", f"{s.mean_plan_time_ms_per_step:.6f}", s.num_episodes,
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 -- MPC/CEM planning with the learned world models.")
    parser.add_argument("--sparse-checkpoint", type=Path, default=Path("models/checkpoints/sparse_3obj_s0.pt"))
    parser.add_argument("--dense-checkpoint", type=Path, default=Path("models/checkpoints/dense_3obj_s0.pt"))
    parser.add_argument("--latent-checkpoint", type=Path,
                        default=Path("models/checkpoints/latent_plan_3obj_s0.pt"),
                        help="TD-MPC2/Dreamer-style latent dynamics baseline (W5).")
    parser.add_argument(
        "--literature-checkpoint-dir", type=Path, default=Path("models/checkpoints"),
        help="Where literature_baselines.py --checkpoint-dir wrote the published baselines.")
    parser.add_argument("--literature-tag", type=str, default="litplan",
                        help="Checkpoint tag the published baselines were saved under.")
    parser.add_argument("--literature-seed", type=int, default=0,
                        help="Which seed's published-baseline checkpoints to plan through.")
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--object-bound", type=float, default=None)
    parser.add_argument("--min-object-separation", type=float, default=None)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80, help="Max real env steps per episode.")
    parser.add_argument("--base-seed", type=int, default=1000, help="Env seed for episode 0 (shared across conditions).")
    parser.add_argument("--plan-seed", type=int, default=0, help="Base seed for CEM/random action sampling.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="Subset of conditions to run.",
    )
    # CEM hyperparameters
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=3.0)
    parser.add_argument("--proximity-weight", type=float, default=0.3)
    parser.add_argument("--run-name", type=str, default="planning_mpc")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    layout = StateLayout(num_objects=args.num_objects)
    cem_config = CEMConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        cem_iters=args.cem_iters,
        elite_frac=args.elite_frac,
        terminal_weight=args.terminal_weight,
        proximity_weight=args.proximity_weight,
    )
    checkpoints = {
        "sparse": args.sparse_checkpoint,
        "dense": args.dense_checkpoint,
        "latent": args.latent_checkpoint,
    }
    # The published baselines follow one naming convention, written by
    # literature_baselines.py --checkpoint-dir, so they need one flag rather than five.
    for condition in LITERATURE_CONDITIONS:
        checkpoints[condition] = (
            args.literature_checkpoint_dir
            / f"{args.literature_tag}_{condition}_{args.num_objects}obj_s{args.literature_seed}.pt"
        )

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config(
        {
            "task": "planning_mpc",
            "num_objects": args.num_objects,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "base_seed": args.base_seed,
            "plan_seed": args.plan_seed,
            "conditions": args.conditions,
            "object_bound": args.object_bound,
            "min_object_separation": args.min_object_separation,
            "cem": vars(cem_config),
            "sparse_checkpoint": str(args.sparse_checkpoint),
            "dense_checkpoint": str(args.dense_checkpoint),
        }
    )

    summaries: list[ConditionSummary] = []
    for condition in args.conditions:
        summary = run_condition(condition, args, checkpoints, layout, cem_config, device)
        summaries.append(summary)
        logger.log_metrics(
            len(summaries),
            **{
                f"{condition}_success_rate": summary.success_rate,
                f"{condition}_mean_final_distance": summary.mean_final_distance,
                f"{condition}_plan_time_ms": summary.mean_plan_time_ms_per_step,
            },
        )

    success_radius = 0.05  # env success threshold on target->goal distance
    md_path = output_dir / "planning_results.md"
    csv_path = output_dir / "planning_results.csv"
    write_results_table(summaries, md_path, csv_path, success_radius)

    summary_json = {
        "results_md": str(md_path),
        "results_csv": str(csv_path),
        "conditions": {
            s.condition: {
                "success_rate": s.success_rate,
                "mean_steps_on_success": s.mean_steps_on_success,
                "mean_final_distance": s.mean_final_distance,
                "mean_plan_time_ms_per_step": s.mean_plan_time_ms_per_step,
                "num_episodes": s.num_episodes,
            }
            for s in summaries
        },
    }
    if "sparse" in summary_json["conditions"] and "dense" in summary_json["conditions"]:
        summary_json["sparse_beats_dense_success"] = bool(
            summaries[args.conditions.index("sparse")].success_rate
            > summaries[args.conditions.index("dense")].success_rate
        )
    logger.log_summary(summary_json)
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
