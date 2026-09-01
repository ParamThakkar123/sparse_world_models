from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments import ExperimentLogger
from models import (
    DELTA_HEAD_TYPES,
    POSE_DIM,
    SparseResidualHead,
    StateLayout,
    TransitionDataset,
    infer_num_objects_from_state_dim,
    reshape_object_pose,
    reshape_object_velocity,
)
from models.sequence_datasets import TransitionSequenceDataset
from models.sparse_residual import sparse_residual_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse residual model with gate metrics.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="sparse_model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--delta-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-num-layers", type=int, default=2)
    parser.add_argument("--delta-num-layers", type=int, default=2)
    parser.add_argument(
        "--delta-head",
        type=str,
        default="mse",
        choices=list(DELTA_HEAD_TYPES),
        help=(
            "Delta parameterisation. 'mse' is the original point regressor; 'gaussian' and "
            "'mdn' are trained by NLL and predict a density (see models/delta_heads.py)."
        ),
    )
    parser.add_argument(
        "--mixture-components",
        type=int,
        default=5,
        help="Number of Gaussian components when --delta-head mdn.",
    )
    parser.add_argument(
        "--detach-delta-gate",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "Cut the gradient from the delta loss back into the gate. 'auto' (default) "
            "detaches only for probabilistic heads, whose large NLL magnitudes otherwise "
            "collapse change detection; 'never' reproduces the original behaviour."
        ),
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=1,
        help=(
            "Steps to unroll during training. 1 is the original one-step objective. "
            ">1 feeds the model its own predicted poses and needs contiguous windows, so "
            "point --train/--val at splits of the UNFILTERED dataset (the hard subset "
            "keeps too few consecutive steps -- see models/sequence_datasets.py)."
        ),
    )
    parser.add_argument(
        "--rollout-target",
        type=str,
        default="correcting",
        choices=["correcting", "recorded"],
        help=(
            "Delta target during unrolled training. 'correcting' (default) supervises "
            "true_next_pose - rolled_pose, teaching drift correction; 'recorded' uses the "
            "dataset's one-step delta. Use 'recorded' as the control for whether the "
            "MSE-vs-likelihood split under rollout is caused by the growing target."
        ),
    )
    parser.add_argument(
        "--rollout-stride",
        type=int,
        default=1,
        help="Stride between window start indices when --rollout-horizon > 1.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Global grad-norm clip (0 disables). Recommended ~1.0 for rollout training.",
    )
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=1e-2)
    parser.add_argument("--positive-class-weight", type=float, default=None)
    parser.add_argument("--auto-balance-bce", action="store_true")
    parser.add_argument(
        "--delta-supervision",
        type=str,
        default="predicted_probs",
        choices=["predicted_probs", "predicted_gates", "target"],
    )
    parser.add_argument("--estimator", type=str, default="gumbel_st")
    parser.add_argument("--feature-mode", type=str, default="global", choices=["global", "invariant", "contact"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def build_object_features(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    num_objects = infer_num_objects_from_state_dim(int(state.shape[-1]))
    layout = StateLayout(num_objects=num_objects)
    pusher_xy = state[:, 0:2]
    object_pose = reshape_object_pose(state, num_objects=num_objects)
    object_velocity = reshape_object_velocity(state, num_objects=num_objects)
    goal_xy = state[:, layout.goal_slice]

    repeated_pusher = pusher_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_goal = goal_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_action = action.unsqueeze(1).expand(-1, num_objects, -1)
    global_pose = object_pose.reshape(state.shape[0], -1).unsqueeze(1).expand(-1, num_objects, -1)

    rel_goal = repeated_goal - object_pose[:, :, :2]
    rel_pusher = repeated_pusher - object_pose[:, :, :2]

    return torch.cat(
        [
            object_pose,
            object_velocity,
            rel_goal,
            rel_pusher,
            repeated_action,
            global_pose,
        ],
        dim=-1,
    )


def build_object_features_invariant(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Count-invariant per-object features (fixed width regardless of object count).

    Identical to :func:`build_object_features` except the count-dependent
    ``global_pose`` term (a flattened concatenation of every object's pose, whose
    width is ``num_objects * POSE_DIM``) is replaced by a permutation-invariant
    aggregate over the *other* objects: their mean relative position, the relative
    position of the nearest other object, and the distance to it. This keeps the
    per-object feature dimension constant (``20``) for any object count, so a head
    trained at one count can be applied verbatim at another. The aggregate still
    exposes local neighbourhood structure (what matters for push-through contacts)
    without hard-coding how many neighbours there are.
    """
    num_objects = infer_num_objects_from_state_dim(int(state.shape[-1]))
    layout = StateLayout(num_objects=num_objects)
    pusher_xy = state[:, 0:2]
    object_pose = reshape_object_pose(state, num_objects=num_objects)
    object_velocity = reshape_object_velocity(state, num_objects=num_objects)
    goal_xy = state[:, layout.goal_slice]

    repeated_pusher = pusher_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_goal = goal_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_action = action.unsqueeze(1).expand(-1, num_objects, -1)

    rel_goal = repeated_goal - object_pose[:, :, :2]
    rel_pusher = repeated_pusher - object_pose[:, :, :2]

    obj_xy = object_pose[:, :, :2]  # (B, N, 2)
    batch_size = obj_xy.shape[0]
    # pairwise[b, i, j] = xy_j - xy_i (self entries are zero on the diagonal).
    pairwise = obj_xy.unsqueeze(1) - obj_xy.unsqueeze(2)  # (B, N, N, 2)
    distances = torch.linalg.norm(pairwise, dim=-1)  # (B, N, N)
    if num_objects > 1:
        # Self diff is zero, so summing over all j and dividing by (N-1) is the mean
        # over the other objects.
        mean_rel = pairwise.sum(dim=2) / (num_objects - 1)  # (B, N, 2)
        eye = torch.eye(num_objects, dtype=torch.bool, device=state.device).unsqueeze(0)
        masked_distances = distances.masked_fill(eye, float("inf"))
        nearest_idx = masked_distances.argmin(dim=2)  # (B, N)
        gather_idx = nearest_idx.view(batch_size, num_objects, 1, 1).expand(batch_size, num_objects, 1, 2)
        nearest_rel = torch.gather(pairwise, 2, gather_idx).squeeze(2)  # (B, N, 2)
        nearest_dist = torch.gather(masked_distances, 2, nearest_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
    else:
        mean_rel = torch.zeros(batch_size, num_objects, 2, dtype=obj_xy.dtype, device=state.device)
        nearest_rel = torch.zeros_like(mean_rel)
        nearest_dist = torch.zeros(batch_size, num_objects, dtype=obj_xy.dtype, device=state.device)

    neighbor = torch.cat([mean_rel, nearest_rel, nearest_dist.unsqueeze(-1)], dim=-1)  # (B, N, 5)

    return torch.cat(
        [
            object_pose,
            object_velocity,
            rel_goal,
            rel_pusher,
            repeated_action,
            neighbor,
        ],
        dim=-1,
    )


# Kinematic constants of the pusher actuator, matching TabletopPushConfig defaults. The
# featurizer only receives (state, action), so it reproduces the env's deterministic pusher
# update here to expose *where the pusher will be after this action* — the signal a planner
# needs to reason about contact.
PUSHER_ACTION_SCALE = 0.04
PUSHER_BOUND = 0.26
# Pusher sphere radius (0.02) + object half-extent (0.025); distances below this imply contact.
CONTACT_RADIUS = 0.045


def build_object_features_contact(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Velocity-free, contact-aware per-object features for planning.

    Motivation: the ``global``/``invariant`` features key change prediction on object
    *velocity* and the scripted policy's contact context, which are out-of-distribution
    when a sampling-based planner queries teleported, near-rest states from arbitrary
    approach angles (the delta head then outputs a near-constant drift regardless of the
    pusher — see RESULTS.md "Downstream planning"). This mode instead exposes the
    *geometry of the impending contact*, which is well-defined for any state a planner
    visits, and drops velocity entirely (the imagined rollout holds velocity constant, so
    feeding it back is OOD).

    Per object, fixed width 19 for any object count:
      pose(3) + rel_goal(2) + rel_pusher(2) + action(2)
      + rel_pusher_next(2)      : object -> pusher position *after* this action
      + signed_contact_dist(1)  : ||pusher_next - obj_xy|| - CONTACT_RADIUS  (<0 => contact)
      + push_dir(2)             : unit vector pusher_next -> object (direction of a push)
      + neighbour(5)            : permutation-invariant mean/nearest relative position + dist
    """
    num_objects = infer_num_objects_from_state_dim(int(state.shape[-1]))
    layout = StateLayout(num_objects=num_objects)
    pusher_xy = state[:, 0:2]
    object_pose = state[:, layout.object_pose_slice].reshape(state.shape[0], num_objects, POSE_DIM)
    goal_xy = state[:, layout.goal_slice]
    obj_xy = object_pose[:, :, :2]  # (B, N, 2)

    # Deterministic post-action pusher position (matches env.step).
    clipped_action = torch.clamp(action, -1.0, 1.0)
    pusher_next = torch.clamp(pusher_xy + clipped_action * PUSHER_ACTION_SCALE, -PUSHER_BOUND, PUSHER_BOUND)

    repeated_goal = goal_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_pusher = pusher_xy.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_pusher_next = pusher_next.unsqueeze(1).expand(-1, num_objects, -1)
    repeated_action = clipped_action.unsqueeze(1).expand(-1, num_objects, -1)

    rel_goal = repeated_goal - obj_xy
    rel_pusher = repeated_pusher - obj_xy
    rel_pusher_next = repeated_pusher_next - obj_xy
    contact_dist = torch.linalg.norm(rel_pusher_next, dim=-1, keepdim=True)  # (B, N, 1)
    signed_contact_dist = contact_dist - CONTACT_RADIUS
    push_dir = -rel_pusher_next / contact_dist.clamp(min=1e-6)  # pusher_next -> object

    neighbor = _neighbour_aggregate(obj_xy, num_objects, state.device)  # (B, N, 5)

    return torch.cat(
        [
            object_pose,
            rel_goal,
            rel_pusher,
            repeated_action,
            rel_pusher_next,
            signed_contact_dist,
            push_dir,
            neighbor,
        ],
        dim=-1,
    )


def _neighbour_aggregate(obj_xy: torch.Tensor, num_objects: int, device) -> torch.Tensor:
    """Permutation-invariant neighbour features: mean relative position of the other
    objects, the relative position of the nearest one, and the distance to it (width 5)."""
    batch_size = obj_xy.shape[0]
    pairwise = obj_xy.unsqueeze(1) - obj_xy.unsqueeze(2)  # (B, N, N, 2): xy_j - xy_i
    distances = torch.linalg.norm(pairwise, dim=-1)  # (B, N, N)
    if num_objects > 1:
        mean_rel = pairwise.sum(dim=2) / (num_objects - 1)
        eye = torch.eye(num_objects, dtype=torch.bool, device=device).unsqueeze(0)
        masked_distances = distances.masked_fill(eye, float("inf"))
        nearest_idx = masked_distances.argmin(dim=2)
        gather_idx = nearest_idx.view(batch_size, num_objects, 1, 1).expand(batch_size, num_objects, 1, 2)
        nearest_rel = torch.gather(pairwise, 2, gather_idx).squeeze(2)
        nearest_dist = torch.gather(masked_distances, 2, nearest_idx.unsqueeze(-1)).squeeze(-1)
    else:
        mean_rel = torch.zeros(batch_size, num_objects, 2, dtype=obj_xy.dtype, device=device)
        nearest_rel = torch.zeros_like(mean_rel)
        nearest_dist = torch.zeros(batch_size, num_objects, dtype=obj_xy.dtype, device=device)
    return torch.cat([mean_rel, nearest_rel, nearest_dist.unsqueeze(-1)], dim=-1)


FEATURE_MODES = ("global", "invariant", "contact")


def build_object_features_by_mode(
    state: torch.Tensor, action: torch.Tensor, feature_mode: str = "global"
) -> torch.Tensor:
    """Dispatch to the requested per-object feature builder.

    ``global`` (default) preserves the original count-dependent features used by all
    existing checkpoints; ``invariant`` produces the fixed-width, count-invariant
    features needed for cross-object-count transfer.
    """
    if feature_mode == "global":
        return build_object_features(state, action)
    if feature_mode == "invariant":
        return build_object_features_invariant(state, action)
    if feature_mode == "contact":
        return build_object_features_contact(state, action)
    raise ValueError(f"Unsupported feature_mode '{feature_mode}'. Expected one of {FEATURE_MODES}.")


def compute_gate_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    pred = pred_mask.bool()
    target = target_mask.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)
    predicted_positive_rate = float(pred.float().mean().item())
    target_positive_rate = float(target.float().mean().item())
    all_changed_fraction = float(pred.all(dim=1).float().mean().item())
    all_unchanged_fraction = float((~pred).all(dim=1).float().mean().item())
    return {
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": f1,
        "gate_accuracy": accuracy,
        "gate_false_positive_rate": false_positive_rate,
        "gate_false_negative_rate": false_negative_rate,
        "gate_predicted_positive_rate": predicted_positive_rate,
        "gate_target_positive_rate": target_positive_rate,
        "gate_positive_rate_gap": predicted_positive_rate - target_positive_rate,
        "gate_all_changed_fraction": all_changed_fraction,
        "gate_all_unchanged_fraction": all_unchanged_fraction,
    }


def select_delta_supervision(
    supervision: str,
    batch: dict[str, torch.Tensor],
    gate_probs: torch.Tensor,
    gate_samples: torch.Tensor,
    detach: bool = False,
) -> torch.Tensor | None:
    """Pick the per-object weighting applied to the delta loss.

    ``detach`` cuts the gradient path from the delta loss back into the gate. That path is
    a nuisance term, not supervision: weighting the delta loss by the predicted probability
    lets the gate reduce the loss by *turning itself down* on objects the delta head finds
    hard, which competes with the BCE that is actually teaching it to detect change.

    With squared error the effect is mild (the delta term is O(0.05)). With an NLL
    objective it is not -- the delta term runs one to two orders of magnitude larger, and
    measured at 3 objects the gate collapsed from F1 0.869 (MSE head) to 0.430 (Gaussian)
    and 0.640 (MDN). That confounds any head comparison, because the heads would be scored
    through gates of very different quality, so detaching is the default whenever a
    probabilistic head is active.
    """
    if supervision == "predicted_probs":
        return gate_probs.detach() if detach else gate_probs
    if supervision == "predicted_gates":
        return gate_samples.detach() if detach else gate_samples
    if supervision == "target":
        return batch["object_change_mask"]
    raise ValueError(f"Unsupported delta supervision mode '{supervision}'.")


def resolve_detach_delta_gate(mode: str, delta_head: str) -> bool:
    """``auto`` detaches exactly when the delta objective is an NLL (see above)."""
    if mode == "auto":
        return delta_head != "mse"
    return mode == "always"


def resolve_gate_estimator(train: bool, estimator: str) -> str:
    if train:
        return estimator
    if estimator in {"gumbel", "gumbel_st"}:
        return "st"
    return estimator


def compute_pose_metrics(masked_delta: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    num_objects = int(batch["object_change_mask"].shape[1])
    current_pose = batch["current_object_pose"].reshape(-1, num_objects, POSE_DIM)
    target_next_pose = batch["next_object_pose"].reshape(-1, num_objects, POSE_DIM)
    predicted_next_pose = current_pose + masked_delta
    error = predicted_next_pose - target_next_pose
    l2 = torch.linalg.norm(error, dim=-1)
    changed_mask = batch["object_change_mask"].bool()
    unchanged_mask = ~changed_mask

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if mask.sum().item() == 0:
            return 0.0
        return float(values[mask].mean().item())

    return {
        "pose_l2": float(l2.mean().item()),
        "changed_pose_l2": masked_mean(l2, changed_mask),
        "unchanged_pose_l2": masked_mean(l2, unchanged_mask),
    }


def run_rollout_epoch(model, loader, device, optimizer, args, train: bool):
    """One pass of multi-step (autoregressive) training.

    At each step of the window the model is fed a state whose **object-pose slice has been
    replaced by its own running prediction**, while the exogenous parts (pusher, object
    velocities, goal) come from ground truth -- the exact protocol
    ``rollout_horizon_error.py`` uses to score rollouts, so training and evaluation see
    the same state distribution.

    The delta target is recomputed each step as ``true_next_pose - rolled_pose`` rather
    than reusing the recorded one-step delta. Those coincide at ``h=0``; afterwards the
    recomputed form is what teaches the model to *correct* accumulated drift instead of
    replaying a displacement measured from a state it is no longer in. Gradients flow
    through the whole unroll (``--grad-clip`` guards the resulting long product), and the
    gate is still supervised by the recorded change mask, so an object the gate leaves off
    is carried forward verbatim exactly as at deployment.
    """
    total = {
        "loss": 0.0,
        "gate_bce": 0.0,
        "delta_l2": 0.0,
        "delta_nll": 0.0,
        "effective_components": 0.0,
        "sparsity_penalty": 0.0,
        "pose_l2": 0.0,
        "changed_pose_l2": 0.0,
        "unchanged_pose_l2": 0.0,
        "final_step_pose_l2": 0.0,
        "positive_class_weight": 0.0,
    }
    pred_masks = []
    target_masks = []
    total_count = 0
    horizon = args.rollout_horizon

    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch["state"].shape[0]
            num_objects = int(batch["object_change_mask"].shape[2])
            layout = StateLayout(num_objects=num_objects)

            rolled_pose = batch["current_object_pose"][:, 0].reshape(batch_size, num_objects, POSE_DIM)
            step_losses = []
            step_stats = {key: 0.0 for key in ("gate_bce", "delta_l2", "delta_nll",
                                               "effective_components", "sparsity_penalty",
                                               "pose_l2", "changed_pose_l2", "unchanged_pose_l2")}
            for step in range(horizon):
                state_step = batch["state"][:, step].clone()
                state_step[:, layout.object_pose_slice] = rolled_pose.reshape(batch_size, -1)
                features = build_object_features_by_mode(
                    state_step, batch["action"][:, step], args.feature_mode
                )
                out = model(
                    features,
                    estimator=resolve_gate_estimator(train, args.estimator),
                    temperature=args.temperature,
                    hard=True,
                )
                target_next_pose = batch["next_object_pose"][:, step].reshape(
                    batch_size, num_objects, POSE_DIM
                )
                changed_mask = batch["object_change_mask"][:, step]
                if args.rollout_target == "correcting":
                    # Where the object must end up, measured from where the rollout
                    # currently believes it is. Teaches drift correction, but the target
                    # grows with accumulated error -- which a squared-error head chases and
                    # a likelihood head can discount. That asymmetry is a candidate
                    # explanation for the MSE/NLL split under rollout, so 'recorded' exists
                    # to test whether the split survives without it.
                    target_delta = target_next_pose - rolled_pose
                else:
                    # The true one-step displacement, independent of where the rollout has
                    # drifted to. Identical to 'correcting' at step 0.
                    target_delta = batch["object_delta"][:, step]

                step_batch = {"object_change_mask": changed_mask}
                losses = sparse_residual_loss(
                    out.gate.logits,
                    out.gate.probs,
                    out.delta,
                    changed_mask,
                    target_delta,
                    gate_loss_weight=args.gate_loss_weight,
                    delta_loss_weight=args.delta_loss_weight,
                    sparsity_weight=args.sparsity_weight,
                    positive_class_weight=args.positive_class_weight,
                    delta_gate=select_delta_supervision(
                        args.delta_supervision, step_batch, out.gate.probs, out.gate.gates,
                        detach=args.detach_gate_resolved,
                    ),
                    delta_dist=out.delta_dist,
                )
                step_losses.append(losses.total)

                rolled_pose = rolled_pose + out.masked_delta
                error = torch.linalg.norm(rolled_pose.detach() - target_next_pose, dim=-1)
                changed_bool = changed_mask.bool()
                step_stats["gate_bce"] += float(losses.gate_bce.item())
                step_stats["delta_l2"] += float(losses.delta_l2.item())
                if losses.delta_nll is not None:
                    step_stats["delta_nll"] += float(losses.delta_nll.item())
                if out.delta_dist is not None and hasattr(out.delta_dist, "effective_components"):
                    step_stats["effective_components"] += float(
                        out.delta_dist.effective_components().item()
                    )
                step_stats["sparsity_penalty"] += float(losses.sparsity_penalty.item())
                step_stats["pose_l2"] += float(error.mean().item())
                step_stats["changed_pose_l2"] += (
                    float(error[changed_bool].mean().item()) if changed_bool.any() else 0.0
                )
                step_stats["unchanged_pose_l2"] += (
                    float(error[~changed_bool].mean().item()) if (~changed_bool).any() else 0.0
                )
                if step == horizon - 1:
                    total["final_step_pose_l2"] += float(error.mean().item()) * batch_size
                    pred_masks.append((out.gate.probs.detach() >= 0.5).float().cpu())
                    target_masks.append(changed_mask.detach().cpu())

            loss = torch.stack(step_losses).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total["loss"] += float(loss.item()) * batch_size
            for key, value in step_stats.items():
                total[key] += (value / horizon) * batch_size
            total["positive_class_weight"] += float(args.positive_class_weight or 1.0) * batch_size
            total_count += batch_size

    metrics = {key: value / max(total_count, 1) for key, value in total.items()}
    gate_metrics = compute_gate_metrics(torch.cat(pred_masks, dim=0), torch.cat(target_masks, dim=0))
    metrics.update(gate_metrics)
    return metrics


def run_epoch(model, loader, device, optimizer, args, train: bool):
    total = {
        "loss": 0.0,
        "gate_bce": 0.0,
        "delta_l2": 0.0,
        "delta_nll": 0.0,
        "effective_components": 0.0,
        "sparsity_penalty": 0.0,
        "pose_l2": 0.0,
        "changed_pose_l2": 0.0,
        "unchanged_pose_l2": 0.0,
        # Same as pose_l2 in the one-step regime; defined here so both epoch functions
        # return the same key set and the logging path stays uniform.
        "final_step_pose_l2": 0.0,
        "positive_class_weight": 0.0,
    }
    pred_masks = []
    target_masks = []
    total_count = 0

    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            object_features = build_object_features_by_mode(
                batch["state"], batch["action"], args.feature_mode
            )
            out = model(
                object_features,
                estimator=resolve_gate_estimator(train, args.estimator),
                temperature=args.temperature,
                hard=True,
            )
            losses = sparse_residual_loss(
                out.gate.logits,
                out.gate.probs,
                out.delta,
                batch["object_change_mask"],
                batch["object_delta"],
                gate_loss_weight=args.gate_loss_weight,
                delta_loss_weight=args.delta_loss_weight,
                sparsity_weight=args.sparsity_weight,
                positive_class_weight=args.positive_class_weight,
                delta_gate=select_delta_supervision(
                    args.delta_supervision,
                    batch,
                    out.gate.probs,
                    out.gate.gates,
                    detach=args.detach_gate_resolved,
                ),
                delta_dist=out.delta_dist,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                optimizer.step()

            batch_size = batch["state"].shape[0]
            pose_metrics = compute_pose_metrics(out.masked_delta.detach(), batch)
            total["loss"] += float(losses.total.item()) * batch_size
            total["gate_bce"] += float(losses.gate_bce.item()) * batch_size
            total["delta_l2"] += float(losses.delta_l2.item()) * batch_size
            if losses.delta_nll is not None:
                total["delta_nll"] += float(losses.delta_nll.item()) * batch_size
            if out.delta_dist is not None and hasattr(out.delta_dist, "effective_components"):
                # Perplexity of the mixture weights. If this collapses to ~1 the MDN has
                # silently become a heteroscedastic Gaussian and the rung means nothing.
                total["effective_components"] += (
                    float(out.delta_dist.effective_components().item()) * batch_size
                )
            total["sparsity_penalty"] += float(losses.sparsity_penalty.item()) * batch_size
            total["pose_l2"] += pose_metrics["pose_l2"] * batch_size
            total["changed_pose_l2"] += pose_metrics["changed_pose_l2"] * batch_size
            total["unchanged_pose_l2"] += pose_metrics["unchanged_pose_l2"] * batch_size
            total["final_step_pose_l2"] += pose_metrics["pose_l2"] * batch_size
            total["positive_class_weight"] += float(losses.positive_class_weight.item()) * batch_size
            total_count += batch_size
            pred_masks.append((out.gate.probs.detach() >= 0.5).float().cpu())
            target_masks.append(batch["object_change_mask"].detach().cpu())

    metrics = {key: value / max(total_count, 1) for key, value in total.items()}
    gate_metrics = compute_gate_metrics(torch.cat(pred_masks, dim=0), torch.cat(target_masks, dim=0))
    metrics.update(gate_metrics)
    return metrics


def main() -> None:
    args = parse_args()
    args.detach_gate_resolved = resolve_detach_delta_gate(args.detach_delta_gate, args.delta_head)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    rollout = args.rollout_horizon > 1
    window_stats: dict[str, float | int] | None = None
    if rollout:
        train_dataset = TransitionSequenceDataset(
            args.train, horizon=args.rollout_horizon, stride=args.rollout_stride,
            max_windows=args.max_train_samples,
        )
        val_dataset = TransitionSequenceDataset(
            args.val, horizon=args.rollout_horizon, stride=args.rollout_stride,
            max_windows=args.max_val_samples,
        )
        window_stats = train_dataset.window_statistics()
        # Loud, because a silently tiny window count is the most likely way a rollout run
        # produces a meaningless result: an aggressive filter can leave a split with
        # thousands of rows but only a handful of usable windows.
        print(f"[rollout] train windows: {json.dumps(window_stats)}", flush=True)
        print(f"[rollout] val windows:   {json.dumps(val_dataset.window_statistics())}", flush=True)
    else:
        train_dataset = TransitionDataset(args.train, max_samples=args.max_train_samples)
        val_dataset = TransitionDataset(args.val, max_samples=args.max_val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    epoch_fn = run_rollout_epoch if rollout else run_epoch

    train_positive_fraction = float(train_dataset.object_change_mask.float().mean().item())
    if args.auto_balance_bce:
        negative_fraction = max(1.0 - train_positive_fraction, 1e-8)
        positive_fraction = max(train_positive_fraction, 1e-8)
        args.positive_class_weight = negative_fraction / positive_fraction

    sample = train_dataset[0]
    # Sequence items carry a leading horizon axis; take its first step so the probe is a
    # single (state, action) pair either way.
    probe_state = sample["state"][0] if rollout else sample["state"]
    probe_action = sample["action"][0] if rollout else sample["action"]
    feature_dim = build_object_features_by_mode(
        probe_state.unsqueeze(0), probe_action.unsqueeze(0), args.feature_mode
    ).shape[-1]

    model = SparseResidualHead(
        object_feature_dim=feature_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_num_layers=args.gate_num_layers,
        delta_hidden_dim=args.delta_hidden_dim,
        delta_num_layers=args.delta_num_layers,
        delta_head_type=args.delta_head,
        num_mixture_components=args.mixture_components,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_parameters = sum(p.numel() for p in model.parameters())

    logger = ExperimentLogger(run_name=args.run_name)
    logger.log_config({
        "model": "SparseResidualHead",
        "train_split": str(args.train),
        "val_split": str(args.val),
        "num_objects": train_dataset.num_objects,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "num_train_samples": len(train_dataset),
        "feature_mode": args.feature_mode,
        "feature_dim": feature_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "gate_hidden_dim": args.gate_hidden_dim,
        "delta_hidden_dim": args.delta_hidden_dim,
        "gate_num_layers": args.gate_num_layers,
        "delta_num_layers": args.delta_num_layers,
        "delta_head_type": args.delta_head,
        "mixture_components": args.mixture_components,
        "num_parameters": num_parameters,
        "rollout_horizon": args.rollout_horizon,
        "rollout_target": args.rollout_target,
        "rollout_stride": args.rollout_stride,
        "grad_clip": args.grad_clip,
        "window_statistics": window_stats,
        "gate_loss_weight": args.gate_loss_weight,
        "delta_loss_weight": args.delta_loss_weight,
        "sparsity_weight": args.sparsity_weight,
        "positive_class_weight": args.positive_class_weight,
        "auto_balance_bce": args.auto_balance_bce,
        "delta_supervision": args.delta_supervision,
        "detach_delta_gate": args.detach_delta_gate,
        "detach_delta_gate_resolved": args.detach_gate_resolved,
        "estimator": args.estimator,
        "eval_estimator": resolve_gate_estimator(False, args.estimator),
        "train_positive_fraction": train_positive_fraction,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": args.device,
    })

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = None
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"{args.run_name}.pt"

    for epoch in range(args.epochs):
        train_metrics = epoch_fn(model, train_loader, device, optimizer, args, train=True)
        val_metrics = epoch_fn(model, val_loader, device, optimizer, args, train=False)
        logger.log_metrics(
            epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            train_gate_precision=train_metrics["gate_precision"],
            val_gate_precision=val_metrics["gate_precision"],
            train_gate_recall=train_metrics["gate_recall"],
            val_gate_recall=val_metrics["gate_recall"],
            train_gate_f1=train_metrics["gate_f1"],
            val_gate_f1=val_metrics["gate_f1"],
            train_pose_l2=train_metrics["pose_l2"],
            val_pose_l2=val_metrics["pose_l2"],
            train_changed_pose_l2=train_metrics["changed_pose_l2"],
            val_changed_pose_l2=val_metrics["changed_pose_l2"],
            train_unchanged_pose_l2=train_metrics["unchanged_pose_l2"],
            val_unchanged_pose_l2=val_metrics["unchanged_pose_l2"],
            train_sparsity_penalty=train_metrics["sparsity_penalty"],
            val_sparsity_penalty=val_metrics["sparsity_penalty"],
            train_gate_bce=train_metrics["gate_bce"],
            val_gate_bce=val_metrics["gate_bce"],
            train_delta_l2=train_metrics["delta_l2"],
            val_delta_l2=val_metrics["delta_l2"],
            train_delta_nll=train_metrics["delta_nll"],
            val_delta_nll=val_metrics["delta_nll"],
            train_effective_components=train_metrics["effective_components"],
            val_effective_components=val_metrics["effective_components"],
            train_gate_predicted_positive_rate=train_metrics["gate_predicted_positive_rate"],
            val_gate_predicted_positive_rate=val_metrics["gate_predicted_positive_rate"],
            train_gate_target_positive_rate=train_metrics["gate_target_positive_rate"],
            val_gate_target_positive_rate=val_metrics["gate_target_positive_rate"],
            train_gate_positive_rate_gap=train_metrics["gate_positive_rate_gap"],
            val_gate_positive_rate_gap=val_metrics["gate_positive_rate_gap"],
            train_gate_false_positive_rate=train_metrics["gate_false_positive_rate"],
            val_gate_false_positive_rate=val_metrics["gate_false_positive_rate"],
            train_gate_false_negative_rate=train_metrics["gate_false_negative_rate"],
            val_gate_false_negative_rate=val_metrics["gate_false_negative_rate"],
            train_gate_all_changed_fraction=train_metrics["gate_all_changed_fraction"],
            val_gate_all_changed_fraction=val_metrics["gate_all_changed_fraction"],
            train_gate_all_unchanged_fraction=train_metrics["gate_all_unchanged_fraction"],
            val_gate_all_unchanged_fraction=val_metrics["gate_all_unchanged_fraction"],
            train_positive_class_weight=train_metrics["positive_class_weight"],
            val_positive_class_weight=val_metrics["positive_class_weight"],
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "feature_dim": feature_dim,
                    "feature_mode": args.feature_mode,
                    "num_objects": train_dataset.num_objects,
                    "gate_hidden_dim": args.gate_hidden_dim,
                    "delta_hidden_dim": args.delta_hidden_dim,
                    "gate_num_layers": args.gate_num_layers,
                    "delta_num_layers": args.delta_num_layers,
                    "delta_head_type": args.delta_head,
                    "mixture_components": args.mixture_components,
                    "num_parameters": num_parameters,
                    "rollout_horizon": args.rollout_horizon,
                    "estimator": args.estimator,
                    "eval_estimator": resolve_gate_estimator(False, args.estimator),
                    "temperature": args.temperature,
                    "gate_loss_weight": args.gate_loss_weight,
                    "delta_loss_weight": args.delta_loss_weight,
                    "sparsity_weight": args.sparsity_weight,
                    "positive_class_weight": args.positive_class_weight,
                    "train_positive_fraction": train_positive_fraction,
                    "delta_supervision": args.delta_supervision,
                },
            }, checkpoint_path)

    if best_val_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")

    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "best_val_gate_precision": best_val_metrics["gate_precision"],
        "best_val_gate_recall": best_val_metrics["gate_recall"],
        "best_val_gate_f1": best_val_metrics["gate_f1"],
        "best_val_pose_l2": best_val_metrics["pose_l2"],
        "best_val_changed_pose_l2": best_val_metrics["changed_pose_l2"],
        "best_val_unchanged_pose_l2": best_val_metrics["unchanged_pose_l2"],
        "best_val_gate_predicted_positive_rate": best_val_metrics["gate_predicted_positive_rate"],
        "best_val_gate_target_positive_rate": best_val_metrics["gate_target_positive_rate"],
        "best_val_gate_positive_rate_gap": best_val_metrics["gate_positive_rate_gap"],
        "best_val_gate_false_positive_rate": best_val_metrics["gate_false_positive_rate"],
        "best_val_gate_false_negative_rate": best_val_metrics["gate_false_negative_rate"],
        "best_val_gate_all_changed_fraction": best_val_metrics["gate_all_changed_fraction"],
        "best_val_gate_all_unchanged_fraction": best_val_metrics["gate_all_unchanged_fraction"],
        "delta_head_type": args.delta_head,
        "mixture_components": args.mixture_components,
        "num_parameters": num_parameters,
        "best_val_delta_nll": best_val_metrics["delta_nll"],
        "best_val_effective_components": best_val_metrics["effective_components"],
        "best_val_final_step_pose_l2": best_val_metrics["final_step_pose_l2"],
        "rollout_horizon": args.rollout_horizon,
        "window_statistics": window_stats,
        "sparsity_weight": args.sparsity_weight,
        "positive_class_weight": args.positive_class_weight,
        "train_positive_fraction": train_positive_fraction,
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
