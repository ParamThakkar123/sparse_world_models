"""Counterfactual data augmentation driven by the learned change gate (W3).

The gate is trained as a change detector, but what it actually estimates is a **local causal
mask**: which objects are in the causal parent set of the next state, given this state and
action. If object ``i`` is genuinely unaffected this step, then its pose is causally
independent of everything else that happened -- so replacing it with *any* other pose that
does not create a new interaction leaves the transition valid. That is the CoDA argument
(Pitis et al.), and it turns the gate from a modelling device into a data generator.

Concretely, given a real transition ``(s, a, s')``:

1. Take the set of objects the mask marks unchanged.
2. For each, propose a replacement pose sampled from a *different* episode.
3. Accept the replacement only if it stays clear of the pusher, the goal, every changed
   object, and every other object -- i.e. it could not have participated in this step's
   contact.
4. Emit ``(s_cf, a, s'_cf)`` where the accepted objects hold their new pose in both ``s``
   and ``s'`` (they did not move), and everything else is copied verbatim.

The result is a transition the model has never seen, in a scene configuration that never
occurred, whose dynamics are still exactly right.

**The clearance test is what makes this sound, and it is deliberately conservative.** An
unchanged object is only known to be independent *in the configuration it was actually in*.
Dropping it somewhere new can manufacture an interaction that the recorded next-state does
not reflect -- the object would have been struck, but our synthetic label says it stayed
put. Rejecting placements near anything that moved, or near the pusher's swept path, keeps
the synthesized label true. ``experiments/counterfactual_augmentation.py`` measures the
resulting validity against the real simulator rather than trusting this argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import POSE_DIM, StateLayout, infer_num_objects_from_state_dim

# Pusher sphere radius (0.02) + object half-extent (0.025), matching
# experiments/train_sparse_model.CONTACT_RADIUS. A replacement pose closer than this to the
# pusher would be in contact.
CONTACT_RADIUS = 0.045
# Two 5 cm boxes need 0.05 between centres to not overlap axis-aligned, and 0.0707 at 45
# degrees. We demand more than the worst case so a replacement never intersects a neighbour.
OBJECT_CLEARANCE = 0.08
# Extra margin around anything that moved this step, and around the pusher's swept path.
# A replacement inside this ring might have been hit, which the copied "unchanged" label
# would then misreport.
INTERACTION_MARGIN = 0.12


@dataclass
class AugmentationStats:
    """Bookkeeping for one augmentation pass -- the rejection rate is the interesting part."""

    proposed: int = 0
    accepted: int = 0
    rejected_no_free_object: int = 0
    rejected_clearance: int = 0
    objects_replaced: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "rejected_no_free_object": self.rejected_no_free_object,
            "rejected_clearance": self.rejected_clearance,
            "objects_replaced": self.objects_replaced,
            "mean_objects_replaced": self.objects_replaced / max(self.accepted, 1),
        }


def _pose_block(state: np.ndarray, layout: StateLayout, num_objects: int) -> np.ndarray:
    return state[layout.object_pose_slice].reshape(num_objects, POSE_DIM)


def placement_is_clear(
    candidate_xy: np.ndarray,
    *,
    other_xy: np.ndarray,
    moved_xy: np.ndarray,
    pusher_xy: np.ndarray,
    pusher_next_xy: np.ndarray,
) -> bool:
    """Whether a replacement pose could have sat there without changing what happened.

    Rejects placements that overlap another object, that sit within the interaction margin
    of anything that moved, or that lie near the segment the pusher swept this step.
    """
    if other_xy.size and np.min(np.linalg.norm(other_xy - candidate_xy, axis=1)) < OBJECT_CLEARANCE:
        return False
    if moved_xy.size and np.min(np.linalg.norm(moved_xy - candidate_xy, axis=1)) < INTERACTION_MARGIN:
        return False
    return _distance_to_segment(candidate_xy, pusher_xy, pusher_next_xy) >= CONTACT_RADIUS + INTERACTION_MARGIN


def _distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Point-to-segment distance, used for the pusher's swept path.

    Testing only the pusher's start and end positions would miss an object the pusher passed
    straight through within a single control step.
    """
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ segment / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def generate_counterfactuals(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    change_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    num_samples: int,
    pose_pool: np.ndarray | None = None,
    respect_mask: bool = True,
    max_attempts_per_sample: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AugmentationStats]:
    """Synthesize counterfactual transitions by relocating causally-independent objects.

    ``change_mask`` is ``(num_transitions, num_objects)`` -- the learned gate's output, or the
    ground-truth mask for the oracle condition.

    ``respect_mask=False`` is the **ablation that makes this experiment mean something**: it
    relocates objects chosen uniformly at random, including ones that moved. Those synthetic
    transitions are physically wrong (an object that was struck is reported as having stayed
    where it was newly placed), so if augmentation helps only in the masked condition, the
    mask is what carries the benefit rather than the extra data volume.

    ``pose_pool`` supplies candidate replacement poses, normally gathered from *other*
    episodes; it falls back to the poses present in this batch.
    """
    num_transitions = state.shape[0]
    num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    pose_slice = layout.object_pose_slice

    if pose_pool is None:
        pose_pool = state[:, pose_slice].reshape(-1, POSE_DIM)

    stats = AugmentationStats()
    out_state, out_action, out_next = [], [], []

    for _ in range(num_samples):
        index = int(rng.integers(num_transitions))
        stats.proposed += 1

        current_pose = _pose_block(state[index], layout, num_objects).copy()
        next_pose = _pose_block(next_state[index], layout, num_objects).copy()
        mask = change_mask[index].astype(bool)

        if respect_mask:
            free = np.flatnonzero(~mask)
        else:
            # Ignore causal structure entirely: any object is fair game.
            free = np.arange(num_objects)
        if free.size == 0:
            stats.rejected_no_free_object += 1
            continue

        pusher_xy = state[index, 0:2]
        clipped_action = np.clip(action[index], -1.0, 1.0)
        pusher_next_xy = np.clip(pusher_xy + clipped_action * 0.04, -0.26, 0.26)
        moved_xy = current_pose[mask, :2]

        new_current = current_pose.copy()
        new_next = next_pose.copy()
        replaced = 0
        for object_index in rng.permutation(free):
            for _ in range(max_attempts_per_sample):
                candidate = pose_pool[int(rng.integers(pose_pool.shape[0]))].copy()
                keep = np.array([i for i in range(num_objects) if i != object_index])
                other_xy = new_current[keep, :2] if keep.size else np.empty((0, 2))
                if not respect_mask or placement_is_clear(
                    candidate[:2],
                    other_xy=other_xy,
                    moved_xy=moved_xy,
                    pusher_xy=pusher_xy,
                    pusher_next_xy=pusher_next_xy,
                ):
                    new_current[object_index] = candidate
                    # The defining assumption: this object is causally inert this step, so
                    # its next pose is its (new) current pose.
                    new_next[object_index] = candidate
                    replaced += 1
                    break

        if replaced == 0:
            stats.rejected_clearance += 1
            continue

        synthetic_state = state[index].copy()
        synthetic_next = next_state[index].copy()
        synthetic_state[pose_slice] = new_current.reshape(-1)
        synthetic_next[pose_slice] = new_next.reshape(-1)

        out_state.append(synthetic_state)
        out_action.append(action[index].copy())
        out_next.append(synthetic_next)
        stats.accepted += 1
        stats.objects_replaced += replaced

    if not out_state:
        empty_state = np.empty((0, state.shape[1]), dtype=state.dtype)
        return (
            empty_state,
            np.empty((0, action.shape[1]), dtype=action.dtype),
            empty_state.copy(),
            stats,
        )
    return (
        np.stack(out_state).astype(np.float32),
        np.stack(out_action).astype(np.float32),
        np.stack(out_next).astype(np.float32),
        stats,
    )


def recompute_labels(
    state: np.ndarray, next_state: np.ndarray, position_eps: float, yaw_eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Derive change mask and delta for synthetic transitions.

    Recomputed from the synthetic poses with the same thresholds the data generator uses,
    rather than copied from the source transition -- copying would silently carry the source
    scene's labels onto a different configuration.
    """
    num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    current = state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
    following = next_state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)

    delta = following - current
    wrapped_yaw = (delta[:, :, 2] + np.pi) % (2.0 * np.pi) - np.pi
    delta = delta.copy()
    delta[:, :, 2] = wrapped_yaw

    moved = np.linalg.norm(delta[:, :, :2], axis=2) > position_eps
    turned = np.abs(wrapped_yaw) > yaw_eps
    return np.logical_or(moved, turned).astype(np.float32), delta.astype(np.float32)
