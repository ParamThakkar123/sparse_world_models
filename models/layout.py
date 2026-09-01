from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


PUSHER_DIM = 2
POSE_DIM = 3
VELOCITY_DIM = 6
GOAL_DIM = 2
STATE_PREFIX_DIM = PUSHER_DIM
STATE_SUFFIX_DIM = GOAL_DIM
PER_OBJECT_STATE_DIM = POSE_DIM + VELOCITY_DIM


@dataclass(frozen=True)
class StateLayout:
    num_objects: int

    @property
    def object_pose_slice(self) -> slice:
        start = PUSHER_DIM
        stop = start + self.num_objects * POSE_DIM
        return slice(start, stop)

    @property
    def object_velocity_slice(self) -> slice:
        start = self.object_pose_slice.stop
        stop = start + self.num_objects * VELOCITY_DIM
        return slice(start, stop)

    @property
    def goal_slice(self) -> slice:
        start = self.object_velocity_slice.stop
        stop = start + GOAL_DIM
        return slice(start, stop)

    @property
    def state_dim(self) -> int:
        return int(self.goal_slice.stop)


def infer_num_objects_from_state_dim(state_dim: int) -> int:
    if state_dim < PUSHER_DIM + GOAL_DIM:
        raise ValueError(f"State dim {state_dim} is too small to encode tabletop state.")
    remainder = state_dim - PUSHER_DIM - GOAL_DIM
    if remainder % PER_OBJECT_STATE_DIM != 0:
        raise ValueError(
            f"State dim {state_dim} is incompatible with tabletop layout: remainder {remainder} is not divisible by {PER_OBJECT_STATE_DIM}."
        )
    num_objects = remainder // PER_OBJECT_STATE_DIM
    if num_objects < 1:
        raise ValueError("Tabletop state must contain at least one object.")
    return int(num_objects)


def layout_from_state_dim(state_dim: int) -> StateLayout:
    return StateLayout(num_objects=infer_num_objects_from_state_dim(state_dim))


def layout_from_state_tensor(state: torch.Tensor | np.ndarray) -> StateLayout:
    state_dim = int(state.shape[-1])
    return layout_from_state_dim(state_dim)


def reshape_object_pose(batch: torch.Tensor | np.ndarray, num_objects: int | None = None):
    inferred_num_objects = num_objects
    if inferred_num_objects is None:
        inferred_num_objects = infer_num_objects_from_state_dim(int(batch.shape[-1]))
    layout = StateLayout(num_objects=inferred_num_objects)
    return batch[..., layout.object_pose_slice].reshape(*batch.shape[:-1], inferred_num_objects, POSE_DIM)


def reshape_object_velocity(batch: torch.Tensor | np.ndarray, num_objects: int | None = None):
    inferred_num_objects = num_objects
    if inferred_num_objects is None:
        inferred_num_objects = infer_num_objects_from_state_dim(int(batch.shape[-1]))
    layout = StateLayout(num_objects=inferred_num_objects)
    return batch[..., layout.object_velocity_slice].reshape(*batch.shape[:-1], inferred_num_objects, VELOCITY_DIM)
