from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .layout import StateLayout, infer_num_objects_from_state_dim


OBJECT_POSE_SLICE = StateLayout(num_objects=3).object_pose_slice


class TransitionDataset(Dataset):
    """Loads transition tuples from a saved `.npz` split."""

    def __init__(
        self,
        path: str | Path,
        predict_delta: bool = False,
        target_slice: slice | None = None,
        max_samples: int | None = None,
    ):
        data = np.load(Path(path))
        state = torch.from_numpy(data["s_t"]).float()
        action = torch.from_numpy(data["a_t"]).float()
        next_state = torch.from_numpy(data["s_t1"]).float()
        object_change_mask = torch.from_numpy(data["object_change_mask"]).float()
        object_delta = torch.from_numpy(data["object_delta"]).float()

        if max_samples is not None:
            state = state[:max_samples]
            action = action[:max_samples]
            next_state = next_state[:max_samples]
            object_change_mask = object_change_mask[:max_samples]
            object_delta = object_delta[:max_samples]

        self.state = state
        self.action = action
        self.next_state = next_state
        self.object_change_mask = object_change_mask
        self.object_delta = object_delta
        self.predict_delta = predict_delta
        self.num_objects = int(object_change_mask.shape[1])
        inferred_num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
        if inferred_num_objects != self.num_objects:
            raise ValueError(
                f"Dataset inconsistency: state implies {inferred_num_objects} objects but mask has {self.num_objects}."
            )
        self.layout = StateLayout(num_objects=self.num_objects)
        self.target_slice = target_slice or self.layout.object_pose_slice

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state = self.state[index]
        action = self.action[index]
        next_state = self.next_state[index]
        next_object_pose = next_state[self.target_slice]
        current_object_pose = state[self.target_slice]
        target = next_object_pose - current_object_pose if self.predict_delta else next_object_pose
        return {
            "state": state,
            "action": action,
            "target": target,
            "next_state": next_state,
            "current_object_pose": current_object_pose,
            "next_object_pose": next_object_pose,
            "object_change_mask": self.object_change_mask[index],
            "object_delta": self.object_delta[index],
        }
