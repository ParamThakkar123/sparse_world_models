from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, _: dict[str, np.ndarray]) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=2)


@dataclass
class ScriptedPushPolicy:
    target_object: int = 0
    approach_offset: float = 0.08
    gain: float = 6.0
    switch_threshold: float = 0.028
    push_distance: float = 0.015

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        goal_xy = obs["goal_xy"]
        pusher_xy = obs["pusher_xy"]
        object_xy = obs["object_poses"][self.target_object, :2]

        push_dir = goal_xy - object_xy
        norm = np.linalg.norm(push_dir)
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float64)
        push_dir = push_dir / norm

        behind_object = object_xy - self.approach_offset * push_dir
        if np.linalg.norm(pusher_xy - behind_object) > self.switch_threshold:
            target_xy = behind_object
        else:
            target_xy = object_xy + self.push_distance * push_dir

        delta = (target_xy - pusher_xy) * self.gain
        return np.clip(delta, -1.0, 1.0)
