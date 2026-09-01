from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mujoco' package is required for TabletopPushEnv. "
        "Install it with `pip install mujoco`."
    ) from exc


OBJECT_MATERIALS = [
    "obj_red",
    "obj_green",
    "obj_blue",
    "obj_orange",
    "obj_teal",
    "obj_pink",
    "obj_olive",
    "obj_slate",
]


@dataclass
class TabletopPushConfig:
    num_objects: int = 3
    control_dt: float = 0.05
    physics_dt: float = 0.005
    action_scale: float = 0.04
    pusher_bounds: tuple[float, float] = (-0.26, 0.26)
    object_bounds: tuple[float, float] = (-0.18, 0.18)
    min_object_separation: float = 0.12
    goal_clearance: float = 0.1
    goal_xy: tuple[float, float] = (0.18, 0.18)
    target_object: int = 0
    max_steps: int = 200
    seed: int | None = None


class TabletopPushEnv:
    """Minimal MuJoCo tabletop pushing environment with configurable free objects."""

    def __init__(self, config: TabletopPushConfig | None = None):
        self.config = config or TabletopPushConfig()
        if self.config.num_objects < 1:
            raise ValueError("num_objects must be at least 1.")
        if not 0 <= self.config.target_object < self.config.num_objects:
            raise ValueError("target_object must index one of the configured objects.")

        xml = build_tabletop_push_xml(self.config.num_objects)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.config.physics_dt
        self.data = mujoco.MjData(self.model)
        self.frames_per_step = max(1, int(round(self.config.control_dt / self.config.physics_dt)))
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0

        self._pusher_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_x")],
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_y")],
            ],
            dtype=np.int32,
        )
        self._object_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{idx}")
            for idx in range(self.config.num_objects)
        ]
        self._object_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{idx}_free")
            for idx in range(self.config.num_objects)
        ]
        self._object_qpos_adrs = [
            self.model.jnt_qposadr[joint_id] for joint_id in self._object_joint_ids
        ]
        # Free joints carry 7 qpos but only 6 qvel, so the two address spaces diverge and
        # the velocity slice has to be looked up rather than derived from the qpos address.
        self._object_qvel_adrs = [
            self.model.jnt_dofadr[joint_id] for joint_id in self._object_joint_ids
        ]
        self._goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._set_pusher_pose(np.array([0.0, -0.22], dtype=np.float64))
        self._set_goal_site()
        self._randomize_objects()
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        current_xy = self.data.qpos[self._pusher_qpos_adr].copy()
        target_xy = current_xy + action * self.config.action_scale
        target_xy = np.clip(target_xy, *self.config.pusher_bounds)
        self.data.ctrl[:] = target_xy

        for _ in range(self.frames_per_step):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self.get_observation()
        reward = self._compute_reward(obs)
        success = self._target_distance(obs) < 0.05
        terminated = success or self.step_count >= self.config.max_steps
        info = {"success": success, "target_distance": self._target_distance(obs)}
        return obs, reward, terminated, info

    def get_observation(self) -> dict[str, np.ndarray]:
        object_poses = []
        object_velocities = []
        for body_id in self._object_body_ids:
            object_poses.append(
                np.concatenate(
                    [
                        self.data.xpos[body_id].copy(),
                        self.data.xquat[body_id].copy(),
                    ]
                )
            )
            object_velocities.append(self.data.cvel[body_id].copy())

        return {
            "pusher_xy": self.data.qpos[self._pusher_qpos_adr].copy(),
            "object_poses": np.stack(object_poses, axis=0),
            "object_velocities": np.stack(object_velocities, axis=0),
            "goal_xy": np.array(self.config.goal_xy, dtype=np.float64),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        obs = self.get_observation()
        yaws = np.array([self._quat_to_yaw(quat) for quat in obs["object_poses"][:, 3:]], dtype=np.float64)
        planar_pose = np.column_stack((obs["object_poses"][:, 0], obs["object_poses"][:, 1], yaws))
        return {
            "object_pose": planar_pose,
            "object_pose_full": obs["object_poses"],
            "object_velocity": obs["object_velocities"],
            "pusher_xy": obs["pusher_xy"],
        }

    def sample_random_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=2)

    def relocate_object(self, index: int, xy: np.ndarray, yaw: float | None = None) -> None:
        """Move one object to a new resting pose, leaving the rest of the state untouched.

        This is the surgical operation counterfactual splicing needs, and the reason
        :meth:`set_planar_state` is *not* usable for verifying it. The reduced planar state
        drops z, tilt, contact history and velocities, so reconstructing a whole mid-episode
        scene from it is not faithful -- teleporting the pusher into a position where it was
        already penetrating an object makes the solver eject that object metres away.
        Restoring an exact :meth:`snapshot` and then relocating only the objects under test
        keeps every other degree of freedom bit-identical, so any difference observed after
        stepping is attributable to the relocation alone.

        The relocated object is placed at rest (zero velocity, upright, resting height),
        which is faithful precisely because the splicing rule only ever relocates objects it
        has established are stationary and clear of contact.
        """
        if not 0 <= index < len(self._object_qpos_adrs):
            raise IndexError(f"object index {index} out of range")
        qpos_adr = self._object_qpos_adrs[index]
        qvel_adr = self._object_qvel_adrs[index]
        if yaw is None:
            yaw = self._quat_to_yaw(self.data.qpos[qpos_adr + 3 : qpos_adr + 7])
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.array(
            [float(xy[0]), float(xy[1]), 0.105], dtype=np.float64
        )
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64
        )
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_planar_state(
        self,
        pusher_xy: np.ndarray,
        object_pose: np.ndarray,
        object_velocity: np.ndarray | None = None,
    ) -> None:
        """Drive the simulator to an arbitrary planar configuration.

        Inverse of the reduced state in ``experiments.generate_transitions.flatten_state``:
        ``object_pose`` is ``(num_objects, 3)`` as ``(x, y, yaw)``. Objects are placed at
        resting height with a yaw-only orientation, and velocities default to zero.

        Unlike :meth:`restore`, this does **not** reproduce a previously visited state -- it
        constructs one that may never have occurred. That is exactly what verifying
        counterfactual (spliced) transitions requires: put the simulator in the synthesized
        configuration, take the recorded action, and check the real dynamics against the
        synthesized label. Because the reduced state omits tilt and contact history, a state
        set this way is only equivalent to the original up to those omissions -- which is
        why the counterfactual generator restricts itself to flat, resting configurations.
        """
        object_pose = np.asarray(object_pose, dtype=np.float64).reshape(-1, 3)
        if object_pose.shape[0] != len(self._object_qpos_adrs):
            raise ValueError(
                f"object_pose has {object_pose.shape[0]} rows but the env has "
                f"{len(self._object_qpos_adrs)} objects."
            )
        self.data.qvel[:] = 0.0
        self._set_pusher_pose(np.asarray(pusher_xy, dtype=np.float64).reshape(2))
        for row, qpos_adr in zip(object_pose, self._object_qpos_adrs):
            yaw = float(row[2])
            self.data.qpos[qpos_adr : qpos_adr + 3] = np.array(
                [float(row[0]), float(row[1]), 0.105], dtype=np.float64
            )
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64
            )
        if object_velocity is not None:
            velocity = np.asarray(object_velocity, dtype=np.float64).reshape(-1, 6)
            for row, qvel_adr in zip(velocity, self._object_qvel_adrs):
                self.data.qvel[qvel_adr : qvel_adr + 6] = row
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict[str, Any]:
        """Capture the full simulator state for later exact restoration.

        Copies the complete MuJoCo dynamical state (``qpos``/``qvel``/``act``/
        ``ctrl``/``time``) plus the step counter. Used by oracle planners that roll
        the *true* simulator forward over candidate action sequences and then rewind
        to the decision point. This is exact -- unlike the reduced flat state, it
        preserves velocities and contact state so a restored rollout is identical.
        """
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "step_count": int(self.step_count),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a state captured by :meth:`snapshot` (inverse operation)."""
        self.data.qpos[:] = snapshot["qpos"]
        self.data.qvel[:] = snapshot["qvel"]
        if self.data.act.size:
            self.data.act[:] = snapshot["act"]
        self.data.ctrl[:] = snapshot["ctrl"]
        self.data.time = snapshot["time"]
        self.step_count = snapshot["step_count"]
        mujoco.mj_forward(self.model, self.data)

    def _set_pusher_pose(self, xy: np.ndarray) -> None:
        self.data.qpos[self._pusher_qpos_adr] = xy
        self.data.ctrl[:] = xy

    def _set_goal_site(self) -> None:
        goal = np.array([self.config.goal_xy[0], self.config.goal_xy[1], 0.081], dtype=np.float64)
        self.model.site_pos[self._goal_site_id] = goal

    def _randomize_objects(self) -> None:
        placements: list[np.ndarray] = []
        lower, upper = self.config.object_bounds
        for qpos_adr in self._object_qpos_adrs:
            xy = self._sample_non_overlapping_xy(lower, upper, placements)
            placements.append(xy)
            self.data.qpos[qpos_adr : qpos_adr + 3] = np.array([xy[0], xy[1], 0.105], dtype=np.float64)
            yaw = self.rng.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    def _sample_non_overlapping_xy(
        self, lower: float, upper: float, placements: list[np.ndarray]
    ) -> np.ndarray:
        goal_xy = np.array(self.config.goal_xy)
        # Budget is generous so dense (tightly-packed) configurations place reliably;
        # for loose configs the loop returns on the first try, so behaviour is unchanged.
        for _ in range(2000):
            xy = self.rng.uniform(lower, upper, size=2)
            if np.linalg.norm(xy - np.array([0.0, -0.22])) < 0.1:
                continue
            if np.linalg.norm(xy - goal_xy) < self.config.goal_clearance:
                continue
            if all(np.linalg.norm(xy - other) > self.config.min_object_separation for other in placements):
                return xy
        raise RuntimeError("Could not sample a valid non-overlapping object placement.")

    def _target_distance(self, obs: dict[str, np.ndarray]) -> float:
        target_xy = obs["object_poses"][self.config.target_object, :2]
        return float(np.linalg.norm(target_xy - obs["goal_xy"]))

    def _compute_reward(self, obs: dict[str, np.ndarray]) -> float:
        return -self._target_distance(obs)

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))


def build_tabletop_push_xml(num_objects: int) -> str:
    if num_objects < 1:
        raise ValueError("num_objects must be at least 1.")

    object_bodies: list[str] = []
    base_positions = [
        (-0.12, 0.02),
        (0.02, -0.03),
        (0.12, 0.07),
        (-0.06, 0.12),
        (0.08, -0.12),
        (0.15, -0.02),
        (-0.15, -0.06),
        (0.0, 0.15),
    ]
    for idx in range(num_objects):
        x, y = base_positions[idx % len(base_positions)]
        material = OBJECT_MATERIALS[idx % len(OBJECT_MATERIALS)]
        object_bodies.append(
            f'''    <body name="object_{idx}" pos="{x:.3f} {y:.3f} 0.105">
      <freejoint name="object_{idx}_free"/>
      <geom name="object_{idx}_geom" type="box" size="0.025 0.025 0.025" material="{material}" mass="0.08" friction="0.12 0.01 0.002"/>
    </body>'''
        )

    object_body_block = "\n\n".join(object_bodies)
    return f'''<mujoco model="tabletop_push_{num_objects}obj">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.96 0.97 0.99" rgb2="0.76 0.83 0.9" width="256" height="256"/>
    <material name="table" rgba="0.75 0.72 0.68 1"/>
    <material name="wall" rgba="0.35 0.4 0.45 1"/>
    <material name="pusher" rgba="0.1 0.1 0.1 1"/>
    <material name="obj_red" rgba="0.85 0.25 0.25 1"/>
    <material name="obj_green" rgba="0.2 0.7 0.3 1"/>
    <material name="obj_blue" rgba="0.2 0.35 0.85 1"/>
    <material name="obj_orange" rgba="0.92 0.55 0.18 1"/>
    <material name="obj_teal" rgba="0.16 0.66 0.62 1"/>
    <material name="obj_pink" rgba="0.86 0.38 0.6 1"/>
    <material name="obj_olive" rgba="0.56 0.63 0.18 1"/>
    <material name="obj_slate" rgba="0.35 0.48 0.62 1"/>
    <material name="goal" rgba="0.95 0.8 0.2 0.35"/>
  </asset>

  <default>
    <geom condim="4" friction="0.8 0.05 0.01" solref="0.01 1"/>
    <joint damping="1" armature="0.01"/>
    <position kp="300"/>
  </default>

  <worldbody>
    <light pos="0 0 1.6" dir="0 0 -1"/>
    <camera name="overview" pos="0 -0.95 0.75" xyaxes="1 0 0 0 0.58 0.82"/>

    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.92 0.92 0.92 1"/>
    <body name="table" pos="0 0 0.04">
      <geom name="tabletop" type="box" size="0.34 0.34 0.04" material="table"/>
      <geom name="wall_north" type="box" pos="0 0.34 0.05" size="0.34 0.01 0.05" material="wall"/>
      <geom name="wall_south" type="box" pos="0 -0.34 0.05" size="0.34 0.01 0.05" material="wall"/>
      <geom name="wall_east" type="box" pos="0.34 0 0.05" size="0.01 0.34 0.05" material="wall"/>
      <geom name="wall_west" type="box" pos="-0.34 0 0.05" size="0.01 0.34 0.05" material="wall"/>
    </body>

    <site name="goal_site" type="cylinder" pos="0.18 0.18 0.081" size="0.045 0.002" material="goal"/>

    <body name="pusher" pos="0 0 0.105">
      <joint name="pusher_x" type="slide" axis="1 0 0" range="-0.26 0.26"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" range="-0.26 0.26"/>
      <geom name="pusher_geom" type="sphere" size="0.02" material="pusher" mass="0.2" friction="0.2 0.02 0.01"/>
      <site name="pusher_site" pos="0 0 0" size="0.01"/>
    </body>

{object_body_block}
  </worldbody>

  <actuator>
    <position name="pusher_x_ctrl" joint="pusher_x" ctrlrange="-0.26 0.26"/>
    <position name="pusher_y_ctrl" joint="pusher_y" ctrlrange="-0.26 0.26"/>
  </actuator>
</mujoco>
'''
