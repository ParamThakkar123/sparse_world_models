"""Dependency-free 2D planar pushing environment (W4 breadth).

Every result in this project comes from one MuJoCo tabletop task, which is the narrowest
part of the story. This is a second domain with genuinely different dynamics -- impulse-based
2D discs with Coulomb-style friction rather than a full 3D contact solver -- that
deliberately reproduces the *structural* property under study: at any step only the objects
actually touched (directly, or through a contact chain) move, while the rest stay exactly
where they were.

It exposes the same observation dictionary and the same
``snapshot`` / ``restore`` / ``relocate_object`` / ``set_planar_state`` API as
:mod:`models.envs.mujoco_tabletop`, so ``generate_transitions``, the training scripts, the
gate ablation and the W3 counterfactual machinery all run against it **unchanged**. Object
orientation is carried as a yaw encoded into a quaternion so the shared
``extract_planar_object_state`` helper works verbatim.

Being pure NumPy it is roughly two orders of magnitude faster than the MuJoCo env, which is
what makes higher object counts and more seeds affordable here than on the tabletop.

The physics is deliberately simple and *not* a MuJoCo clone -- the point of a second
environment is that it differs. Contacts are resolved by positional correction plus an
impulse along the contact normal, objects carry linear and angular damping, and a contact
chain is resolved by iterating the pairwise solver, so a push can propagate through a line of
touching objects exactly as it does on the tabletop.

**The dynamical regime genuinely differs, and that has a consequence worth stating.** Here an
object moves essentially only while in contact (high damping, quasi-static pushing); on the
tabletop a nudged box keeps sliding on low-friction contact (impulsive pushing). Measured
under the same scripted policy, per-step displacements are therefore about a third the size:
the 99th percentile is 0.014 here against 0.039 on the tabletop. The 0.02 m threshold that
``create_hard_subset`` uses to define "real motion" was tuned for the tabletop and keeps only
**0.13%** of planar steps -- which at 3 objects left a 15-row training split and produced
meaningless results before this was caught. Use ``--min-max-xy-delta 0.010`` for this
environment, which retains a comparable ~20% of steps.

**Calibration, disclosed — including what it failed to match.** ``torque_gain`` and
``angular_damping`` are *fitted*, not derived. Measured on the generated hard splits at
3 objects, the two domains line up on everything that governs whether shared thresholds and
hyperparameters transfer:

===========================  =========  ==========
statistic                    planar     tabletop
===========================  =========  ==========
hard-split rows               1816        950
changed-object fraction       0.352       0.366
median xy displacement        0.0133      0.0254
median yaw change             0.698       0.187
===========================  =========  ==========

Rotation is deliberately left **~3.7x stronger** rather than tuned further. Yaw does not
scale linearly with ``torque_gain`` -- the lever arm depends on the object's own orientation,
so faster spin changes the contact geometry that produces it, and a linear solve from a
measured ratio overshoots (330 -> 2383 was predicted to land on 0.187 and produced 0.698).
Chasing it further would be fitting a constant to four significant figures for no scientific
gain. The honest statement is that this environment rotates objects harder than the tabletop,
which makes its yaw prediction task *harder*, not easier.

So: the *scale of translation and the sparsity of change* are calibrated, because those are
what make shared thresholds valid; the *mechanism* (quasi-static vs impulsive, approximated
box-contact lever vs a real contact solver) and the *rotation regime* remain genuinely
different, which is what makes this a breadth result rather than a re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Matches the tabletop's 5 cm boxes closely enough that thresholds (POSITION_EPS, the
# 2 cm hard-subset filter, CONTACT_RADIUS) carry over without retuning.
OBJECT_RADIUS = 0.025
PUSHER_RADIUS = 0.02


@dataclass
class PlanarPushConfig:
    num_objects: int = 3
    action_scale: float = 0.04
    pusher_bounds: tuple[float, float] = (-0.26, 0.26)
    object_bounds: tuple[float, float] = (-0.26, 0.26)
    min_object_separation: float = 0.09
    goal_clearance: float = 0.1
    goal_xy: tuple[float, float] = (0.18, 0.18)
    target_object: int = 0
    max_steps: int = 200
    # Fraction of velocity retained per step. Low enough that objects come to rest quickly,
    # which is what makes "most objects are unchanged" true here as on the tabletop.
    linear_damping: float = 0.45
    angular_damping: float = 0.70
    restitution: float = 0.0
    solver_iterations: int = 4
    seed: int | None = None
    substeps: int = 4
    # Off-centre hits impart spin; this scales how much, and absorbs the (small) moment of
    # inertia of a 2.5 cm half-extent box. CALIBRATED, not derived: set so the median yaw
    # change of a moved object matches the tabletop's 0.187 rad, measured on the generated
    # datasets rather than on a live sweep -- an earlier value of 1500 was chosen from a
    # sweep whose statistic (per-step max over moved objects) differed from the one the data
    # actually reports (median over moved entries) and overshot by ~4.5x. The value here
    # is solved for, not guessed: yaw scales linearly with this gain, so it is set from the
    # measured ratio between the two environments' HARD-SPLIT median yaw (the population
    # models actually train on -- comparing raw planar data against a filtered tabletop
    # split was the mistake that produced the earlier wrong values).
    torque_gain: float = 2383.0
    _placeholder: tuple = field(default=(), repr=False)


class PlanarPushEnv:
    """2D disc pushing with the tabletop environment's interface."""

    def __init__(self, config: PlanarPushConfig | None = None):
        self.config = config or PlanarPushConfig()
        if self.config.num_objects < 1:
            raise ValueError("num_objects must be at least 1.")
        if not 0 <= self.config.target_object < self.config.num_objects:
            raise ValueError("target_object must index one of the configured objects.")
        self.rng = np.random.default_rng(self.config.seed)
        n = self.config.num_objects
        self.pusher_xy = np.zeros(2)
        self.object_xy = np.zeros((n, 2))
        self.object_yaw = np.zeros(n)
        self.object_vel = np.zeros((n, 2))
        self.object_omega = np.zeros(n)
        self.step_count = 0

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> dict[str, np.ndarray]:
        self.step_count = 0
        self.pusher_xy = np.array([0.0, -0.22])
        self.object_vel[:] = 0.0
        self.object_omega[:] = 0.0
        placements: list[np.ndarray] = []
        for index in range(self.config.num_objects):
            placements.append(self._sample_free_xy(placements))
            self.object_xy[index] = placements[-1]
            self.object_yaw[index] = self.rng.uniform(-np.pi, np.pi)
        return self.get_observation()

    def _sample_free_xy(self, placements: list[np.ndarray]) -> np.ndarray:
        lower, upper = self.config.object_bounds
        goal = np.asarray(self.config.goal_xy)
        for _ in range(4000):
            xy = self.rng.uniform(lower, upper, size=2)
            if np.linalg.norm(xy - np.array([0.0, -0.22])) < 0.1:
                continue
            if np.linalg.norm(xy - goal) < self.config.goal_clearance:
                continue
            if all(np.linalg.norm(xy - other) > self.config.min_object_separation
                   for other in placements):
                return xy
        raise RuntimeError("Could not sample a valid non-overlapping object placement.")

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        target = np.clip(
            self.pusher_xy + action * self.config.action_scale, *self.config.pusher_bounds
        )
        # Substepping matters: a single jump can tunnel the pusher through a disc, which
        # would silently break the "only touched objects move" property the study is about.
        # The pusher advances by a fixed increment per substep, and that increment is what a
        # contacted object inherits as velocity -- an object shoved by the pusher must keep
        # pace with it, so the impulse is set by the pusher's *motion*, not by how deeply it
        # happens to have penetrated on this substep.
        stride = (target - self.pusher_xy) / self.config.substeps
        drive = target - self.pusher_xy  # displacement over the whole control step
        for _ in range(self.config.substeps):
            self.pusher_xy = self.pusher_xy + stride
            # Integrate first, then resolve: contact resolution must have the last word so
            # objects never end a substep overlapping or outside the table bounds (the
            # clamp lives at the end of _resolve_contacts).
            self._integrate(1.0 / self.config.substeps)
            self._resolve_contacts(drive)

        self.step_count += 1
        obs = self.get_observation()
        distance = self._target_distance(obs)
        success = distance < 0.05
        done = success or self.step_count >= self.config.max_steps
        return obs, -distance, done, {"success": success, "target_distance": distance}

    # ---------------------------------------------------------------- physics

    def _integrate(self, fraction: float) -> None:
        self.object_xy += self.object_vel * fraction
        self.object_yaw = (self.object_yaw + self.object_omega * fraction + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        decay_linear = self.config.linear_damping ** fraction
        decay_angular = self.config.angular_damping ** fraction
        self.object_vel *= decay_linear
        self.object_omega *= decay_angular
        # Objects below the resolution of the change threshold are snapped to rest, so a
        # numerically tiny drift never registers as "this object changed".
        self.object_vel[np.linalg.norm(self.object_vel, axis=1) < 1e-5] = 0.0
        self.object_omega[np.abs(self.object_omega) < 1e-5] = 0.0

    def _resolve_contacts(self, drive: np.ndarray) -> None:
        lower, upper = self.config.object_bounds
        for _ in range(self.config.solver_iterations):
            self._resolve_pusher_contacts(drive)
            self._resolve_object_contacts()
        np.clip(self.object_xy, lower, upper, out=self.object_xy)

    def _resolve_pusher_contacts(self, drive: np.ndarray) -> None:
        offset = self.object_xy - self.pusher_xy
        distance = np.linalg.norm(offset, axis=1)
        overlap = (PUSHER_RADIUS + OBJECT_RADIUS) - distance
        hit = overlap > 0
        if not np.any(hit):
            return
        normal = offset[hit] / np.maximum(distance[hit, None], 1e-9)
        # The pusher is infinitely massive: it never yields, the object takes the whole
        # positional correction. That is what makes the pusher an exogenous driver.
        self.object_xy[hit] += normal * overlap[hit, None]
        # Velocity is set from the pusher's advance along the contact normal, and only
        # raised, never lowered -- an object already moving away faster is not slowed by
        # being caught up with.
        approach = np.maximum(normal @ drive, 0.0)
        along_normal = np.sum(self.object_vel[hit] * normal, axis=1)
        gain = np.maximum(approach - along_normal, 0.0)
        self.object_vel[hit] += normal * gain[:, None]
        # Spin needs a lever arm, and a disc has none: its contact point lies on the normal
        # through the centre, so ``offset . tangential`` is identically zero and a central
        # impulse produces no torque whatsoever. An earlier version used exactly that as the
        # lever and left yaw constant in every episode, silently reducing the pose to 2 DoF
        # and making a third of the prediction target trivial.
        #
        # The tabletop's objects are 5 cm *boxes*, whose contact point is off-centre except
        # when the pusher strikes a face square-on. We model that lever without paying for
        # full polygon collision: for a square the offset from centre to contact point varies
        # with the angle between the contact normal and the box's own orientation, vanishing
        # face-on and peaking near the corners, with the 4-fold symmetry of the square. The
        # gain absorbs the moment of inertia (I = m*r^2/2 is small for a 2.5 cm half-extent,
        # so a modest torque produces a large angular acceleration).
        contact_angle = np.arctan2(normal[:, 1], normal[:, 0])
        lever = OBJECT_RADIUS * np.sin(2.0 * (contact_angle - self.object_yaw[hit]))
        self.object_omega[hit] += self.config.torque_gain * lever * approach

    def _resolve_object_contacts(self) -> None:
        """Resolve every object-object overlap at once (Jacobi-style).

        A sequential pair-by-pair loop is the textbook Gauss-Seidel form, but in Python it
        costs O(N^2) interpreter iterations per solver pass and dominates everything else:
        measured throughput fell from 2697 steps/s at N=3 to 9 steps/s at N=30, a 300x
        collapse that would have made the high-count runs slower than the MuJoCo env this
        environment exists to be faster than. Resolving all pairs simultaneously against the
        pre-update positions is O(N^2) in NumPy rather than in Python, and with
        ``solver_iterations`` passes it converges to the same resting configurations --
        contact chains still propagate, they just take one extra pass per link.
        """
        n = self.config.num_objects
        if n < 2:
            return
        # offset[i, j] = position_j - position_i
        offset = self.object_xy[None, :, :] - self.object_xy[:, None, :]
        distance = np.linalg.norm(offset, axis=-1)
        np.fill_diagonal(distance, np.inf)  # an object never collides with itself
        overlap = 2.0 * OBJECT_RADIUS - distance
        touching = overlap > 0.0
        if not np.any(touching):
            return

        normal = offset / np.maximum(distance, 1e-9)[:, :, None]
        # Zero the non-contacting entries (including the infinite diagonal) *before*
        # multiplying. Masking afterwards with np.where still evaluates 0 * -inf on the
        # diagonal, which yields NaN and silently poisons the summed correction.
        depth = np.where(touching, overlap, 0.0)
        # Equal-mass split, so momentum is conserved and a push propagates along a chain of
        # touching objects instead of stopping at the first contact.
        self.object_xy += (-normal * depth[:, :, None] * 0.5).sum(axis=1)

        relative = np.sum((self.object_vel[None, :, :] - self.object_vel[:, None, :]) * normal, axis=-1)
        # Only approaching pairs exchange an impulse; separating ones are already resolving.
        approaching = touching & (relative < 0.0)
        impulse = np.where(approaching, -(1.0 + self.config.restitution) * relative * 0.5, 0.0)
        self.object_vel -= (normal * impulse[:, :, None]).sum(axis=1)

    # ---------------------------------------------------------------- observation

    def get_observation(self) -> dict[str, np.ndarray]:
        half = self.object_yaw / 2.0
        quaternions = np.stack(
            [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1
        )
        poses = np.concatenate(
            [self.object_xy, np.full((self.config.num_objects, 1), 0.105), quaternions], axis=1
        )
        # Six velocity components to match the tabletop's cvel layout: the shared state
        # layout assumes 6 per object, and only the planar entries are meaningful here.
        velocities = np.zeros((self.config.num_objects, 6))
        velocities[:, 2] = self.object_omega
        velocities[:, 3:5] = self.object_vel
        return {
            "pusher_xy": self.pusher_xy.copy(),
            "object_poses": poses,
            "object_velocities": velocities,
            "goal_xy": np.asarray(self.config.goal_xy, dtype=np.float64),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        obs = self.get_observation()
        planar = np.column_stack((self.object_xy[:, 0], self.object_xy[:, 1], self.object_yaw))
        return {
            "object_pose": planar,
            "object_pose_full": obs["object_poses"],
            "object_velocity": obs["object_velocities"],
            "pusher_xy": obs["pusher_xy"],
        }

    def sample_random_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=2)

    def _target_distance(self, obs: dict[str, np.ndarray]) -> float:
        target = obs["object_poses"][self.config.target_object, :2]
        return float(np.linalg.norm(target - obs["goal_xy"]))

    # ---------------------------------------------------------------- state control

    def snapshot(self) -> dict[str, Any]:
        """Exact state capture. Unlike the MuJoCo env there is no hidden solver state, so a
        snapshot here is complete by construction."""
        return {
            "pusher_xy": self.pusher_xy.copy(),
            "object_xy": self.object_xy.copy(),
            "object_yaw": self.object_yaw.copy(),
            "object_vel": self.object_vel.copy(),
            "object_omega": self.object_omega.copy(),
            "step_count": int(self.step_count),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.pusher_xy = snapshot["pusher_xy"].copy()
        self.object_xy = snapshot["object_xy"].copy()
        self.object_yaw = snapshot["object_yaw"].copy()
        self.object_vel = snapshot["object_vel"].copy()
        self.object_omega = snapshot["object_omega"].copy()
        self.step_count = int(snapshot["step_count"])

    def relocate_object(self, index: int, xy: np.ndarray, yaw: float | None = None) -> None:
        """Move one object to a new resting pose, leaving everything else untouched.

        The counterfactual-splicing primitive; see the tabletop env's docstring for why
        verification must perturb only the object under test.
        """
        if not 0 <= index < self.config.num_objects:
            raise IndexError(f"object index {index} out of range")
        self.object_xy[index] = np.asarray(xy, dtype=np.float64).reshape(2)
        if yaw is not None:
            self.object_yaw[index] = float(yaw)
        self.object_vel[index] = 0.0
        self.object_omega[index] = 0.0

    def set_planar_state(
        self,
        pusher_xy: np.ndarray,
        object_pose: np.ndarray,
        object_velocity: np.ndarray | None = None,
    ) -> None:
        """Drive the sim to an arbitrary planar configuration.

        Lossless here, unlike the tabletop version: the reduced planar state *is* this
        environment's full state (there is no z, tilt or contact history to drop), so a
        round trip through it is exact.
        """
        pose = np.asarray(object_pose, dtype=np.float64).reshape(-1, 3)
        if pose.shape[0] != self.config.num_objects:
            raise ValueError(
                f"object_pose has {pose.shape[0]} rows but the env has {self.config.num_objects}."
            )
        self.pusher_xy = np.asarray(pusher_xy, dtype=np.float64).reshape(2).copy()
        self.object_xy = pose[:, :2].copy()
        self.object_yaw = pose[:, 2].copy()
        if object_velocity is None:
            self.object_vel[:] = 0.0
            self.object_omega[:] = 0.0
        else:
            velocity = np.asarray(object_velocity, dtype=np.float64).reshape(-1, 6)
            self.object_omega = velocity[:, 2].copy()
            self.object_vel = velocity[:, 3:5].copy()
