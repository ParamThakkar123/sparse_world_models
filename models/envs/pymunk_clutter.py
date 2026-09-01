"""Chipmunk2D high-friction clutter domain -- a third independent contact solver.

Why this environment exists
---------------------------
This is the second of the two third-party-engine domains added to test whether the momentum
shortcut is a property of physical pushing tasks or a property of our simulators. Where
:mod:`models.envs.box2d_billiards` picks the regime that should make the shortcut
*strongest* (near-elastic, long post-contact motion), this one picks the regime that should
make it *weakest* among plausible manipulation settings: high friction, heavy damping, dense
clutter, so objects stop almost as soon as the pusher leaves them and most change is a
short, contact-chained shove.

The two therefore bracket the manipulation regime from both ends, on two engines that share
no code with each other or with us. **The pre-registered prediction is that the trivial
"already moving" rule still beats the learned gate here** -- less decisively than in the
billiards domain, but decisively enough that the ranking is unchanged. If it did not, the
shortcut would be a property of long post-contact motion specifically, and the claim would
have to narrow to that.

Chipmunk (via pymunk) differs from Box2D and MuJoCo in ways that matter for this test: it
uses its own sequential-impulse solver with elastic-iteration handling, resolves friction
with a different tangent model, and applies damping as a per-second velocity retention
factor rather than a per-step multiplier. Objects here are true polygons with real
corner-to-face contacts, unlike ``planar_push``'s discs-with-a-modelled-lever.

Units and the internal scale factor
-----------------------------------
Chipmunk's ``collision_slop`` defaults to 0.1 world units -- twice the size of one of our
5 cm objects, which would let objects sink through each other. Rather than only shrinking
the tolerance (which makes the solver stiff and jittery at small scale), the world runs at
``_SCALE`` = 10x internally *and* sets an explicit slop, and the boundary converts back. So
externally this env speaks the same units as every other domain and the shared 2 cm motion
threshold applies unchanged.

Interface
---------
Exposes the observation dictionary and the
``snapshot`` / ``restore`` / ``relocate_object`` / ``set_planar_state`` API of
:mod:`models.envs.mujoco_tabletop`, so the whole pipeline runs against it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard only
    import pymunk
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pymunk_clutter requires the 'pymunk' package (pip install pymunk)."
    ) from exc


OBJECT_HALF_EXTENT = 0.025
PUSHER_RADIUS = 0.02

# See the module docstring: Chipmunk's absolute tolerances are an order of magnitude too
# coarse at our object scale, so the world runs 10x larger internally.
_SCALE = 10.0


@dataclass
class PymunkClutterConfig:
    num_objects: int = 3
    # CALIBRATED, not chosen: with the default damping below, this is the value at which
    # this domain's changed-object fraction (0.367) matches the MuJoCo tabletop's (0.366)
    # and its median moved-object displacement (0.0148) lands between the planar domain's
    # (0.0133) and the tabletop's (0.0254). Matching the *sparsity of change* is what makes
    # a cross-domain comparison a comparison of engines rather than of task difficulty.
    action_scale: float = 0.06
    pusher_bounds: tuple[float, float] = (-0.26, 0.26)
    object_bounds: tuple[float, float] = (-0.26, 0.26)
    # Deliberately tighter than the other domains' 0.09: clutter is the point here, so
    # pushes chain through neighbours rather than moving one object in isolation.
    min_object_separation: float = 0.07
    goal_clearance: float = 0.1
    goal_xy: tuple[float, float] = (0.18, 0.18)
    target_object: int = 0
    max_steps: int = 200
    seed: int | None = None
    # Regime knobs: inelastic, frictional, strongly damped. Chipmunk's ``damping`` is the
    # fraction of velocity RETAINED PER SECOND, so smaller means MORE damping; at 0.8 over a
    # 0.1 s control step an object retains ~0.98 of its speed while in contact but coasts
    # only a step or two after losing it. Calibrated jointly with ``action_scale`` -- see the
    # note there. ``mass`` is deliberately irrelevant to the dynamics (the pusher is
    # kinematic and all objects are equal-mass, so it cancels); it was swept and confirmed
    # to change nothing, and is kept only because Chipmunk requires a value.
    damping: float = 0.8
    friction: float = 0.5
    elasticity: float = 0.0
    mass: float = 0.2
    substeps: int = 6
    solver_iterations: int = 12
    dt: float = 1.0 / 60.0
    collision_slop: float = 0.002
    rest_speed: float = 1e-5

    # ---------------------------------------------------------------- placement
    # ``random`` scatters objects subject to ``min_object_separation`` -- the setting every
    # other domain uses, and the one the shortcut audit showed is trivially solvable: with a
    # point-like pusher and separated objects, the object that moves is always the one nearest
    # the pusher, so a zero-parameter nearest-object rule reaches F1 0.925.
    #
    # ``chain`` instead places the objects in a near-touching line running from the target
    # object toward the goal. Because the scripted policy pushes the target *along* that line,
    # a push propagates down it, and two things become true that no distance rule can
    # represent:
    #   * several objects sit at nearly the same distance from the pusher, so "nearest" stops
    #     identifying the mover;
    #   * objects the pusher never comes close to still move, via transferred impulse.
    # Chains are otherwise vanishingly rare -- measured at 0.00-0.05% of steps across all four
    # domains and every packing density from 0.09 down to 0.055 -- so they have to be designed
    # into the scene rather than filtered for.
    placement: str = "random"
    # Centre-to-centre spacing along the chain. Objects are 0.05 across, so this leaves a
    # 5 mm gap: close enough that one control step of pusher motion carries the contact down
    # the line, far enough that they start genuinely separate and at rest.
    chain_spacing: float = 0.055
    # How many objects join the chain; the rest are scattered as usual so the scene is not
    # degenerate in the opposite direction (a pure line with nothing else in it).
    chain_length: int = 4


class PymunkClutterEnv:
    """High-friction cluttered pushing on Chipmunk2D, with the tabletop env's interface."""

    def __init__(self, config: PymunkClutterConfig | None = None):
        self.config = config or PymunkClutterConfig()
        if self.config.num_objects < 1:
            raise ValueError("num_objects must be at least 1.")
        if not 0 <= self.config.target_object < self.config.num_objects:
            raise ValueError("target_object must index one of the configured objects.")
        self.rng = np.random.default_rng(self.config.seed)
        self.pusher_xy = np.zeros(2)
        self.step_count = 0
        self._bodies: list = []
        self._build_world()

    # ---------------------------------------------------------------- construction

    def _build_world(self) -> None:
        cfg = self.config
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = cfg.damping
        space.iterations = cfg.solver_iterations
        space.collision_slop = cfg.collision_slop * _SCALE

        lower, upper = cfg.object_bounds
        # Static boundary so a chained shove cannot eject an object off the table. Computed
        # entirely in external units and scaled once, at construction.
        span = (upper - lower) / 2.0 + OBJECT_HALF_EXTENT
        centre = (upper + lower) / 2.0
        corners = [
            (centre - span, centre - span),
            (centre - span, centre + span),
            (centre + span, centre + span),
            (centre + span, centre - span),
        ]
        for index in range(4):
            a = np.array(corners[index]) * _SCALE
            b = np.array(corners[(index + 1) % 4]) * _SCALE
            segment = pymunk.Segment(space.static_body, tuple(a), tuple(b), 0.01 * _SCALE)
            segment.friction = cfg.friction
            segment.elasticity = cfg.elasticity
            space.add(segment)

        size = (2 * OBJECT_HALF_EXTENT * _SCALE, 2 * OBJECT_HALF_EXTENT * _SCALE)
        moment = pymunk.moment_for_box(cfg.mass, size)
        for _ in range(cfg.num_objects):
            body = pymunk.Body(cfg.mass, moment)
            body.position = (0.0, 0.0)
            shape = pymunk.Poly.create_box(body, size)
            shape.friction = cfg.friction
            shape.elasticity = cfg.elasticity
            space.add(body, shape)
            self._bodies.append(body)

        # KINEMATIC pusher: driven by us, never by the scene, so it is an exogenous input
        # exactly as in the other three domains.
        pusher = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        pusher.position = (0.0, 0.0)
        pusher_shape = pymunk.Circle(pusher, PUSHER_RADIUS * _SCALE)
        pusher_shape.friction = cfg.friction
        pusher_shape.elasticity = cfg.elasticity
        space.add(pusher, pusher_shape)

        self._space = space
        self._pusher_body = pusher

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> dict[str, np.ndarray]:
        self.step_count = 0
        self.pusher_xy = np.array([0.0, -0.22])
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.velocity = (0.0, 0.0)

        placements = (
            self._chain_placements() if self.config.placement == "chain"
            else self._random_placements()
        )
        for body, xy in zip(self._bodies, placements):
            body.position = tuple(xy * _SCALE)
            body.angle = float(self.rng.uniform(-np.pi, np.pi))
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0
        # Chipmunk caches contact information lazily; one zero-length settle keeps the first
        # observation consistent with what a later restore() of the same state would give.
        self._space.step(1e-9)
        return self.get_observation()

    def _random_placements(self) -> list[np.ndarray]:
        placements: list[np.ndarray] = []
        for _ in range(self.config.num_objects):
            placements.append(self._sample_free_xy(placements))
        return placements

    def _chain_placements(self) -> list[np.ndarray]:
        """Target object plus a near-touching line running from it toward the goal.

        The line points from the target to the goal because that is the direction the
        scripted policy pushes, so the push travels down the chain rather than across it. The
        target is sampled far enough from the goal that the whole chain fits on the table;
        if a chain slot would fall out of bounds the line is truncated there and the
        remaining objects are scattered, so this can never fail to produce a scene.
        """
        config = self.config
        lower, upper = config.object_bounds
        goal = np.asarray(config.goal_xy)

        for _ in range(500):
            target = self._sample_free_xy([])
            direction = goal - target
            norm = float(np.linalg.norm(direction))
            if norm < 0.15:  # too close to the goal to lay a chain along the approach
                continue
            direction = direction / norm
            placements = [target]
            for step in range(1, min(config.chain_length, config.num_objects)):
                candidate = target + direction * config.chain_spacing * step
                if np.any(np.abs(candidate) > upper) or np.any(candidate < lower):
                    break
                if np.linalg.norm(candidate - np.array([0.0, -0.22])) < 0.1:
                    break
                placements.append(candidate)
            if len(placements) >= 2:
                break
        else:  # pragma: no cover - only reachable if the table is pathologically small
            return self._random_placements()

        # Remaining objects are scattered normally, so the scene still contains distractors
        # that the chain does not reach -- otherwise "everything moves" would be the new
        # trivial rule.
        while len(placements) < config.num_objects:
            placements.append(self._sample_free_xy(placements))
        return placements

    def _sample_free_xy(self, placements: list[np.ndarray]) -> np.ndarray:
        lower, upper = self.config.object_bounds
        goal = np.asarray(self.config.goal_xy)
        for _ in range(4000):
            xy = self.rng.uniform(lower, upper, size=2)
            if np.linalg.norm(xy - np.array([0.0, -0.22])) < 0.1:
                continue
            if np.linalg.norm(xy - goal) < self.config.goal_clearance:
                continue
            if all(
                np.linalg.norm(xy - other) > self.config.min_object_separation
                for other in placements
            ):
                return xy
        raise RuntimeError("Could not sample a valid non-overlapping object placement.")

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        cfg = self.config
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        target = np.clip(self.pusher_xy + action * cfg.action_scale, *cfg.pusher_bounds)

        # A kinematic body must be driven by velocity: teleporting its position generates no
        # contact impulse and objects would be passed straight through.
        stride = (target - self.pusher_xy) / cfg.substeps
        self._pusher_body.velocity = tuple(stride / cfg.dt * _SCALE)
        for _ in range(cfg.substeps):
            self._space.step(cfg.dt)
        self._pusher_body.velocity = (0.0, 0.0)
        self.pusher_xy = target
        self._pusher_body.position = tuple(target * _SCALE)

        self._snap_resting_objects()

        self.step_count += 1
        obs = self.get_observation()
        distance = self._target_distance(obs)
        success = distance < 0.05
        done = success or self.step_count >= cfg.max_steps
        return obs, -distance, done, {"success": success, "target_distance": distance}

    def _snap_resting_objects(self) -> None:
        """Zero velocities below the change threshold's resolution.

        Without this the solver's residual jitter leaves every object permanently at ~1e-7
        velocity, which would empty the ``at_rest`` population and silently delete the onset
        events this whole study is about.
        """
        threshold = self.config.rest_speed * _SCALE
        for body in self._bodies:
            vx, vy = body.velocity
            if (vx * vx + vy * vy) ** 0.5 < threshold:
                body.velocity = (0.0, 0.0)
            if abs(body.angular_velocity) < threshold:
                body.angular_velocity = 0.0

    # ---------------------------------------------------------------- observation

    def get_observation(self) -> dict[str, np.ndarray]:
        n = self.config.num_objects
        xy = np.array([[b.position[0], b.position[1]] for b in self._bodies]) / _SCALE
        yaw = np.array([b.angle for b in self._bodies], dtype=np.float64)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        vel = np.array([[b.velocity[0], b.velocity[1]] for b in self._bodies]) / _SCALE
        omega = np.array([b.angular_velocity for b in self._bodies], dtype=np.float64)

        half = yaw / 2.0
        quaternions = np.stack(
            [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1
        )
        poses = np.concatenate([xy, np.full((n, 1), 0.105), quaternions], axis=1)
        velocities = np.zeros((n, 6))
        velocities[:, 2] = omega
        velocities[:, 3:5] = vel
        return {
            "pusher_xy": self.pusher_xy.copy(),
            "object_poses": poses,
            "object_velocities": velocities,
            "goal_xy": np.asarray(self.config.goal_xy, dtype=np.float64),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        obs = self.get_observation()
        yaw = np.array([b.angle for b in self._bodies], dtype=np.float64)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        planar = np.column_stack((obs["object_poses"][:, 0], obs["object_poses"][:, 1], yaw))
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
        """Position/velocity capture.

        Chipmunk keeps cached contact impulses for warm starting, and pymunk exposes no way
        to read or restore them, so a restored rollout is not bit-identical to an
        uninterrupted one. **Measured**, at 3 objects after 5 settling steps: the restored
        step differs from the original by **1.1e-4 m**, and two successive restores differ
        from each other by 6.8e-6 m. (Box2D, for comparison, comes in at 6.9e-7 m and is
        exactly reproducible between restores.)

        That residual is 250x smaller than this domain's 0.029 m motion threshold, so it
        cannot flip a change label, and it is quoted here rather than described as "small" so
        the claim can be checked. The counterfactual-validity machinery would surface a real
        failure independently, as a non-unit ``others_match_fraction``.
        """
        return {
            "pusher_xy": self.pusher_xy.copy(),
            "object_xy": np.array([[b.position[0], b.position[1]] for b in self._bodies]),
            "object_angle": np.array([b.angle for b in self._bodies]),
            "object_vel": np.array([[b.velocity[0], b.velocity[1]] for b in self._bodies]),
            "object_omega": np.array([b.angular_velocity for b in self._bodies]),
            "step_count": int(self.step_count),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.pusher_xy = snapshot["pusher_xy"].copy()
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.velocity = (0.0, 0.0)
        for index, body in enumerate(self._bodies):
            body.position = tuple(snapshot["object_xy"][index])
            body.angle = float(snapshot["object_angle"][index])
            body.velocity = tuple(snapshot["object_vel"][index])
            body.angular_velocity = float(snapshot["object_omega"][index])
        self.step_count = int(snapshot["step_count"])
        self._space.reindex_static()

    def relocate_object(self, index: int, xy: np.ndarray, yaw: float | None = None) -> None:
        """Move one object to a new resting pose, leaving everything else untouched."""
        if not 0 <= index < self.config.num_objects:
            raise IndexError(f"object index {index} out of range")
        body = self._bodies[index]
        body.position = tuple(np.asarray(xy, dtype=np.float64).reshape(2) * _SCALE)
        if yaw is not None:
            body.angle = float(yaw)
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0

    def set_planar_state(
        self,
        pusher_xy: np.ndarray,
        object_pose: np.ndarray,
        object_velocity: np.ndarray | None = None,
    ) -> None:
        """Drive the sim to an arbitrary planar configuration."""
        pose = np.asarray(object_pose, dtype=np.float64).reshape(-1, 3)
        if pose.shape[0] != self.config.num_objects:
            raise ValueError(
                f"object_pose has {pose.shape[0]} rows but the env has {self.config.num_objects}."
            )
        self.pusher_xy = np.asarray(pusher_xy, dtype=np.float64).reshape(2).copy()
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.velocity = (0.0, 0.0)
        velocity = (
            None
            if object_velocity is None
            else np.asarray(object_velocity, dtype=np.float64).reshape(-1, 6)
        )
        for index, body in enumerate(self._bodies):
            body.position = tuple(pose[index, :2] * _SCALE)
            body.angle = float(pose[index, 2])
            if velocity is None:
                body.velocity = (0.0, 0.0)
                body.angular_velocity = 0.0
            else:
                body.velocity = tuple(velocity[index, 3:5] * _SCALE)
                body.angular_velocity = float(velocity[index, 2])
