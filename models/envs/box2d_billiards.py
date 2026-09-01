"""Box2D elastic-collision domain -- the momentum shortcut's best case.

Why this environment exists
---------------------------
The momentum-shortcut finding (``experiments/momentum_shortcut.py``) says that
change-detection benchmarks are won by a one-line rule -- "this object will change iff it
is already moving" -- because the filtered evaluation population is dominated by objects
*continuing* to move rather than *starting* to. That finding was measured in two
environments, and both of them are ours: the MuJoCo tabletop and ``planar_push``. The first
objection any reviewer raises is that the shortcut is a property of how we wrote our
simulators.

This domain answers that objection from one side and sharpens the claim from another. It is
built on **Box2D** -- an independent, widely-used, sequential-impulse solver with its own
integrator, its own contact model and none of our code in it -- so agreement here cannot be
a shared-implementation artefact. And it deliberately picks the dynamical regime that should
make the shortcut *strongest*: near-elastic collisions (``restitution`` 0.85) with light
damping, so a struck object keeps moving for tens of steps afterwards. If the shortcut is
real rather than an artefact of quasi-static pushing, this is where it should be most
extreme, and the trivial rule should approach F1 1.0.

That prediction is recorded here before the measurement, in the same style as the rest of
the project: **the trivial rule should score higher here than in either existing domain, and
the learned gate's deficit against it should be the largest we have measured.**

Regime summary across the four domains now in the suite::

    domain            engine     contact       post-contact motion   change is
    ----------------  ---------  ------------  -------------------  ------------------
    mujoco_tabletop   MuJoCo     impulsive 3D  short slide          contact-driven
    planar_push       ours       quasi-static  stops immediately    contact-driven
    box2d_billiards   Box2D      near-elastic  very long            contact + rebound
    pymunk_clutter    Chipmunk2D high-friction short, chained       contact chains

Units and the internal scale factor
-----------------------------------
Box2D's solver is tuned for objects roughly 0.1-10 m across: ``b2_linearSlop`` is 5 mm and
``b2_velocityThreshold`` 1 m/s, both absolute. Our objects are 5 cm boxes, so at native
scale the allowed penetration would be 20% of an object and restitution would be silently
disabled below 1 m/s -- the elastic regime this domain exists to provide would not actually
happen. Everything inside the world therefore runs at ``_SCALE`` = 10x, and the boundary
(observations, actions, snapshots) converts back, so externally this env speaks exactly the
same units as the MuJoCo tabletop and the same thresholds apply unchanged.

Interface
---------
Exposes the observation dictionary and the
``snapshot`` / ``restore`` / ``relocate_object`` / ``set_planar_state`` API of
:mod:`models.envs.mujoco_tabletop`, so ``generate_transitions``, the training scripts, the
gate ablation and the counterfactual machinery run against it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard only
    from Box2D import b2World
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "box2d_billiards requires the 'Box2D' package (pip install box2d-py)."
    ) from exc


# Matches the tabletop's 5 cm boxes and the planar env's 2.5 cm-radius discs, so the 2 cm
# hard-subset motion filter and POSITION_EPS carry over without retuning.
OBJECT_HALF_EXTENT = 0.025
PUSHER_RADIUS = 0.02

# See the module docstring: Box2D's absolute tolerances are wrong by an order of magnitude
# at our object scale, so the world runs 10x larger internally.
_SCALE = 10.0


@dataclass
class Box2DBilliardsConfig:
    num_objects: int = 3
    action_scale: float = 0.04
    pusher_bounds: tuple[float, float] = (-0.26, 0.26)
    object_bounds: tuple[float, float] = (-0.26, 0.26)
    min_object_separation: float = 0.09
    goal_clearance: float = 0.1
    goal_xy: tuple[float, float] = (0.18, 0.18)
    target_object: int = 0
    max_steps: int = 200
    seed: int | None = None
    # The regime knobs. High restitution + low damping is what makes this the momentum
    # shortcut's best case: a struck object rebounds off walls and neighbours and stays in
    # motion for tens of steps, so P(changed | already moving) should be near 1.
    restitution: float = 0.85
    friction: float = 0.05
    linear_damping: float = 0.15
    angular_damping: float = 0.15
    density: float = 1.0
    # Box2D substeps per control step. The pusher advances in increments so it cannot
    # tunnel through an object, which would break the "only touched objects move" property.
    substeps: int = 6
    velocity_iterations: int = 8
    position_iterations: int = 3
    dt: float = 1.0 / 60.0
    # Below this planar speed an object is snapped to rest, so numerically tiny drift never
    # registers as a change. Mirrors planar_push's 1e-5 in external units.
    rest_speed: float = 1e-5


class Box2DBilliardsEnv:
    """Near-elastic top-down pushing on Box2D, with the tabletop environment's interface."""

    def __init__(self, config: Box2DBilliardsConfig | None = None):
        self.config = config or Box2DBilliardsConfig()
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
        self._world = b2World(gravity=(0.0, 0.0), doSleep=False)

        # Walls. Objects rebound off them, which is a second source of "already moving and
        # will keep moving" beyond object-object collisions -- and a source of change with
        # no nearby contact partner at all, which is exactly the kind of event a
        # contact-geometry featurisation cannot see and a momentum rule can.
        lower, upper = cfg.object_bounds
        # All wall geometry is computed in EXTERNAL units and scaled exactly once, at the
        # CreatePolygonFixture call. Mixing scaled and unscaled terms here is an easy and
        # near-invisible bug: the walls would land in the wrong place and objects would
        # either escape the table or be trapped in a box far too small.
        centre = (upper + lower) / 2.0
        half = (upper - lower) / 2.0 + OBJECT_HALF_EXTENT
        thickness = 0.05
        span = half + thickness
        wall = self._world.CreateStaticBody(position=(centre * _SCALE, centre * _SCALE))
        for dx, dy, hx, hy in (
            (0.0, span, span, thickness),
            (0.0, -span, span, thickness),
            (span, 0.0, thickness, span),
            (-span, 0.0, thickness, span),
        ):
            wall.CreatePolygonFixture(
                box=(hx * _SCALE, hy * _SCALE, (dx * _SCALE, dy * _SCALE), 0.0),
                friction=cfg.friction,
                restitution=cfg.restitution,
            )

        for _ in range(cfg.num_objects):
            body = self._world.CreateDynamicBody(
                position=(0.0, 0.0),
                linearDamping=cfg.linear_damping,
                angularDamping=cfg.angular_damping,
                bullet=True,  # continuous collision: fast objects must not tunnel
            )
            body.CreatePolygonFixture(
                box=(OBJECT_HALF_EXTENT * _SCALE, OBJECT_HALF_EXTENT * _SCALE),
                density=cfg.density,
                friction=cfg.friction,
                restitution=cfg.restitution,
            )
            self._bodies.append(body)

        # The pusher is KINEMATIC: it drives the scene and is never driven by it, which is
        # what makes it an exogenous input exactly as in the other three domains.
        pusher = self._world.CreateKinematicBody(position=(0.0, 0.0))
        pusher.CreateCircleFixture(
            radius=PUSHER_RADIUS * _SCALE, friction=cfg.friction, restitution=cfg.restitution
        )
        self._pusher_body = pusher

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> dict[str, np.ndarray]:
        self.step_count = 0
        self.pusher_xy = np.array([0.0, -0.22])
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.linearVelocity = (0.0, 0.0)

        placements: list[np.ndarray] = []
        for body in self._bodies:
            xy = self._sample_free_xy(placements)
            placements.append(xy)
            body.position = tuple(xy * _SCALE)
            body.angle = float(self.rng.uniform(-np.pi, np.pi))
            body.linearVelocity = (0.0, 0.0)
            body.angularVelocity = 0.0
            body.awake = True
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

        # A kinematic body is moved by setting its velocity, not its position: teleporting
        # it would generate no contact impulse and objects would be passed straight through.
        stride = (target - self.pusher_xy) / cfg.substeps
        velocity = stride / cfg.dt
        self._pusher_body.linearVelocity = tuple(velocity * _SCALE)
        for _ in range(cfg.substeps):
            self._world.Step(cfg.dt, cfg.velocity_iterations, cfg.position_iterations)
            self._world.ClearForces()
        self._pusher_body.linearVelocity = (0.0, 0.0)
        # Box2D integrates the kinematic body itself; read back rather than assume, then
        # keep our own copy authoritative so the clamp above is respected exactly.
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

        Without this, Box2D's residual jitter leaves every object with a ~1e-7 velocity
        forever, which would make ``at_rest`` empty and silently delete the onset population
        this whole study is about.
        """
        threshold = self.config.rest_speed * _SCALE
        for body in self._bodies:
            vx, vy = body.linearVelocity
            if (vx * vx + vy * vy) ** 0.5 < threshold:
                body.linearVelocity = (0.0, 0.0)
            if abs(body.angularVelocity) < threshold:
                body.angularVelocity = 0.0

    # ---------------------------------------------------------------- observation

    def get_observation(self) -> dict[str, np.ndarray]:
        n = self.config.num_objects
        xy = np.array([[b.position[0], b.position[1]] for b in self._bodies]) / _SCALE
        yaw = np.array([b.angle for b in self._bodies], dtype=np.float64)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        vel = np.array([[b.linearVelocity[0], b.linearVelocity[1]] for b in self._bodies]) / _SCALE
        omega = np.array([b.angularVelocity for b in self._bodies], dtype=np.float64)

        half = yaw / 2.0
        quaternions = np.stack(
            [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1
        )
        poses = np.concatenate([xy, np.full((n, 1), 0.105), quaternions], axis=1)
        # Six velocity components to match the shared cvel layout: angular first, then
        # linear; only the planar entries are meaningful in a 2D domain.
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
        planar = np.column_stack(
            (obs["object_poses"][:, 0], obs["object_poses"][:, 1], yaw)
        )
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
        """Exact state capture.

        Box2D keeps warm-starting impulses in its contact manager which this does *not*
        capture, so a restore is exact in position/velocity but may differ by one solver
        iteration's worth of warm start. That is disclosed rather than hidden: the
        counterfactual-validity machinery compares a restored rollout against a freshly
        stepped one, so any discrepancy surfaces there as a non-unit
        ``others_match_fraction`` rather than passing silently.
        """
        return {
            "pusher_xy": self.pusher_xy.copy(),
            "object_xy": np.array([[b.position[0], b.position[1]] for b in self._bodies]),
            "object_angle": np.array([b.angle for b in self._bodies]),
            "object_vel": np.array(
                [[b.linearVelocity[0], b.linearVelocity[1]] for b in self._bodies]
            ),
            "object_omega": np.array([b.angularVelocity for b in self._bodies]),
            "step_count": int(self.step_count),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.pusher_xy = snapshot["pusher_xy"].copy()
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.linearVelocity = (0.0, 0.0)
        for index, body in enumerate(self._bodies):
            body.position = tuple(snapshot["object_xy"][index])
            body.angle = float(snapshot["object_angle"][index])
            body.linearVelocity = tuple(snapshot["object_vel"][index])
            body.angularVelocity = float(snapshot["object_omega"][index])
            body.awake = True
        self.step_count = int(snapshot["step_count"])

    def relocate_object(self, index: int, xy: np.ndarray, yaw: float | None = None) -> None:
        """Move one object to a new resting pose, leaving everything else untouched."""
        if not 0 <= index < self.config.num_objects:
            raise IndexError(f"object index {index} out of range")
        body = self._bodies[index]
        body.position = tuple(np.asarray(xy, dtype=np.float64).reshape(2) * _SCALE)
        if yaw is not None:
            body.angle = float(yaw)
        body.linearVelocity = (0.0, 0.0)
        body.angularVelocity = 0.0
        body.awake = True

    def set_planar_state(
        self,
        pusher_xy: np.ndarray,
        object_pose: np.ndarray,
        object_velocity: np.ndarray | None = None,
    ) -> None:
        """Drive the sim to an arbitrary planar configuration.

        Lossless in position and velocity -- this domain is genuinely 2D, so the reduced
        planar state carries everything except Box2D's internal warm-start cache.
        """
        pose = np.asarray(object_pose, dtype=np.float64).reshape(-1, 3)
        if pose.shape[0] != self.config.num_objects:
            raise ValueError(
                f"object_pose has {pose.shape[0]} rows but the env has {self.config.num_objects}."
            )
        self.pusher_xy = np.asarray(pusher_xy, dtype=np.float64).reshape(2).copy()
        self._pusher_body.position = tuple(self.pusher_xy * _SCALE)
        self._pusher_body.linearVelocity = (0.0, 0.0)
        velocity = (
            None
            if object_velocity is None
            else np.asarray(object_velocity, dtype=np.float64).reshape(-1, 6)
        )
        for index, body in enumerate(self._bodies):
            body.position = tuple(pose[index, :2] * _SCALE)
            body.angle = float(pose[index, 2])
            if velocity is None:
                body.linearVelocity = (0.0, 0.0)
                body.angularVelocity = 0.0
            else:
                body.linearVelocity = tuple(velocity[index, 3:5] * _SCALE)
                body.angularVelocity = float(velocity[index, 2])
            body.awake = True
