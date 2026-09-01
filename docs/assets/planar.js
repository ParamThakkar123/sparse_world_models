// The planar pushing environment, ported to JavaScript.
//
// A port of `models/envs/planar_push.py`. MuJoCo, Box2D and Chipmunk cannot run in a browser,
// so those three domains are replayed from recorded states -- but the planar environment is
// ours and is a few hundred lines of arithmetic, so here it runs live. That is what makes the
// sandbox real rather than a video: the viewer drives the pusher, this integrates the true
// physics, and the model predicts against a ground truth that did not exist until the viewer
// created it.
//
// Constants and the resolution order are kept identical to the Python. In particular contact
// resolution runs AFTER integration within each substep, so objects never end a substep
// overlapping, and the pusher is infinitely massive so it never yields.

export const OBJECT_RADIUS = 0.025;
export const PUSHER_RADIUS = 0.02;

export const DEFAULT_CONFIG = {
  numObjects: 3,
  actionScale: 0.04,
  pusherBounds: [-0.26, 0.26],
  objectBounds: [-0.26, 0.26],
  minObjectSeparation: 0.09,
  goalClearance: 0.1,
  goalXY: [0.18, 0.18],
  targetObject: 0,
  linearDamping: 0.45,
  angularDamping: 0.7,
  restitution: 0.0,
  solverIterations: 4,
  substeps: 4,
  torqueGain: 2383.0,
};

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Deterministic PRNG so a scene can be reproduced from its seed, as the Python does with
 *  `np.random.default_rng(seed)`. Values will not match numpy's stream -- only the physics
 *  needs to agree, and the layout is resampled in the browser anyway. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class PlanarPushEnv {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.random = mulberry32(this.config.seed ?? 1);
    this.reset();
  }

  reset(seed) {
    if (seed !== undefined) this.random = mulberry32(seed);
    const n = this.config.numObjects;
    this.pusher = [0.0, -0.22];
    this.objectXY = [];
    this.objectYaw = new Array(n).fill(0);
    this.objectVel = Array.from({ length: n }, () => [0, 0]);
    this.objectOmega = new Array(n).fill(0);
    this.stepCount = 0;
    for (let i = 0; i < n; i += 1) {
      this.objectXY.push(this.sampleFreeXY());
      this.objectYaw[i] = (this.random() * 2 - 1) * Math.PI;
    }
    return this.observation();
  }

  sampleFreeXY() {
    const [lower, upper] = this.config.objectBounds;
    const goal = this.config.goalXY;
    for (let attempt = 0; attempt < 4000; attempt += 1) {
      const xy = [
        lower + this.random() * (upper - lower),
        lower + this.random() * (upper - lower),
      ];
      if (Math.hypot(xy[0] - 0.0, xy[1] + 0.22) < 0.1) continue;
      if (Math.hypot(xy[0] - goal[0], xy[1] - goal[1]) < this.config.goalClearance) continue;
      const clear = this.objectXY.every(
        (other) => Math.hypot(xy[0] - other[0], xy[1] - other[1]) > this.config.minObjectSeparation,
      );
      if (clear) return xy;
    }
    // Falling back rather than throwing: in the browser a crowded layout should degrade to a
    // slightly tighter scene, not to a blank page.
    return [lower + this.random() * (upper - lower), lower + this.random() * (upper - lower)];
  }

  step(action) {
    const [low, high] = this.config.pusherBounds;
    const clipped = [clamp(action[0], -1, 1), clamp(action[1], -1, 1)];
    const target = [
      clamp(this.pusher[0] + clipped[0] * this.config.actionScale, low, high),
      clamp(this.pusher[1] + clipped[1] * this.config.actionScale, low, high),
    ];
    // Substepping matters: one jump can tunnel the pusher through a disc, which would break
    // the "only touched objects move" property the whole study is about.
    const stride = [
      (target[0] - this.pusher[0]) / this.config.substeps,
      (target[1] - this.pusher[1]) / this.config.substeps,
    ];
    const drive = [target[0] - this.pusher[0], target[1] - this.pusher[1]];
    for (let sub = 0; sub < this.config.substeps; sub += 1) {
      this.pusher = [this.pusher[0] + stride[0], this.pusher[1] + stride[1]];
      this.integrate(1 / this.config.substeps);
      this.resolveContacts(drive);
    }
    this.stepCount += 1;
    return this.observation();
  }

  integrate(fraction) {
    const decayLinear = this.config.linearDamping ** fraction;
    const decayAngular = this.config.angularDamping ** fraction;
    for (let i = 0; i < this.objectXY.length; i += 1) {
      this.objectXY[i][0] += this.objectVel[i][0] * fraction;
      this.objectXY[i][1] += this.objectVel[i][1] * fraction;
      let yaw = this.objectYaw[i] + this.objectOmega[i] * fraction + Math.PI;
      yaw = ((yaw % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      this.objectYaw[i] = yaw;
      this.objectVel[i][0] *= decayLinear;
      this.objectVel[i][1] *= decayLinear;
      this.objectOmega[i] *= decayAngular;
      // Snap sub-threshold drift to rest, so numerical noise never registers as change.
      if (Math.hypot(this.objectVel[i][0], this.objectVel[i][1]) < 1e-5) {
        this.objectVel[i] = [0, 0];
      }
      if (Math.abs(this.objectOmega[i]) < 1e-5) this.objectOmega[i] = 0;
    }
  }

  resolveContacts(drive) {
    const [lower, upper] = this.config.objectBounds;
    for (let iteration = 0; iteration < this.config.solverIterations; iteration += 1) {
      this.resolvePusherContacts(drive);
      this.resolveObjectContacts();
    }
    for (const xy of this.objectXY) {
      xy[0] = clamp(xy[0], lower, upper);
      xy[1] = clamp(xy[1], lower, upper);
    }
  }

  resolvePusherContacts(drive) {
    for (let i = 0; i < this.objectXY.length; i += 1) {
      const offset = [this.objectXY[i][0] - this.pusher[0], this.objectXY[i][1] - this.pusher[1]];
      const distance = Math.hypot(offset[0], offset[1]);
      const overlap = PUSHER_RADIUS + OBJECT_RADIUS - distance;
      if (overlap <= 0) continue;
      const normal = [offset[0] / Math.max(distance, 1e-9), offset[1] / Math.max(distance, 1e-9)];
      // The pusher never yields: the object takes the whole positional correction. That is
      // what makes the pusher an exogenous driver.
      this.objectXY[i][0] += normal[0] * overlap;
      this.objectXY[i][1] += normal[1] * overlap;
      // Velocity is set by the pusher's advance along the normal, and only raised: an object
      // already moving away faster is not slowed by being caught up with.
      const approach = Math.max(normal[0] * drive[0] + normal[1] * drive[1], 0);
      const alongNormal = this.objectVel[i][0] * normal[0] + this.objectVel[i][1] * normal[1];
      const gain = Math.max(approach - alongNormal, 0);
      this.objectVel[i][0] += normal[0] * gain;
      this.objectVel[i][1] += normal[1] * gain;
      // A disc has no lever arm, so a central impulse would leave yaw constant forever and
      // silently make a third of the prediction target trivial. The tabletop's objects are
      // boxes, whose contact point is off-centre except face-on; this models that lever with
      // the 4-fold symmetry of a square.
      const contactAngle = Math.atan2(normal[1], normal[0]);
      const lever = OBJECT_RADIUS * Math.sin(2 * (contactAngle - this.objectYaw[i]));
      this.objectOmega[i] += this.config.torqueGain * lever * approach;
    }
  }

  resolveObjectContacts() {
    const n = this.objectXY.length;
    if (n < 2) return;
    // Jacobi-style: every pair resolved against the pre-update positions, so a push
    // propagates along a chain of touching objects one link per solver pass.
    const correction = Array.from({ length: n }, () => [0, 0]);
    const impulse = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        const offset = [this.objectXY[j][0] - this.objectXY[i][0],
                        this.objectXY[j][1] - this.objectXY[i][1]];
        const distance = Math.hypot(offset[0], offset[1]);
        const overlap = 2 * OBJECT_RADIUS - distance;
        if (overlap <= 0) continue;
        const safe = Math.max(distance, 1e-9);
        const normal = [offset[0] / safe, offset[1] / safe];
        correction[i][0] -= normal[0] * overlap * 0.5;
        correction[i][1] -= normal[1] * overlap * 0.5;
        const relative = (this.objectVel[j][0] - this.objectVel[i][0]) * normal[0]
                       + (this.objectVel[j][1] - this.objectVel[i][1]) * normal[1];
        if (relative < 0) {
          // Equal-mass split, so momentum is conserved and the push carries down the chain.
          const magnitude = -(1 + this.config.restitution) * relative * 0.5;
          impulse[i][0] -= normal[0] * magnitude;
          impulse[i][1] -= normal[1] * magnitude;
        }
      }
    }
    for (let i = 0; i < n; i += 1) {
      this.objectXY[i][0] += correction[i][0];
      this.objectXY[i][1] += correction[i][1];
      this.objectVel[i][0] += impulse[i][0];
      this.objectVel[i][1] += impulse[i][1];
    }
  }

  /** The flat state vector the model consumes: pusher(2), poses(N*3), velocities(N*6), goal(2).
   *  Layout and velocity-column placement match `generate_transitions.flatten_state`. */
  state() {
    const flat = [this.pusher[0], this.pusher[1]];
    for (let i = 0; i < this.objectXY.length; i += 1) {
      flat.push(this.objectXY[i][0], this.objectXY[i][1], this.objectYaw[i]);
    }
    for (let i = 0; i < this.objectXY.length; i += 1) {
      // Six components to match the tabletop's cvel layout; only the planar entries are
      // meaningful, and columns 3:5 are the linear ones every rule reads.
      flat.push(0, 0, this.objectOmega[i], this.objectVel[i][0], this.objectVel[i][1], 0);
    }
    flat.push(this.config.goalXY[0], this.config.goalXY[1]);
    return flat;
  }

  observation() {
    return {
      pusher: [...this.pusher],
      poses: this.objectXY.map((xy, i) => [xy[0], xy[1], this.objectYaw[i]]),
      goal: [...this.config.goalXY],
    };
  }

  snapshot() {
    return {
      pusher: [...this.pusher],
      objectXY: this.objectXY.map((xy) => [...xy]),
      objectYaw: [...this.objectYaw],
      objectVel: this.objectVel.map((v) => [...v]),
      objectOmega: [...this.objectOmega],
      stepCount: this.stepCount,
    };
  }

  restore(snapshot) {
    this.pusher = [...snapshot.pusher];
    this.objectXY = snapshot.objectXY.map((xy) => [...xy]);
    this.objectYaw = [...snapshot.objectYaw];
    this.objectVel = snapshot.objectVel.map((v) => [...v]);
    this.objectOmega = [...snapshot.objectOmega];
    this.stepCount = snapshot.stepCount;
  }
}

/** Ground-truth change labels between two frames, using the same thresholds the datasets
 *  were built with (`generate_transitions.POSITION_EPS` / `YAW_EPS`). */
export const POSITION_EPS = 1e-3;
export const YAW_EPS = 1e-2;

export function changedMask(before, after) {
  return before.map((pose, i) => {
    const dx = after[i][0] - pose[0];
    const dy = after[i][1] - pose[1];
    let dyaw = after[i][2] - pose[2];
    dyaw = ((dyaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return (Math.hypot(dx, dy) > POSITION_EPS || Math.abs(dyaw) > YAW_EPS) ? 1 : 0;
  });
}
