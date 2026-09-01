// The trained sparse/residual world model, running in the browser.
//
// This is a port of `models/sparse_residual.py` (gate + delta head) and of
// `build_object_features_contact` in `experiments/train_sparse_model.py`, driven by the
// weights `experiments/export_web_model.py` exports from a real checkpoint. The models are
// two-layer MLPs of a few thousand parameters, so the whole forward pass is a handful of
// matrix multiplies and needs no runtime, no WASM, and no network beyond the weight file.
//
// It has to be an exact port, not a lookalike: `tests/test_web_export.py` runs this file
// under node against outputs recorded from PyTorch and requires agreement to 1e-4. A
// transposed weight or a mis-ordered feature block would still produce plausible-looking
// gates on screen, which is exactly the kind of quiet wrongness this project exists to
// complain about.

export const POSE_DIM = 3;
export const VELOCITY_DIM = 6;
export const GOAL_DIM = 2;
export const PUSHER_DIM = 2;

/** Unpack the flat state vector into the pieces the feature builder needs.
 *  Layout matches `models/layout.StateLayout`: pusher(2), poses(N*3), velocities(N*6), goal(2). */
export function unpackState(state, numObjects) {
  const poseStart = PUSHER_DIM;
  const velStart = poseStart + numObjects * POSE_DIM;
  const goalStart = velStart + numObjects * VELOCITY_DIM;
  const poses = [];
  const velocities = [];
  for (let i = 0; i < numObjects; i += 1) {
    poses.push(state.slice(poseStart + i * POSE_DIM, poseStart + (i + 1) * POSE_DIM));
    velocities.push(state.slice(velStart + i * VELOCITY_DIM, velStart + (i + 1) * VELOCITY_DIM));
  }
  return {
    pusher: state.slice(0, PUSHER_DIM),
    poses,
    velocities,
    goal: state.slice(goalStart, goalStart + GOAL_DIM),
  };
}

/** Planar speed per object: columns 3:5 of the 6-wide velocity block, matching
 *  `momentum_shortcut.planar_speed`. Everything that asks "is this object moving" must use
 *  this and only this, or the trivial rules stop being the rules the paper reports. */
export function planarSpeeds(state, numObjects) {
  const { velocities } = unpackState(state, numObjects);
  return velocities.map((v) => Math.hypot(v[3], v[4]));
}

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Where the pusher ends up after this action. Deterministic, and identical to `env.step`. */
export function pusherNext(pusher, action, constants) {
  const scale = constants.pusher_action_scale;
  const bound = constants.pusher_bound;
  return [
    clamp(pusher[0] + clamp(action[0], -1, 1) * scale, -bound, bound),
    clamp(pusher[1] + clamp(action[1], -1, 1) * scale, -bound, bound),
  ];
}

/** The velocity-FREE contact featurisation, width 19 per object at any object count.
 *
 *  Fixed width is what lets one exported checkpoint drive 3-, 5- and 8-object scenes: the
 *  neighbour block is a permutation-invariant summary rather than a concatenation of every
 *  other object, so nothing in the input grows with N.
 *
 *  Block order below is load-bearing -- it must match the `torch.cat` in
 *  `build_object_features_contact` exactly.
 */
export function contactFeatures(state, action, numObjects, constants) {
  const { pusher, poses, goal } = unpackState(state, numObjects);
  const clipped = [clamp(action[0], -1, 1), clamp(action[1], -1, 1)];
  const next = pusherNext(pusher, clipped, constants);
  const xy = poses.map((pose) => [pose[0], pose[1]]);

  const features = [];
  for (let i = 0; i < numObjects; i += 1) {
    const [ox, oy] = xy[i];
    const relGoal = [goal[0] - ox, goal[1] - oy];
    const relPusher = [pusher[0] - ox, pusher[1] - oy];
    const relNext = [next[0] - ox, next[1] - oy];
    const contactDistance = Math.hypot(relNext[0], relNext[1]);
    const signed = contactDistance - constants.contact_radius;
    const safe = Math.max(contactDistance, 1e-6);
    const pushDir = [-relNext[0] / safe, -relNext[1] / safe];

    // Neighbour block: mean relative position of the others, the nearest one's relative
    // position, and the distance to it. pairwise[i][j] = xy_j - xy_i, diagonal excluded.
    let meanX = 0;
    let meanY = 0;
    let nearestDistance = Infinity;
    let nearestRel = [0, 0];
    if (numObjects > 1) {
      for (let j = 0; j < numObjects; j += 1) {
        if (j === i) continue;
        const dx = xy[j][0] - ox;
        const dy = xy[j][1] - oy;
        meanX += dx;
        meanY += dy;
        const distance = Math.hypot(dx, dy);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestRel = [dx, dy];
        }
      }
      meanX /= numObjects - 1;
      meanY /= numObjects - 1;
    } else {
      nearestDistance = 0;
    }

    features.push([
      poses[i][0], poses[i][1], poses[i][2],
      relGoal[0], relGoal[1],
      relPusher[0], relPusher[1],
      clipped[0], clipped[1],
      relNext[0], relNext[1],
      signed,
      pushDir[0], pushDir[1],
      meanX, meanY, nearestRel[0], nearestRel[1], nearestDistance,
    ]);
  }
  return features;
}

/** One `nn.Linear`: out = W x + b, with W stored row-major as PyTorch does. */
function linear(input, layer) {
  const { weight, bias } = layer;
  const output = new Array(weight.length);
  for (let row = 0; row < weight.length; row += 1) {
    const w = weight[row];
    let sum = bias[row];
    for (let col = 0; col < w.length; col += 1) sum += w[col] * input[col];
    output[row] = sum;
  }
  return output;
}

/** ReLU between every pair of layers and nothing after the last, matching
 *  `ObjectChangeGate.mlp` and the delta head's `nn.Sequential`. */
function mlp(input, layers) {
  let activation = input;
  for (let index = 0; index < layers.length; index += 1) {
    activation = linear(activation, layers[index]);
    if (index < layers.length - 1) {
      activation = activation.map((value) => (value > 0 ? value : 0));
    }
  }
  return activation;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class SparseResidualModel {
  constructor(weights) {
    this.weights = weights;
    this.constants = weights.constants;
  }

  /** Load the exported bundle. It holds one entry per trained checkpoint, because the demo
   *  runs in two regimes: the live sandbox is the planar environment and gets the
   *  planar-trained gate, while the replay tab shows recorded MuJoCo/Box2D/Chipmunk episodes
   *  and gets the tabletop-trained one. Scoring a tabletop model on live planar physics would
   *  show a domain shift and invite a viewer to read it as the model being bad. */
  static async loadBundle(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load model weights from ${url}`);
    const bundle = await response.json();
    const models = new Map();
    for (const [name, weights] of Object.entries(bundle.models)) {
      models.set(name, new SparseResidualModel(weights));
    }
    return { models, default: bundle.default };
  }

  /** Gate probabilities, hard gates and per-object deltas for one transition.
   *
   *  The deployed gate is a threshold at 0.5 on the sigmoid, which is what
   *  `momentum_shortcut.predicted_mask` uses to score every number in the paper. The
   *  Gumbel noise in the training-time estimator is deliberately absent: it would make the
   *  same scene score differently on every frame, and evaluation never uses it either.
   */
  predict(state, action, numObjects) {
    const features = contactFeatures(state, action, numObjects, this.constants);
    const probs = [];
    const gates = [];
    const deltas = [];
    for (const objectFeatures of features) {
      const probability = sigmoid(mlp(objectFeatures, this.weights.gate)[0]);
      probs.push(probability);
      gates.push(probability >= 0.5 ? 1 : 0);
      deltas.push(mlp(objectFeatures, this.weights.delta));
    }
    return { features, probs, gates, deltas };
  }

  /** Next poses under the model: pose + gate * delta. Objects the gate leaves off are
   *  copied forward verbatim -- the whole architectural claim in one line. */
  step(state, action, numObjects) {
    const { poses } = unpackState(state, numObjects);
    const { probs, gates, deltas } = this.predict(state, action, numObjects);
    const next = poses.map((pose, i) => [
      pose[0] + gates[i] * deltas[i][0],
      pose[1] + gates[i] * deltas[i][1],
      pose[2] + gates[i] * deltas[i][2],
    ]);
    return { poses: next, probs, gates, deltas };
  }
}
