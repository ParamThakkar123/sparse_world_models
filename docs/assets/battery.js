// The trivial-rule battery, live in the browser.
//
// A port of `experiments/onset_shortcut_audit.py`. This is the part of the demo that carries
// the paper's actual claim: the point is not that the model works, it is that a rule with
// zero parameters keeps beating it. Watching `nearest_to_pusher` track the label frame by
// frame on a scene the viewer is driving themselves is a more honest demonstration than any
// table, and it is the same computation the paper reports.
//
// Rules whose inputs are unavailable are omitted rather than scored zero, exactly as in
// `experiments/audit_battery.py`.

import { planarSpeeds, pusherNext, unpackState } from './model.js';

export const REST_SPEED = 2.55e-5;

export const RULE_DESCRIPTIONS = {
  already_moving: 'Predict change iff the object is already moving. Zero parameters. Wins the motion benchmark.',
  pusher_near: 'Predict change iff the pusher will be within a fitted radius. One parameter.',
  pusher_approaching: 'As pusher_near, but only when the action closes the gap rather than opening it.',
  nearest_to_pusher: 'Predict change for the single object closest to the pusher. Zero parameters. Wins the onset benchmark.',
  second_nearest_to_pusher: 'The obvious follow-up once "nearest" is defeated.',
  two_nearest_to_pusher: 'The two closest objects.',
  moving_or_near: 'Moving OR near the pusher — the best a reader could do by combining both shortcuts by hand.',
  near_a_mover: 'Within a radius of an object that is already moving. The natural shortcut for a contact chain.',
  near_pusher_or_mover: 'Near the pusher OR near a mover.',
  moving_or_near_mover: 'Moving OR near a mover. Wins the interaction benchmark.',
  always_change: 'Flag everything. The degeneracy floor — every ungated published model sits here (recall 0.999–1.000).',
};

export const ZERO_PARAMETER_RULES = new Set([
  'already_moving', 'nearest_to_pusher', 'second_nearest_to_pusher',
  'two_nearest_to_pusher', 'always_change',
]);

/** Every rule's per-object prediction for one transition.
 *
 *  `radius` is the contact radius the thresholded rules use. In the paper it is fitted on
 *  the validation split and applied unchanged to test; here it is a slider, so a viewer can
 *  check for themselves that no particular setting is doing the work.
 */
export function evaluateRules(state, action, numObjects, radius, constants) {
  const { pusher, poses } = unpackState(state, numObjects);
  const speeds = planarSpeeds(state, numObjects);
  const next = pusherNext(pusher, action, constants);
  const xy = poses.map((pose) => [pose[0], pose[1]]);

  const moving = speeds.map((speed) => (speed > REST_SPEED ? 1 : 0));
  const distance = xy.map(([x, y]) => Math.hypot(next[0] - x, next[1] - y));
  const distanceBefore = xy.map(([x, y]) => Math.hypot(pusher[0] - x, pusher[1] - y));
  const near = distance.map((d) => (d <= radius ? 1 : 0));
  const closing = distance.map((d, i) => (d < distanceBefore[i] ? 1 : 0));

  const order = distance.map((d, i) => [d, i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
  const oneHot = (indices) => {
    const mask = new Array(numObjects).fill(0);
    indices.forEach((index) => { if (index !== undefined) mask[index] = 1; });
    return mask;
  };

  // Distance from each object to the nearest object that is ALREADY moving. With nothing
  // moving this is Infinity, so those rows predict nothing rather than everything.
  const distanceToMover = xy.map(([x, y], i) => {
    let best = Infinity;
    for (let j = 0; j < numObjects; j += 1) {
      if (j === i || !moving[j]) continue;
      best = Math.min(best, Math.hypot(xy[j][0] - x, xy[j][1] - y));
    }
    return best;
  });
  const nearMover = distanceToMover.map((d) => (d <= radius ? 1 : 0));

  const either = (a, b) => a.map((value, i) => Math.max(value, b[i]));

  const rules = {
    already_moving: moving,
    pusher_near: near,
    pusher_approaching: near.map((value, i) => value * closing[i]),
    nearest_to_pusher: oneHot([order[0]]),
    moving_or_near: either(moving, near),
    always_change: new Array(numObjects).fill(1),
    near_a_mover: nearMover,
    near_pusher_or_mover: either(near, nearMover),
    moving_or_near_mover: either(moving, nearMover),
  };
  if (numObjects >= 2) {
    rules.second_nearest_to_pusher = oneHot([order[1]]);
    rules.two_nearest_to_pusher = oneHot([order[0], order[1]]);
  }
  return { rules, speeds, moving, distance };
}

/** Running confusion-matrix totals, so F1 can be reported over a whole episode rather than
 *  a single frame. One frame of a three-object scene is far too small a sample to rank
 *  anything, and a per-frame F1 would flicker between 0 and 1 and mean nothing. */
export class RunningScore {
  constructor() { this.reset(); }

  reset() {
    this.truePositive = 0;
    this.falsePositive = 0;
    this.falseNegative = 0;
    this.onsetTruePositive = 0;
    this.onsetFalsePositive = 0;
    this.onsetFalseNegative = 0;
  }

  update(prediction, target, atRest) {
    for (let i = 0; i < prediction.length; i += 1) {
      const predicted = prediction[i] > 0.5;
      const actual = target[i] > 0.5;
      if (predicted && actual) this.truePositive += 1;
      else if (predicted) this.falsePositive += 1;
      else if (actual) this.falseNegative += 1;
      // Onset: objects currently at rest, which can only start moving through contact. This
      // is the half of the task that requires prediction rather than continuation.
      if (atRest[i]) {
        if (predicted && actual) this.onsetTruePositive += 1;
        else if (predicted) this.onsetFalsePositive += 1;
        else if (actual) this.onsetFalseNegative += 1;
      }
    }
  }

  static f1(truePositive, falsePositive, falseNegative) {
    const precision = truePositive + falsePositive ? truePositive / (truePositive + falsePositive) : 0;
    const recall = truePositive + falseNegative ? truePositive / (truePositive + falseNegative) : 0;
    return precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  }

  get f1() { return RunningScore.f1(this.truePositive, this.falsePositive, this.falseNegative); }

  get onsetF1() {
    return RunningScore.f1(this.onsetTruePositive, this.onsetFalsePositive, this.onsetFalseNegative);
  }

  get support() { return this.truePositive + this.falseNegative; }
}

/** A scoreboard over every rule plus the learned model, updated one frame at a time. */
export class Scoreboard {
  constructor() {
    this.scores = new Map();
  }

  reset() { this.scores.clear(); }

  update(name, prediction, target, atRest) {
    if (!this.scores.has(name)) this.scores.set(name, new RunningScore());
    this.scores.get(name).update(prediction, target, atRest);
  }

  /** Rows sorted by F1, best first, with the model tagged so the UI can highlight it. */
  rows() {
    return [...this.scores.entries()]
      .map(([name, score]) => ({
        name,
        f1: score.f1,
        onsetF1: score.onsetF1,
        support: score.support,
        isModel: name === 'model',
        parameters: name === 'model' ? null : (ZERO_PARAMETER_RULES.has(name) ? 0 : 1),
      }))
      .sort((a, b) => b.f1 - a.f1);
  }
}
