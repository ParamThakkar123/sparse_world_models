// The demo's UI: four tasks, one canvas, one live scoreboard.
//
// The tasks exist to make one point checkable rather than assertable. Every frame the
// trained gate predicts which objects will move, the eleven trivial rules predict the same
// thing, the physics (live for planar, recorded for the other three engines) says who was
// right, and the scoreboard accumulates F1 for all of them side by side.

import { SparseResidualModel, planarSpeeds, unpackState } from './model.js';
import { REST_SPEED, RULE_DESCRIPTIONS, Scoreboard, evaluateRules } from './battery.js';
import { OBJECT_RADIUS, PUSHER_RADIUS, PlanarPushEnv, changedMask } from './planar.js';

const WORLD_BOUND = 0.28;   // a little beyond the 0.26 table edge, so walls are visible
const STEP_MS = 60;         // one control step per 60 ms: fast enough to read, slow enough to follow

// Frames are scored only when some object actually moved by more than this, which is the
// `--min-max-xy-delta 0.02` motion filter every benchmark in the paper is built with. Without
// it the scoreboard would be dominated by the ~95% of steps where the pusher is still
// travelling and nothing has happened, every rule would score near zero, and the numbers here
// would not be comparable to any number in the paper.
const MOTION_THRESHOLD = 0.02;

const TASKS = {
  sandbox: {
    title: 'Sandbox — you drive, the model predicts',
    copy: 'Live planar physics, running in this tab. Each frame the model outputs a gate '
        + 'per object and a delta; the dashed outline is where it thinks each object will be. '
        + 'Switch to “I drive” and push objects yourself — the ground truth is created by you, '
        + 'so nothing here is a recording.',
    hint: 'Mouse mode: the pusher moves toward your cursor. The model has never seen the '
        + 'states you create this way, which is the point.',
  },
  replay: {
    title: 'Four engines — MuJoCo, Box2D, Chipmunk2D, ours',
    copy: 'MuJoCo, Box2D and Chipmunk cannot run in a browser, so these are recorded episodes '
        + 'from the paper’s datasets — but the model is predicting on each frame live, and the '
        + 'labels are the real ones. The shortcut’s existence condition holds on all four '
        + 'engines: P(change | already moving) exceeds P(change | at rest) by 36–84×.',
    hint: 'Watch already_moving climb on billiards, where objects roll for a long time after '
        + 'contact, and the learned gate fail to beat it.',
  },
  transfer: {
    title: 'Count transfer — one checkpoint, any number of objects',
    copy: 'The same weights, trained on three-object scenes, driving scenes with up to twelve. '
        + 'The contact featurisation is fixed-width, so nothing has to be retrained. Watch the '
        + 'objects the gate leaves off: they are copied forward exactly, which is the one '
        + 'architectural claim that survived proper baselines.',
    hint: 'Turn the count up. The trivial rules degrade too — but they degrade more slowly.',
  },
  planning: {
    title: 'Planning — the model as a forward simulator',
    copy: 'Sampling-based MPC (CEM), replanning every step, using the model to imagine each '
        + 'candidate action sequence. The planner is sound: with the true simulator in place of '
        + 'the model it succeeds every time. Anything it fails to do here is model quality.',
    hint: 'In the paper this reaches 0.25 success against a dense monolith’s 0.00 — and a '
        + 'published probabilistic ensemble (PETS) beats it at 0.35.',
  },
};

const state = {
  models: null,
  model: null,
  modelName: null,
  episodes: null,
  task: 'sandbox',
  running: true,
  drive: 'auto',
  radius: 0.07,
  numObjects: 3,
  env: null,
  replayName: null,
  replayIndex: 0,
  scoreboard: new Scoreboard(),
  frames: 0,
  pointer: null,
  lastPrediction: null,
  lastTruth: null,
  seenFrames: 0,
  planSuccess: 0,
  planAttempts: 0,
};

const canvas = document.getElementById('scene');
const context = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const scoresBody = document.querySelector('#scores tbody');
const framesEl = document.getElementById('frames');
const taskCopy = document.getElementById('task-copy');
const hintEl = document.getElementById('hint');

// ---------------------------------------------------------------- drawing

function toCanvas(x, y) {
  const scale = canvas.width / (2 * WORLD_BOUND);
  return [(x + WORLD_BOUND) * scale, (WORLD_BOUND - y) * scale];
}

function styleValue(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function drawScene({ poses, pusher, goal, gates, probs, predicted, truth }) {
  const scale = canvas.width / (2 * WORLD_BOUND);
  context.clearRect(0, 0, canvas.width, canvas.height);

  // Table
  const [tableX, tableY] = toCanvas(-0.26, 0.26);
  context.fillStyle = styleValue('--bg');
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = styleValue('--line');
  context.lineWidth = 2;
  context.strokeRect(tableX, tableY, 0.52 * scale, 0.52 * scale);

  // Goal
  if (goal) {
    const [gx, gy] = toCanvas(goal[0], goal[1]);
    context.beginPath();
    context.arc(gx, gy, 0.03 * scale, 0, Math.PI * 2);
    context.strokeStyle = styleValue('--goal');
    context.setLineDash([4, 4]);
    context.lineWidth = 2;
    context.stroke();
    context.setLineDash([]);
  }

  // Predicted next poses first, so the true objects sit on top of their own ghosts.
  if (predicted) {
    context.strokeStyle = styleValue('--ghost');
    context.setLineDash([5, 4]);
    context.lineWidth = 2;
    predicted.forEach((pose) => drawBox(pose, scale, null));
    context.setLineDash([]);
  }

  poses.forEach((pose, index) => {
    const gateOn = gates && gates[index];
    const changed = truth && truth[index];
    context.lineWidth = 3;
    context.fillStyle = gateOn
      ? `color-mix(in srgb, ${styleValue('--gate')} 55%, transparent)`
      : styleValue('--surface');
    context.strokeStyle = changed ? styleValue('--truth') : styleValue('--line');
    drawBox(pose, scale, true);

    if (probs) {
      const [px, py] = toCanvas(pose[0], pose[1]);
      context.fillStyle = styleValue('--ink-soft');
      context.font = '11px ui-monospace, monospace';
      context.textAlign = 'center';
      context.fillText(probs[index].toFixed(2), px, py - 0.035 * scale);
    }
  });

  const [px, py] = toCanvas(pusher[0], pusher[1]);
  context.beginPath();
  context.arc(px, py, PUSHER_RADIUS * scale, 0, Math.PI * 2);
  context.fillStyle = styleValue('--ink');
  context.fill();
}

function drawBox(pose, scale, filled) {
  const [x, y] = toCanvas(pose[0], pose[1]);
  const half = OBJECT_RADIUS * scale;
  context.save();
  context.translate(x, y);
  context.rotate(-pose[2]);
  context.beginPath();
  context.rect(-half, -half, 2 * half, 2 * half);
  if (filled) context.fill();
  context.stroke();
  context.restore();
}

// ---------------------------------------------------------------- scoring

/** Did any object move enough for this frame to belong in a benchmark split? */
function passesMotionFilter(before, after) {
  return before.some((pose, i) => Math.hypot(after[i][0] - pose[0], after[i][1] - pose[1])
    > MOTION_THRESHOLD);
}

function scoreFrame(stateVector, action, numObjects, truthMask) {
  const speeds = planarSpeeds(stateVector, numObjects);
  const atRest = speeds.map((speed) => speed <= REST_SPEED);
  const { rules } = evaluateRules(stateVector, action, numObjects, state.radius,
                                  state.model.constants);
  for (const [name, prediction] of Object.entries(rules)) {
    state.scoreboard.update(name, prediction, truthMask, atRest);
  }
  const { gates } = state.lastPrediction;
  state.scoreboard.update('model', gates, truthMask, atRest);
  state.frames += 1;
}

function renderScores() {
  const rows = state.scoreboard.rows();
  scoresBody.innerHTML = '';
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.isModel) tr.className = 'model';
    else if (row.parameters === 0) tr.className = 'zero-param';
    const label = row.isModel ? 'the trained model' : row.name;
    tr.innerHTML =
      `<td><span class="rule-name">${label}</span></td>` +
      `<td>${row.isModel ? state.model.weights.num_parameters.toLocaleString() : row.parameters}</td>` +
      `<td>${row.f1.toFixed(3)}</td>` +
      `<td>${row.support ? row.onsetF1.toFixed(3) : '—'}</td>`;
    if (!row.isModel && RULE_DESCRIPTIONS[row.name]) tr.title = RULE_DESCRIPTIONS[row.name];
    scoresBody.appendChild(tr);
  }
  framesEl.textContent = state.seenFrames
    ? `${state.frames} scored / ${state.seenFrames} seen`
    : '0 frames';
}

// ---------------------------------------------------------------- policies

/** A simple scripted driver: approach the target object from the side away from the goal,
 *  then push through it. Enough to generate contact events without a learned policy. */
function autoAction(env) {
  const target = env.objectXY[env.config.targetObject];
  const goal = env.config.goalXY;
  const toGoal = [goal[0] - target[0], goal[1] - target[1]];
  const norm = Math.hypot(toGoal[0], toGoal[1]) || 1;
  const behind = [
    target[0] - (toGoal[0] / norm) * 0.06,
    target[1] - (toGoal[1] / norm) * 0.06,
  ];
  const distanceToBehind = Math.hypot(behind[0] - env.pusher[0], behind[1] - env.pusher[1]);
  const aim = distanceToBehind > 0.02 ? behind : target;
  const dx = aim[0] - env.pusher[0];
  const dy = aim[1] - env.pusher[1];
  const magnitude = Math.hypot(dx, dy) || 1;
  return [dx / magnitude, dy / magnitude];
}

function mouseAction(env) {
  if (!state.pointer) return [0, 0];
  const dx = state.pointer[0] - env.pusher[0];
  const dy = state.pointer[1] - env.pusher[1];
  const magnitude = Math.hypot(dx, dy);
  if (magnitude < 0.004) return [0, 0];
  return [dx / magnitude, dy / magnitude];
}

/** CEM-MPC using the model as the forward simulator, mirroring `experiments/planning_mpc.py`:
 *  sample action sequences, roll them out through the MODEL (never the simulator), keep the
 *  elites, refit, and execute only the first action. */
function planAction(env, horizon = 8, samples = 48, iterations = 2, elite = 0.15) {
  const numObjects = env.config.numObjects;
  const goal = env.config.goalXY;
  const target = env.config.targetObject;
  let mean = Array.from({ length: horizon }, () => [0, 0]);
  let std = Array.from({ length: horizon }, () => [0.6, 0.6]);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const scored = [];
    for (let sample = 0; sample < samples; sample += 1) {
      const sequence = mean.map(([mx, my], t) => [
        Math.max(-1, Math.min(1, mx + gaussian() * std[t][0])),
        Math.max(-1, Math.min(1, my + gaussian() * std[t][1])),
      ]);
      scored.push([rollout(env, sequence, numObjects, goal, target), sequence]);
    }
    scored.sort((a, b) => a[0] - b[0]);
    const keep = scored.slice(0, Math.max(2, Math.round(samples * elite))).map(([, s]) => s);
    mean = mean.map((_, t) => [
      keep.reduce((sum, s) => sum + s[t][0], 0) / keep.length,
      keep.reduce((sum, s) => sum + s[t][1], 0) / keep.length,
    ]);
    std = mean.map((_, t) => [
      Math.max(0.05, standardDeviation(keep.map((s) => s[t][0]))),
      Math.max(0.05, standardDeviation(keep.map((s) => s[t][1]))),
    ]);
  }
  return mean[0];
}

/** Imagine a sequence through the model. The pusher is advanced analytically (it is exogenous
 *  and its dynamics are known exactly); only the objects come from the model, which is what
 *  makes this a test of the world model rather than of the controller. */
function rollout(env, sequence, numObjects, goal, target) {
  let vector = env.state().slice();
  let cost = 0;
  const constants = state.model.constants;
  for (let t = 0; t < sequence.length; t += 1) {
    const { poses } = state.model.step(vector, sequence[t], numObjects);
    const unpacked = unpackState(vector, numObjects);
    const pusher = [
      Math.max(-constants.pusher_bound, Math.min(constants.pusher_bound,
        unpacked.pusher[0] + Math.max(-1, Math.min(1, sequence[t][0])) * constants.pusher_action_scale)),
      Math.max(-constants.pusher_bound, Math.min(constants.pusher_bound,
        unpacked.pusher[1] + Math.max(-1, Math.min(1, sequence[t][1])) * constants.pusher_action_scale)),
    ];
    vector = rebuildState(pusher, poses, unpacked.velocities, unpacked.goal);
    const distance = Math.hypot(poses[target][0] - goal[0], poses[target][1] - goal[1]);
    const proximity = Math.hypot(pusher[0] - poses[target][0], pusher[1] - poses[target][1]);
    cost += distance + 0.3 * proximity;
  }
  const finalPoses = unpackState(vector, numObjects).poses;
  // Terminal weight 3.0, matching the planner the paper reports.
  cost += 3.0 * Math.hypot(finalPoses[target][0] - goal[0], finalPoses[target][1] - goal[1]);
  return cost;
}

function rebuildState(pusher, poses, velocities, goal) {
  const flat = [...pusher];
  poses.forEach((pose) => flat.push(pose[0], pose[1], pose[2]));
  velocities.forEach((v) => flat.push(...v));
  flat.push(goal[0], goal[1]);
  return flat;
}

let spare = null;
function gaussian() {
  if (spare !== null) { const value = spare; spare = null; return value; }
  let u = 0; let v = 0; let s = 0;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  const factor = Math.sqrt((-2 * Math.log(s)) / s);
  spare = v * factor;
  return u * factor;
}

function standardDeviation(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length);
}

// ---------------------------------------------------------------- the loop

function stepLive() {
  const env = state.env;
  const numObjects = env.config.numObjects;
  const before = env.observation().poses.map((pose) => [...pose]);
  const vector = env.state();

  let action;
  if (state.task === 'planning') action = planAction(env);
  else if (state.drive === 'mouse') action = mouseAction(env);
  else action = autoAction(env);

  state.lastPrediction = state.model.step(vector, action, numObjects);
  env.step(action);
  const after = env.observation().poses;
  const truth = changedMask(before, after);
  state.lastTruth = truth;
  state.seenFrames += 1;
  if (passesMotionFilter(before, after)) scoreFrame(vector, action, numObjects, truth);

  if (state.task === 'planning') {
    const target = env.config.targetObject;
    const distance = Math.hypot(after[target][0] - env.config.goalXY[0],
                                after[target][1] - env.config.goalXY[1]);
    if (distance < 0.05) { state.planSuccess += 1; state.planAttempts += 1; newScene(); return; }
    if (env.stepCount >= 60) { state.planAttempts += 1; newScene(); return; }
  }

  drawScene({
    poses: after,
    pusher: env.pusher,
    goal: env.config.goalXY,
    gates: state.lastPrediction.gates,
    probs: state.lastPrediction.probs,
    predicted: state.lastPrediction.poses,
    truth,
  });
}

function stepReplay() {
  const episode = state.episodes[state.replayName];
  const numObjects = episode.num_objects;
  if (state.replayIndex >= episode.state.length) state.replayIndex = 0;
  const vector = episode.state[state.replayIndex];
  const action = episode.action[state.replayIndex];
  const truth = episode.target_mask[state.replayIndex];

  state.lastPrediction = state.model.step(vector, action, numObjects);
  const { poses, pusher, goal } = unpackState(vector, numObjects);
  state.seenFrames += 1;
  const next = episode.state[state.replayIndex + 1];
  if (next === undefined || passesMotionFilter(poses, unpackState(next, numObjects).poses)) {
    scoreFrame(vector, action, numObjects, truth);
  }

  drawScene({
    poses,
    pusher,
    goal,
    gates: state.lastPrediction.gates,
    probs: state.lastPrediction.probs,
    predicted: state.lastPrediction.poses,
    truth,
  });
  state.replayIndex += 1;
}

let lastStep = 0;
function frame(timestamp) {
  if (state.running && timestamp - lastStep > STEP_MS) {
    lastStep = timestamp;
    try {
      if (state.task === 'replay') stepReplay();
      else stepLive();
      renderScores();
      updateStatus();
    } catch (error) {
      statusEl.textContent = `Stopped: ${error.message}`;
      state.running = false;
    }
  }
  requestAnimationFrame(frame);
}

function updateStatus() {
  if (state.task === 'planning' && state.planAttempts > 0) {
    const rate = (state.planSuccess / state.planAttempts).toFixed(2);
    statusEl.textContent =
      `Planning through the model: ${state.planSuccess}/${state.planAttempts} episodes solved `
      + `(success ${rate}). The paper reports 0.25 at 20 episodes; a handful of browser `
      + `episodes is not a measurement.`;
    return;
  }
  const rows = state.scoreboard.rows();
  const best = rows.find((row) => !row.isModel);
  const model = rows.find((row) => row.isModel);
  if (!best || !model || state.frames < 15) {
    statusEl.textContent = 'Collecting frames…';
    return;
  }
  const margin = model.f1 - best.f1;
  statusEl.textContent = margin <= 0
    ? `Best trivial rule: ${best.name} (${best.parameters} params) at F1 ${best.f1.toFixed(3)} — `
      + `beating the trained model by ${(-margin).toFixed(3)}.`
    : `The model leads ${best.name} by ${margin.toFixed(3)} F1 on this scene.`;
}

// ---------------------------------------------------------------- wiring

function newScene() {
  if (state.task === 'replay') {
    state.replayIndex = 0;
  } else {
    state.env = new PlanarPushEnv({
      numObjects: state.numObjects,
      seed: Math.floor(Math.random() * 1e6),
      // Planning needs room to manoeuvre; the sandbox and transfer tasks want the packed
      // scenes where the interesting contact chains happen.
      minObjectSeparation: state.task === 'planning' ? 0.12 : 0.09,
    });
  }
}

function resetScores() {
  state.scoreboard.reset();
  state.frames = 0;
  state.seenFrames = 0;
  renderScores();
}

/** The live sandbox is planar physics, so it gets the planar-trained gate; the replay tab is
 *  recorded tabletop/Box2D/Chipmunk episodes, so it gets the tabletop-trained one. Using one
 *  model everywhere would put a domain shift on screen with nothing to label it. */
function selectModel(name) {
  state.modelName = state.models.has(name) ? name : state.modelDefault;
  state.model = state.models.get(state.modelName);
  const label = document.getElementById('model-label');
  if (label) {
    label.textContent =
      `${state.model.weights.source_checkpoint} · `
      + `${state.model.weights.num_parameters.toLocaleString()} parameters`;
  }
  renderDomainNotice();
}

/** Say so, on the page, when the running checkpoint was not trained on what it is being
 *  scored on. Only the tabletop episodes have a matched checkpoint; Box2D and Chipmunk do
 *  not, and a model losing to a one-liner under a domain shift is a weaker claim than a
 *  model losing to it in distribution. Leaving that unlabelled would overstate the result. */
function renderDomainNotice() {
  const notice = document.getElementById('domain-notice');
  if (!notice) return;
  const engine = state.task === 'replay' ? state.replayName : 'planar';
  const matched = (state.task === 'replay' && engine === 'tabletop' && state.modelName === 'tabletop')
    || (state.task !== 'replay' && state.modelName === 'planar');
  notice.hidden = matched;
  if (!matched) {
    notice.textContent =
      `Out of domain: no checkpoint in this project was trained on ${engine}, so the `
      + `${state.modelName} gate is running on a distribution it has not seen. Its score here is `
      + `a domain-shift result, not the paper's in-distribution comparison — read the tabletop `
      + `episode for that.`;
  }
}

function selectTask(task) {
  state.task = task;
  document.querySelectorAll('.tasks button').forEach((button) => {
    button.classList.toggle('active', button.dataset.task === task);
  });
  const info = TASKS[task];
  taskCopy.innerHTML = `<h2>${info.title}</h2><p>${info.copy}</p>`;
  hintEl.textContent = info.hint;
  document.getElementById('count-row').hidden = task !== 'transfer';
  document.getElementById('episode-row').hidden = task !== 'replay';
  document.getElementById('drive-row').hidden = task === 'replay' || task === 'planning';
  state.planSuccess = 0;
  state.planAttempts = 0;
  selectModel(task === 'replay' ? 'tabletop' : 'planar');
  newScene();
  resetScores();
}

async function main() {
  try {
    const bundle = await SparseResidualModel.loadBundle('assets/model.json');
    state.models = bundle.models;
    state.modelDefault = bundle.default;
    const response = await fetch('assets/episodes.json');
    state.episodes = await response.json();
  } catch (error) {
    statusEl.textContent =
      'Could not load the model. This page uses ES modules and fetch, so it needs to be served '
      + 'over http — open it from GitHub Pages, or run `python -m http.server` in docs/.';
    return;
  }

  selectModel('planar');
  document.getElementById('param-count').value =
    state.model.weights.num_parameters.toLocaleString();

  const episodeSelect = document.getElementById('episode');
  for (const [name, episode] of Object.entries(state.episodes)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = `${name} — ${episode.engine} (${episode.num_objects} objects)`;
    episodeSelect.appendChild(option);
  }
  state.replayName = Object.keys(state.episodes)[0];
  episodeSelect.value = state.replayName;
  episodeSelect.addEventListener('change', () => {
    state.replayName = episodeSelect.value;
    state.replayIndex = 0;
    // The tabletop gate is the right model for the MuJoCo episodes; for the other engines
    // every available checkpoint is out of domain, and the page says so rather than hiding it.
    selectModel('tabletop');
    renderDomainNotice();
    resetScores();
  });

  document.querySelectorAll('.tasks button').forEach((button) => {
    button.addEventListener('click', () => selectTask(button.dataset.task));
  });

  const play = document.getElementById('play');
  play.addEventListener('click', () => {
    state.running = !state.running;
    play.textContent = state.running ? 'Pause' : 'Play';
  });
  document.getElementById('reset').addEventListener('click', () => { newScene(); resetScores(); });

  const radius = document.getElementById('radius');
  const radiusOut = document.getElementById('radius-out');
  radius.addEventListener('input', () => {
    state.radius = Number(radius.value);
    radiusOut.textContent = `${state.radius.toFixed(3)} m`;
    // Changing the radius changes what the fitted rules predict, so previously accumulated
    // frames were scored under a different rule. Resetting is the honest thing to do.
    resetScores();
  });

  const count = document.getElementById('count');
  const countOut = document.getElementById('count-out');
  count.addEventListener('input', () => {
    state.numObjects = Number(count.value);
    countOut.textContent = count.value;
    newScene();
    resetScores();
  });

  document.querySelectorAll('input[name="drive"]').forEach((input) => {
    input.addEventListener('change', () => { state.drive = input.value; });
  });

  canvas.addEventListener('pointermove', (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 * WORLD_BOUND - WORLD_BOUND;
    const y = WORLD_BOUND - ((event.clientY - rect.top) / rect.height) * 2 * WORLD_BOUND;
    state.pointer = [x, y];
  });
  canvas.addEventListener('pointerleave', () => { state.pointer = null; });

  selectTask('sandbox');
  requestAnimationFrame(frame);
}

main();
