import { SparseResidualModel, planarSpeeds, unpackState } from './model.js';
import { REST_SPEED, RULE_DESCRIPTIONS, Scoreboard, evaluateRules } from './battery.js';
import { OBJECT_RADIUS, PUSHER_RADIUS, PlanarPushEnv, changedMask } from './planar.js';

const WORLD_BOUND = 0.28;
const BASE_STEP_MS = 60;
const MOTION_THRESHOLD = 0.02;
const HF_BASES = [
  'https://huggingface.co/datasets/ParamThakkar123/sparse_world_models/resolve/main',
  'https://huggingface.co/datasets/ParamTh/sparse_world_models/resolve/main',
];

const TASKS = {
  sandbox: {
    title: 'Sandbox — you drive, you judge',
    copy: 'Live physics! Drag with mouse or use arrows. <strong>Dashed = AI guess, Red ring = actually moved.</strong> '
        + 'Can you make the AI right and the simple rule wrong?',
    hint: 'Mouse: finger follows cursor. Keys: arrows or WASD. Auto: chases goal.',
  },
  replay: {
    title: 'Four engines — same trick everywhere',
    copy: 'MuJoCo, Box2D and Chipmunk cannot run here, so these are recorded episodes, but <strong>live scoring</strong>. The same 1-line rule wins on all four. Billiards is the most fun — things keep rolling.',
    hint: 'Watch already_moving dominate on billiards.',
  },
  transfer: {
    title: 'More boxes — does the trick survive?',
    copy: 'Same weights (trained on 3) driving up to 12 boxes. <strong>Watch still boxes:</strong> AI copies them perfectly, so zero drift. That is its real strength.',
    hint: 'Slide boxes to 12 — who degrades slower?',
  },
  planning: {
    title: 'Make it plan — AI as imagination',
    copy: 'CEM planner imagines futures with the model. With the true simulator it never fails, so failures equal model error.',
    hint: 'Paper: sparse 0.25 vs dense 0.00, PETS beats both at 0.35.',
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
  speed: 1,
  showTrails: false,
  showProbs: true,
  showGrid: true,
  showVel: false,
  sortBy: 'f1',
  trails: [],
  keys: {},
  inspector: null,
};

const canvas = document.getElementById('scene');
const context = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const scoresBody = document.querySelector('#scores tbody');
const framesEl = document.getElementById('frames');
const taskCopy = document.getElementById('task-copy');
const hintEl = document.getElementById('hint');
const tourEl = document.getElementById('tour');
const tourSteps = [
  {title:'Push a box',text:'Drag on the canvas or pick Mouse or Keys. Dashed ghost is the AI prediction.'},
  {title:'Watch the scoreboard',text:'Onset F1 counts resting boxes only. nearest_to_pusher has 0 params and still wins.'},
  {title:'Break it',text:'Try a chain reaction: push one box into another, or set Boxes to 12.'},
];
let tourIdx = 0;
function renderTour(){const t=document.getElementById('tour-title');const tx=document.getElementById('tour-text');const dots=document.getElementById('tour-dots');const nxt=document.getElementById('tour-next');if(!t||!tx||!dots||!nxt||!tourEl) return; t.textContent=tourSteps[tourIdx].title;tx.textContent=tourSteps[tourIdx].text;dots.innerHTML='';tourSteps.forEach((_,i)=>{const d=document.createElement('i');if(i===tourIdx) d.className='on';dots.appendChild(d)});nxt.textContent=tourIdx===tourSteps.length-1?'Lets play!':'Next';}
function openTour(){if(!tourEl) return; tourIdx=0; renderTour(); tourEl.hidden=false; tourEl.style.display='grid';}
function closeTour(){if(!tourEl) return; tourEl.hidden=true; tourEl.style.display='none'; try{localStorage.setItem('tourDone','1')}catch{} }
(function initTourEarly(){
  const ctaPlay=document.getElementById('cta-play');if(ctaPlay) ctaPlay.addEventListener('click',()=>{const s=document.getElementById('scene'); if(s) s.scrollIntoView({behavior:'smooth',block:'center'}); state.running=true; syncPlayButton(); toast('Go! Drag a box');});
  const ctaTour=document.getElementById('cta-tour');if(ctaTour) ctaTour.addEventListener('click', openTour);
  const tNext=document.getElementById('tour-next');if(tNext) tNext.addEventListener('click',()=>{if(tourIdx<tourSteps.length-1){tourIdx++;renderTour()}else{closeTour();toast('Have fun!');const s=document.getElementById('scene'); if(s) s.scrollIntoView({behavior:'smooth',block:'center'}); state.running=true; syncPlayButton();}});
  const tClose=document.getElementById('tour-close');if(tClose) tClose.addEventListener('click', closeTour);
  const tSkip=document.getElementById('tour-skip');if(tSkip) tSkip.addEventListener('click', closeTour);
  if(tourEl) tourEl.addEventListener('click',(e)=>{if(e.target===tourEl) closeTour()});
  document.querySelectorAll('.path').forEach(b=>{b.addEventListener('click',()=>{document.querySelectorAll('.path').forEach(x=>x.classList.remove('active'));b.classList.add('active');const p=b.dataset.path;if(p==='play'){const s=document.getElementById('scene'); if(s) s.scrollIntoView({behavior:'smooth'});}else if(p==='learn'){const s=document.getElementById('explain-strip'); if(s) s.scrollIntoView({behavior:'smooth'}); openTour();}else{const s=document.querySelector('.explainer'); if(s) s.scrollIntoView({behavior:'smooth'});}})});
  const toggleStrip=document.getElementById('toggle-strip');if(toggleStrip) toggleStrip.addEventListener('click',()=>{const s=document.querySelector('.comic');const f=document.querySelector('.strip-foot');if(!s) return; const hid=s.hidden; s.hidden=!hid; if(f) f.hidden=!hid; toggleStrip.textContent=hid?'Hide':'Show';});
})();

function toCanvas(x, y) {
  const scale = canvas.width / (2 * WORLD_BOUND);
  return [(x + WORLD_BOUND) * scale, (WORLD_BOUND - y) * scale];
}
function styleValue(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function worldFromPointer(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * 2 * WORLD_BOUND - WORLD_BOUND;
  const y = WORLD_BOUND - ((event.clientY - rect.top) / rect.height) * 2 * WORLD_BOUND;
  return [x, y];
}
function drawGrid(scale) {
  if (!state.showGrid) return;
  context.strokeStyle = 'color-mix(in srgb, var(--line) 55%, transparent)';
  context.lineWidth = 1;
  context.setLineDash([2, 6]);
  for (let g = -0.2; g <= 0.2; g += 0.1) {
    const [x0, y0] = toCanvas(g, -0.26);
    const [x1, y1] = toCanvas(g, 0.26);
    context.beginPath(); context.moveTo(x0, y0); context.lineTo(x1, y1); context.stroke();
    const [hx0, hy0] = toCanvas(-0.26, g);
    const [hx1, hy1] = toCanvas(0.26, g);
    context.beginPath(); context.moveTo(hx0, hy0); context.lineTo(hx1, hy1); context.stroke();
  }
  context.setLineDash([]);
}
function drawScene({ poses, pusher, goal, gates, probs, predicted, truth, velocities }) {
  const scale = canvas.width / (2 * WORLD_BOUND);
  const dpr = window.devicePixelRatio || 1;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
  const [tableX, tableY] = toCanvas(-0.26, 0.26);
  context.fillStyle = styleValue('--bg');
  context.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid(scale);
  context.strokeStyle = styleValue('--line');
  context.lineWidth = 2;
  context.strokeRect(tableX / dpr, tableY / dpr, 0.52 * scale / dpr, 0.52 * scale / dpr);
  if (goal) {
    const [gx, gy] = toCanvas(goal[0], goal[1]);
    context.beginPath();
    context.arc(gx / dpr, gy / dpr, 0.03 * scale / dpr, 0, Math.PI * 2);
    context.strokeStyle = styleValue('--goal');
    context.setLineDash([4, 4]);
    context.lineWidth = 2;
    context.stroke();
    context.setLineDash([]);
  }
  if (state.showTrails && state.trails.length) {
    state.trails.forEach((trail, idx) => {
      if (trail.length < 2) return;
      context.strokeStyle = styleValue('--ghost');
      context.lineWidth = 1.5;
      context.globalAlpha = 0.35;
      context.beginPath();
      trail.forEach((p, i) => {
        const [cx, cy] = toCanvas(p[0], p[1]);
        if (i === 0) context.moveTo(cx / dpr, cy / dpr);
        else context.lineTo(cx / dpr, cy / dpr);
      });
      context.stroke();
      context.globalAlpha = 1;
    });
  }
  if (predicted) {
    context.strokeStyle = styleValue('--ghost');
    context.setLineDash([5, 4]);
    context.lineWidth = 2;
    predicted.forEach((pose) => drawBox(pose, scale, null, dpr));
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
    drawBox(pose, scale, true, dpr);
    if (state.showProbs && probs) {
      const [px, py] = toCanvas(pose[0], pose[1]);
      context.fillStyle = styleValue('--ink-soft');
      context.font = '11px ui-monospace, monospace';
      context.textAlign = 'center';
      context.fillText(probs[index].toFixed(2), px / dpr, py / dpr - 0.035 * scale / dpr);
    }
    if (state.showVel && velocities) {
      const v = velocities[index];
      if (v) {
        const [px, py] = toCanvas(pose[0], pose[1]);
        const vx = v[3] || 0;
        const vy = v[4] || 0;
        const mag = Math.hypot(vx, vy);
        if (mag > 1e-4) {
          context.strokeStyle = styleValue('--gate');
          context.lineWidth = 1.5;
          context.beginPath();
          context.moveTo(px / dpr, py / dpr);
          context.lineTo((px + vx * 2 * scale) / dpr, (py - vy * 2 * scale) / dpr);
          context.stroke();
        }
      }
    }
    if (state.inspector && state.inspector.index === index) {
      const [px, py] = toCanvas(pose[0], pose[1]);
      context.strokeStyle = styleValue('--accent');
      context.lineWidth = 2;
      context.setLineDash([3, 3]);
      context.beginPath();
      context.arc(px / dpr, py / dpr, OBJECT_RADIUS * scale / dpr + 4, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
    }
  });
  const [px, py] = toCanvas(pusher[0], pusher[1]);
  context.beginPath();
  context.arc(px / dpr, py / dpr, PUSHER_RADIUS * scale / dpr, 0, Math.PI * 2);
  context.fillStyle = styleValue('--ink');
  context.fill();
  if (state.drive === 'keys') {
    context.fillStyle = styleValue('--accent');
    context.font = '10px ui-sans-serif, sans-serif';
    context.textAlign = 'center';
    context.fillText('KEYS', px / dpr, py / dpr - 18);
  }
}
function drawBox(pose, scale, filled, dpr) {
  const [x, y] = toCanvas(pose[0], pose[1]);
  const half = OBJECT_RADIUS * scale / dpr;
  context.save();
  context.translate(x / dpr, y / dpr);
  context.rotate(-pose[2]);
  context.beginPath();
  context.rect(-half, -half, 2 * half, 2 * half);
  if (filled) context.fill();
  context.stroke();
  context.restore();
}
function passesMotionFilter(before, after) {
  return before.some((pose, i) => Math.hypot(after[i][0] - pose[0], after[i][1] - pose[1]) > MOTION_THRESHOLD);
}
function scoreFrame(stateVector, action, numObjects, truthMask) {
  const speeds = planarSpeeds(stateVector, numObjects);
  const atRest = speeds.map((speed) => speed <= REST_SPEED);
  const { rules } = evaluateRules(stateVector, action, numObjects, state.radius, state.model.constants);
  for (const [name, prediction] of Object.entries(rules)) {
    state.scoreboard.update(name, prediction, truthMask, atRest);
  }
  const { gates } = state.lastPrediction;
  state.scoreboard.update('model', gates, truthMask, atRest);
  state.frames += 1;
}
function renderScores() {
  let rows = state.scoreboard.rows();
  if (state.sortBy === 'onset') rows = [...rows].sort((a, b) => b.onsetF1 - a.onsetF1 || b.f1 - a.f1);
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
      `<td>${row.support ? row.onsetF1.toFixed(3) : '-'}</td>`;
    if (!row.isModel && RULE_DESCRIPTIONS[row.name]) tr.title = RULE_DESCRIPTIONS[row.name];
    scoresBody.appendChild(tr);
  }
  framesEl.textContent = state.seenFrames ? `${state.frames} scored / ${state.seenFrames} seen` : '0 frames';
}
function autoAction(env) {
  const target = env.objectXY[env.config.targetObject];
  const goal = env.config.goalXY;
  const toGoal = [goal[0] - target[0], goal[1] - target[1]];
  const norm = Math.hypot(toGoal[0], toGoal[1]) || 1;
  const behind = [target[0] - (toGoal[0] / norm) * 0.06, target[1] - (toGoal[1] / norm) * 0.06];
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
function keysAction() {
  let dx = 0, dy = 0;
  if (state.keys['ArrowUp'] || state.keys['w'] || state.keys['W']) dy += 1;
  if (state.keys['ArrowDown'] || state.keys['s'] || state.keys['S']) dy -= 1;
  if (state.keys['ArrowLeft'] || state.keys['a'] || state.keys['A']) dx -= 1;
  if (state.keys['ArrowRight'] || state.keys['d'] || state.keys['D']) dx += 1;
  const mag = Math.hypot(dx, dy);
  if (!mag) return [0, 0];
  const scale = state.keys['Shift'] ? 0.5 : 1;
  return [(dx / mag) * scale, (dy / mag) * scale];
}
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
function rollout(env, sequence, numObjects, goal, target) {
  let vector = env.state().slice();
  let cost = 0;
  const constants = state.model.constants;
  for (let t = 0; t < sequence.length; t += 1) {
    const { poses } = state.model.step(vector, sequence[t], numObjects);
    const unpacked = unpackState(vector, numObjects);
    const pusher = [
      Math.max(-constants.pusher_bound, Math.min(constants.pusher_bound, unpacked.pusher[0] + Math.max(-1, Math.min(1, sequence[t][0])) * constants.pusher_action_scale)),
      Math.max(-constants.pusher_bound, Math.min(constants.pusher_bound, unpacked.pusher[1] + Math.max(-1, Math.min(1, sequence[t][1])) * constants.pusher_action_scale)),
    ];
    vector = rebuildState(pusher, poses, unpacked.velocities, unpacked.goal);
    const distance = Math.hypot(poses[target][0] - goal[0], poses[target][1] - goal[1]);
    const proximity = Math.hypot(pusher[0] - poses[target][0], pusher[1] - poses[target][1]);
    cost += distance + 0.3 * proximity;
  }
  const finalPoses = unpackState(vector, numObjects).poses;
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
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  const factor = Math.sqrt((-2 * Math.log(s)) / s);
  spare = v * factor; return u * factor;
}
function standardDeviation(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length);
}
function pushTrails(poses) {
  if (!state.showTrails) return;
  if (!state.trails.length || state.trails[0].length === 0 || state.trails.length !== poses.length) {
    state.trails = poses.map((p) => [[p[0], p[1]]]);
    return;
  }
  poses.forEach((p, i) => {
    state.trails[i].push([p[0], p[1]]);
    if (state.trails[i].length > 24) state.trails[i].shift();
  });
}
function stepLive() {
  const env = state.env;
  const numObjects = env.config.numObjects;
  const before = env.observation().poses.map((pose) => [...pose]);
  const vector = env.state();
  let action;
  if (state.task === 'planning') action = planAction(env);
  else if (state.drive === 'mouse') action = mouseAction(env);
  else if (state.drive === 'keys') action = keysAction(env);
  else action = autoAction(env);
  state.lastPrediction = state.model.step(vector, action, numObjects);
  env.step(action);
  const after = env.observation().poses;
  const truth = changedMask(before, after);
  state.lastTruth = truth;
  state.seenFrames += 1;
  if (passesMotionFilter(before, after)) scoreFrame(vector, action, numObjects, truth);
  pushTrails(after);
  if (state.task === 'planning') {
    const target = env.config.targetObject;
    const distance = Math.hypot(after[target][0] - env.config.goalXY[0], after[target][1] - env.config.goalXY[1]);
    if (distance < 0.05) { state.planSuccess += 1; state.planAttempts += 1; newScene(); return; }
    if (env.stepCount >= 60) { state.planAttempts += 1; newScene(); return; }
  }
  const unpacked = unpackState(env.state(), numObjects);
  drawScene({ poses: after, pusher: env.pusher, goal: env.config.goalXY, gates: state.lastPrediction.gates, probs: state.lastPrediction.probs, predicted: state.lastPrediction.poses, truth, velocities: [unpacked.velocities] && unpacked.velocities ? unpacked.velocities : null });
}
function stepReplay() {
  const episode = state.episodes[state.replayName];
  const numObjects = episode.num_objects;
  if (state.replayIndex >= episode.state.length) state.replayIndex = 0;
  const vector = episode.state[state.replayIndex];
  const action = episode.action[state.replayIndex];
  const truth = episode.target_mask[state.replayIndex];
  state.lastPrediction = state.model.step(vector, action, numObjects);
  const { poses, pusher, goal, velocities } = unpackState(vector, numObjects);
  state.seenFrames += 1;
  const next = episode.state[state.replayIndex + 1];
  if (next === undefined || passesMotionFilter(poses, unpackState(next, numObjects).poses)) {
    scoreFrame(vector, action, numObjects, truth);
  }
  pushTrails(poses);
  drawScene({ poses, pusher, goal, gates: state.lastPrediction.gates, probs: state.lastPrediction.probs, predicted: state.lastPrediction.poses, truth, velocities });
  state.replayIndex += 1;
}
let lastStep = 0;
let fpsLast = performance.now();
let fpsCount = 0;
function frame(timestamp) {
  fpsCount += 1;
  if (timestamp - fpsLast > 500) {
    const fps = Math.round((fpsCount * 1000) / (timestamp - fpsLast));
    const el = document.getElementById('fps');
    if (el) el.textContent = `${fps} fps`;
    fpsLast = timestamp; fpsCount = 0;
  }
  const stepMs = BASE_STEP_MS / state.speed;
  if (state.running && timestamp - lastStep > stepMs) {
    lastStep = timestamp;
    try {
      if (state.task === 'replay') stepReplay();
      else stepLive();
      renderScores();
      updateStatus();
      updateInspector();
    } catch (error) {
      statusEl.textContent = `Stopped: ${error.message}`;
      state.running = false;
      syncPlayButton();
    }
  }
  requestAnimationFrame(frame);
}
function toast(msg){const s=document.getElementById('toasts');if(!s)return;const t=document.createElement('div');t.className='toast';t.textContent=msg;s.appendChild(t);setTimeout(()=>t.remove(),2200)}
function burstConfetti(){const c=document.getElementById('confetti');if(!c)return;for(let i=0;i<18;i++){const el=document.createElement('i');el.style.left=Math.random()*100+'vw';el.style.top='-10px';el.style.background=['#ff5a2b','#7c5cff','#f5b400','#0e9b8b'][i%4];el.style.transform=`rotate(${Math.random()*360}deg)`;el.style.animationDelay=Math.random()*0.3+'s';c.appendChild(el);setTimeout(()=>el.remove(),1400)}}
const challenges={push:false,chain:false,trick:false};let lastTrickFrame=-999;let insightCooldown=0;
function updateChallenges(){const any=state.lastTruth&&state.lastPrediction;if(!any)return;const truth=state.lastTruth;const moved=truth.some(v=>v===1);const twoMoved=truth.filter(v=>v===1).length>=2;if(moved&&!challenges.push){challenges.push=true;document.querySelector('[data-ch="push"]').checked=true;toast('First push! You moved a box');}
if(twoMoved&&!challenges.chain){challenges.chain=true;document.querySelector('[data-ch="chain"]').checked=true;toast('Chain reaction! Box hit box');burstConfetti();}
const rows=state.scoreboard.rows();const best=rows.find(r=>!r.isModel);const model=rows.find(r=>r.isModel);if(best&&model&&model.onsetF1+1e-9<best.onsetF1&&!challenges.trick&&state.frames>12&&state.seenFrames-lastTrickFrame>30){challenges.trick=true;document.querySelector('[data-ch="trick"]').checked=true;toast(`${best.name} beat AI on onset`);burstConfetti();lastTrickFrame=state.seenFrames;}
document.getElementById('ch-progress').textContent=`${Object.values(challenges).filter(Boolean).length}/3`;if(Object.values(challenges).every(Boolean)&&insightCooldown===0){insightCooldown=1;toast('All challenges done — you get it! Now try More Boxes');}}
function updateStatus() {
  updateChallenges();
  const insight=document.getElementById('insight');
  if (state.task === 'planning' && state.planAttempts > 0) {
    const rate = (state.planSuccess / state.planAttempts).toFixed(2);
    statusEl.textContent = `Planning: ${state.planSuccess}/${state.planAttempts} solved (rate ${rate}) — paper says 0.25, PETS 0.35.`;
    if(insight) insight.textContent='';
    return;
  }
  const rows = state.scoreboard.rows();
  const best = rows.find((row) => !row.isModel);
  const model = rows.find((row) => row.isModel);
  if (!best || !model || state.frames < 15) { statusEl.textContent = 'Collecting frames — push something!'; if(insight) insight.textContent=''; return; }
  const mOn= model.onsetF1, bOn=best.onsetF1;
  const margin = mOn - bOn;
  if(margin <= 0){ statusEl.textContent = `${best.name} (0 params) beats AI on onset ${bOn.toFixed(3)} vs ${mOn.toFixed(3)} — by ${(-margin).toFixed(3)}.`; if(insight) insight.textContent='That is the point: "nearest to finger" predicts contact. Try a chain reaction to confuse it.';}
  else { statusEl.textContent = `AI leads ${best.name} by ${margin.toFixed(3)} on onset — rare! Keep playing, can you flip it?`; if(insight) insight.textContent='AI is ahead on this scene — try More Boxes or a chain hit.'; }
}
function updateInspector() {
  const el = document.getElementById('inspector');
  if (!el || !state.lastPrediction || !state.env) { if (el) el.hidden = true; return; }
  if (!state.inspector) { el.hidden = true; return; }
  const idx = state.inspector.index;
  const prob = state.lastPrediction.probs[idx];
  const gate = state.lastPrediction.gates[idx];
  const truth = state.lastTruth ? state.lastTruth[idx] : null;
  const dist = state.inspector.distance.toFixed(3);
  const speed = state.inspector.speed.toFixed(4);
  el.innerHTML = `obj ${idx}<br>dist to pusher ${dist} m<br>speed ${speed} m/s<br>gate ${gate} prob ${prob.toFixed(2)}<br>changed ${truth === 1 ? 'yes' : 'no'}`;
  el.hidden = false;
}
function syncPlayButton() {
  const play = document.getElementById('play');
  const overlay = document.getElementById('paused-overlay');
  if (play) play.textContent = state.running ? 'Pause' : 'Play';
  if (overlay) overlay.hidden = state.running;
}
function newScene() {
  state.trails = [];
  if (state.task === 'replay') { state.replayIndex = 0; }
  else {
    state.env = new PlanarPushEnv({ numObjects: state.numObjects, seed: Math.floor(Math.random() * 1e6), minObjectSeparation: state.task === 'planning' ? 0.12 : 0.09 });
  }
  pushUrlState();
}
function resetScores() { state.scoreboard.reset(); state.frames = 0; state.seenFrames = 0; state.trails = []; renderScores(); }
function selectModel(name) {
  state.modelName = state.models.has(name) ? name : state.modelDefault;
  state.model = state.models.get(state.modelName);
  const label = document.getElementById('model-label');
  if (label) label.textContent = `${state.model.weights.source_checkpoint} · ${state.model.weights.num_parameters.toLocaleString()} parameters`;
  renderDomainNotice();
}
function renderDomainNotice() {
  const notice = document.getElementById('domain-notice');
  if (!notice) return;
  const engine = state.task === 'replay' ? state.replayName : 'planar';
  const matched = (state.task === 'replay' && engine === 'tabletop' && state.modelName === 'tabletop') || (state.task !== 'replay' && state.modelName === 'planar');
  notice.hidden = matched;
  if (!matched) notice.textContent = `Out of domain: no checkpoint in this project was trained on ${engine}, so the ${state.modelName} gate is running on a distribution it has not seen. Its score here is a domain shift result, not the paper in distribution comparison. Read the tabletop episode for that.`;
}
function selectTask(task) {
  state.task = task;
  document.querySelectorAll('.tasks button').forEach((button) => { button.classList.toggle('active', button.dataset.task === task); });
  const info = TASKS[task];
  taskCopy.innerHTML = `<h2>${info.title}</h2><p>${info.copy}</p>`;
  hintEl.textContent = info.hint;
  document.getElementById('count-row').hidden = task !== 'transfer';
  document.getElementById('episode-row').hidden = task !== 'replay';
  document.getElementById('drive-row').hidden = task === 'replay' || task === 'planning';
  document.getElementById('keys-hint').hidden = state.drive !== 'keys';
  state.planSuccess = 0; state.planAttempts = 0;
  selectModel(task === 'replay' ? 'tabletop' : 'planar');
  newScene(); resetScores(); pushUrlState();
}
function pushUrlState() {
  const url = new URL(window.location);
  url.searchParams.set('task', state.task);
  url.searchParams.set('count', String(state.numObjects));
  url.searchParams.set('radius', String(state.radius));
  url.searchParams.set('speed', String(state.speed));
  url.searchParams.set('drive', state.drive);
  history.replaceState(null, '', url);
}
function applyUrlState() {
  const p = new URLSearchParams(window.location.search);
  if (p.get('task') && TASKS[p.get('task')]) state.task = p.get('task');
  if (p.get('count')) state.numObjects = Math.max(2, Math.min(12, Number(p.get('count')) || 3));
  if (p.get('radius')) state.radius = Math.max(0.02, Math.min(0.22, Number(p.get('radius')) || 0.07));
  if (p.get('speed')) state.speed = Math.max(0.25, Math.min(3, Number(p.get('speed')) || 1));
  if (p.get('drive') && ['auto', 'mouse', 'keys'].includes(p.get('drive'))) state.drive = p.get('drive');
}
async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label} not found at ${url}`);
  return res.json();
}
async function loadBundleWithFallback() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get('modelUrl');
  const candidates = [];
  if (override) candidates.push(override);
  candidates.push('assets/model.json');
  for (const base of HF_BASES) candidates.push(`${base}/docs/assets/model.json`);
  let lastError = null;
  for (const url of candidates) {
    try {
      const bundle = await SparseResidualModel.loadBundle(url);
      return bundle;
    } catch (e) { lastError = e; }
  }
  throw lastError || new Error('could not load model bundle');
}
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const size = Math.round(rect.width * dpr);
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
}
async function main() {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  applyUrlState();
  const progress = document.getElementById('progress');
  const progressBar = document.getElementById('progress-bar');
  if (progress) progress.hidden = false;
  if (progressBar) progressBar.style.width = '30%';
  try {
    const bundle = await loadBundleWithFallback();
    state.models = bundle.models; state.modelDefault = bundle.default;
    if (progressBar) progressBar.style.width = '60%';
    let episodes = null;
    try { episodes = await fetchWithProgress('assets/episodes.json', 'episodes'); }
    catch {
      let lastErr = null;
      for (const base of HF_BASES) {
        try { episodes = await fetchWithProgress(`${base}/docs/assets/episodes.json`, 'episodes'); break; } catch (e) { lastErr = e; }
      }
      if (!episodes) throw lastErr;
    }
    state.episodes = episodes;
    if (progressBar) progressBar.style.width = '100%';
    setTimeout(() => { if (progress) progress.hidden = true; }, 400);
  } catch (error) {
    statusEl.textContent = 'Could not load the model. This page uses ES modules and fetch, so it needs to be served over http. Open it from GitHub Pages, or run `python -m http.server` in docs/.';
    if (progress) progress.hidden = true;
    return;
  }
  selectModel('planar');
  const saved = JSON.parse(localStorage.getItem('demoPrefs') || '{}');
  if (saved.showTrails) state.showTrails = saved.showTrails;
  if (saved.showGrid === false) state.showGrid = false;
  if (saved.showVel) state.showVel = saved.showVel;
  if (saved.showProbs === false) state.showProbs = false;
  document.getElementById('param-count').textContent = state.model.weights.num_parameters.toLocaleString();
  const episodeSelect = document.getElementById('episode');
  for (const [name, episode] of Object.entries(state.episodes)) {
    const option = document.createElement('option');
    option.value = name; option.textContent = `${name}: ${episode.engine} (${episode.num_objects} objects)`;
    episodeSelect.appendChild(option);
  }
  state.replayName = Object.keys(state.episodes)[0];
  episodeSelect.value = state.replayName;
  episodeSelect.addEventListener('change', () => {
    state.replayName = episodeSelect.value; state.replayIndex = 0;
    selectModel('tabletop'); renderDomainNotice(); resetScores();
  });
  document.querySelectorAll('.tasks button').forEach((button) => {
    button.addEventListener('click', () => selectTask(button.dataset.task));
  });
  const play = document.getElementById('play');
  play.addEventListener('click', () => { state.running = !state.running; syncPlayButton(); });
  document.getElementById('reset').addEventListener('click', () => { newScene(); resetScores(); });
  const doStep = () => {
    const was = state.running; state.running = true;
    if (state.task === 'replay') stepReplay(); else stepLive();
    renderScores(); updateStatus(); updateInspector();
    state.running = was; syncPlayButton();
  };
  document.getElementById('step').addEventListener('click', doStep);
  document.getElementById('btn-step').addEventListener('click', doStep);
  document.getElementById('share').addEventListener('click', async () => {
    pushUrlState();
    const url = window.location.href;
    try { await navigator.clipboard.writeText(url); statusEl.textContent = 'Link copied to clipboard.'; setTimeout(updateStatus, 1500); } catch { statusEl.textContent = url; }
  });
  const radius = document.getElementById('radius');
  const radiusOut = document.getElementById('radius-out');
  radius.value = state.radius; radiusOut.textContent = `${state.radius.toFixed(3)} m`;
  radius.addEventListener('input', () => { state.radius = Number(radius.value); radiusOut.textContent = `${state.radius.toFixed(3)} m`; resetScores(); pushUrlState(); });
  const count = document.getElementById('count');
  const countOut = document.getElementById('count-out');
  count.value = state.numObjects; countOut.textContent = count.value;
  count.addEventListener('input', () => { state.numObjects = Number(count.value); countOut.textContent = count.value; newScene(); resetScores(); });
  const speed = document.getElementById('speed');
  const speedOut = document.getElementById('speed-out');
  speed.value = state.speed; speedOut.textContent = `${state.speed}x`;
  speed.addEventListener('input', () => { state.speed = Number(speed.value); speedOut.textContent = `${state.speed}x`; pushUrlState(); });
  const checkTrails = document.getElementById('check-trails');
  const btnTrails = document.getElementById('btn-trails');
  checkTrails.checked = state.showTrails; if (btnTrails) btnTrails.setAttribute('aria-pressed', String(state.showTrails));
  const toggleTrails = () => { state.showTrails = !state.showTrails; checkTrails.checked = state.showTrails; if (btnTrails) btnTrails.setAttribute('aria-pressed', String(state.showTrails)); if (!state.showTrails) state.trails = []; persistPrefs(); };
  checkTrails.addEventListener('change', toggleTrails); if (btnTrails) btnTrails.addEventListener('click', toggleTrails);
  const checkVel = document.getElementById('check-vel');
  checkVel.checked = state.showVel; checkVel.addEventListener('change', () => { state.showVel = checkVel.checked; persistPrefs(); });
  const checkGrid = document.getElementById('check-grid');
  checkGrid.checked = state.showGrid; checkGrid.addEventListener('change', () => { state.showGrid = checkGrid.checked; persistPrefs(); });
  const btnProbs = document.getElementById('btn-probs');
  const syncProbs = () => { if (btnProbs) btnProbs.setAttribute('aria-pressed', String(state.showProbs)); };
  syncProbs();
  if (btnProbs) btnProbs.addEventListener('click', () => { state.showProbs = !state.showProbs; syncProbs(); persistPrefs(); });
  document.getElementById('btn-screenshot').addEventListener('click', () => {
    const a = document.createElement('a'); a.download = `demo-${state.task}-${Date.now()}.png`; a.href = canvas.toDataURL('image/png'); a.click();
  });
  document.getElementById('sort-f1').addEventListener('click', () => { state.sortBy = 'f1'; renderScores(); });
  document.getElementById('sort-onset').addEventListener('click', () => { state.sortBy = 'onset'; renderScores(); });
  document.getElementById('copy-scores').addEventListener('click', async () => {
    const rows = state.scoreboard.rows(); const csv = ['rule,params,f1,onsetF1'].concat(rows.map(r => `${r.name},${r.isModel ? state.model.weights.num_parameters : r.parameters},${r.f1.toFixed(3)},${r.support ? r.onsetF1.toFixed(3) : ''}`)).join('\n');
    try { await navigator.clipboard.writeText(csv); statusEl.textContent = 'Scores copied as CSV.'; setTimeout(updateStatus, 1500); } catch { statusEl.textContent = csv.slice(0, 80); }
  });
  document.querySelectorAll('input[name="drive"]').forEach((input) => {
    if (input.value === state.drive) input.checked = true;
    input.addEventListener('change', () => { state.drive = input.value; document.getElementById('keys-hint').hidden = state.drive !== 'keys'; pushUrlState(); });
  });
  document.getElementById('keys-hint').hidden = state.drive !== 'keys';
  function persistPrefs() { localStorage.setItem('demoPrefs', JSON.stringify({ showTrails: state.showTrails, showVel: state.showVel, showGrid: state.showGrid, showProbs: state.showProbs })); }
  canvas.addEventListener('pointermove', (event) => {
    state.pointer = worldFromPointer(event);
    const env = state.env;
    const poses = env ? env.observation().poses : (state.episodes && state.episodes[state.replayName] ? unpackState(state.episodes[state.replayName].state[state.replayIndex] || state.episodes[state.replayName].state[0], state.episodes[state.replayName].num_objects).poses : null);
    const pusher = env ? env.pusher : (state.episodes && state.episodes[state.replayName] ? unpackState(state.episodes[state.replayName].state[state.replayIndex] || state.episodes[state.replayName].state[0], state.episodes[state.replayName].num_objects).pusher : null);
    if (!poses || !pusher) return;
    let best = null; let bestDist = Infinity;
    poses.forEach((pose, i) => {
      const d = Math.hypot(state.pointer[0] - pose[0], state.pointer[1] - pose[1]);
      if (d < bestDist && d < 0.06) { bestDist = d; best = i; }
    });
    if (best !== null) {
      const speedVal = state.env ? Math.hypot(state.env.objectVel[best][0], state.env.objectVel[best][1]) : 0;
      const distToPusher = Math.hypot(poses[best][0] - pusher[0], poses[best][1] - pusher[1]);
      state.inspector = { index: best, distance: distToPusher, speed: speedVal };
    } else state.inspector = null;
  });
  canvas.addEventListener('pointerleave', () => { state.pointer = null; state.inspector = null; const el = document.getElementById('inspector'); if (el) el.hidden = true; });
  canvas.addEventListener('pointerdown', (e) => { if (state.drive === 'mouse') state.pointer = worldFromPointer(e); });
  window.addEventListener('keydown', (e) => {
    state.keys[e.key] = true;
    if (e.code === 'Space') { e.preventDefault(); state.running = !state.running; syncPlayButton(); }
    else if (e.key.toLowerCase() === 'r') { newScene(); resetScores(); }
    else if (e.key === '.') { const was = state.running; state.running = true; if (state.task === 'replay') stepReplay(); else stepLive(); renderScores(); updateStatus(); state.running = was; syncPlayButton(); }
    else if (e.key.toLowerCase() === 't') { state.showTrails = !state.showTrails; document.getElementById('check-trails').checked = state.showTrails; const b = document.getElementById('btn-trails'); if (b) b.setAttribute('aria-pressed', String(state.showTrails)); if (!state.showTrails) state.trails = []; }
    else if (e.key.toLowerCase() === 'p') { state.showProbs = !state.showProbs; const b = document.getElementById('btn-probs'); if (b) b.setAttribute('aria-pressed', String(state.showProbs)); }
    else if (e.key.toLowerCase() === 'g') { state.showGrid = !state.showGrid; document.getElementById('check-grid').checked = state.showGrid; }
    else if (['1','2','3','4'].includes(e.key)) { const tasks = ['sandbox','replay','transfer','planning']; selectTask(tasks[Number(e.key)-1]); }
  });
  window.addEventListener('keyup', (e) => { state.keys[e.key] = false; });
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'light' ? 'dark' : cur === 'dark' ? 'light' : (matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark');
    if (next === 'light') document.documentElement.setAttribute('data-theme', 'light');
    else if (next === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    else document.documentElement.removeAttribute('data-theme');
  });
  try{ if(!localStorage.getItem('tourDone')) setTimeout(openTour,900);}catch{}
  selectTask(state.task);
  syncPlayButton();
  state.sortBy='onset';renderScores();
  requestAnimationFrame(frame);
}
main();
