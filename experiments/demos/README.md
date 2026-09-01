# Blog / demo assets

Visual comparisons built from the **same checkpoints and held-out data** as the
experiments in `experiments/`. Nothing here produces paper numbers — every demo
renders models that were already evaluated, so each figure traces back to a row
in [`experiments/RESULTS.md`](../RESULTS.md).

The environment is state-only (no MuJoCo renderer is used anywhere in the repo),
so scenes are drawn top-down from the planar state by `render2d.py`. For planar
pushing that is not a compromise — a top-down view shows prediction error far
more legibly than a perspective render.

Every demo writes **both** a rendered asset (GIF/PNG) **and** the raw per-frame
trajectory as JSON, so the same run drives a static blog image and an interactive
page without recomputation.

Output goes to `experiments/runs/demos/`.

---

## Run everything

```bash
# 1. Rollout drift — a typical active window, and a high-motion one
python -m experiments.demos.demo_rollout_drift --motion-percentile 50 --name-suffix _typical
python -m experiments.demos.demo_rollout_drift --motion-percentile 90 --name-suffix _active

# 2. Gate-probability field (Round 1 vs Round 2 features)
python -m experiments.demos.demo_gate_field

# 3. Planning episodes (~8 min: the oracle plans ~30x slower than the models)
python -m experiments.demos.demo_planning --num-episodes 12 --oracle-episodes 4

# 4. Cross-count transfer — needs the count-invariant checkpoints once:
for N in 3 5 8; do
  python -m experiments.train_sparse_model \
    --train data/transitions/splits_${N}obj_s0/scale_${N}obj_s0_hard_train.npz \
    --val   data/transitions/splits_${N}obj_s0/scale_${N}obj_s0_hard_val.npz \
    --run-name sparse_invariant_${N}obj_s0 --feature-mode invariant \
    --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed 0
done
python -m experiments.demos.demo_count_transfer

# 5. Bundle the JSON trajectories into one self-contained interactive page
python -m experiments.demos.build_interactive
```

---

## What each asset shows, and how to caption it

### 1. `rollout_drift_3obj_typical.gif` — the lead image

Four panels (ground truth / sparse / dense / no-op), 20-step autoregressive
rollout on held-out trajectories. Dashed grey outlines are the ground truth, so
the gap between a solid box and its outline *is* the error.

Companion static figure: `rollout_drift_3obj_typical_error_curve.png`.

> **Caption.** Twenty steps of closed-loop prediction on a held-out push. The dense
> MLP perturbs every object every step, so error accumulates on the boxes that
> never moved and the scene drifts apart (final error 1.34). The sparse model's
> gate copies unchanged objects verbatim, so its green box stays locked exactly on
> its ground-truth outline (0.26). The no-op reference — "assume nothing moves" —
> lands at 0.35, which the dense model is **3.8× worse than**.

**Selection is not cherry-picked, and say so:** the window is chosen by a stated
rule — among launch points whose ground-truth motion is above a percentile, take
the one whose *sparse* final error is the **median**. The rule and the candidate
distribution are recorded in the summary JSON.

### 1b. `rollout_drift_3obj_active.gif` — the honest counterweight

Same demo on the most active 10% of windows. **Publish this one too.** It shows
sparse's own failure mode, and the summary JSON quantifies it:

| on the top-10% most active windows | sparse | dense |
|---|---|---|
| final-horizon error (median) | **0.47** | 1.62 |
| predicted an object off the table | **86%** | 0% |
| furthest predicted position (median) | 0.57 m | 0.19 m |

The table half-width is 0.34 m. So sparse wins the aggregate metric by ~3.4×
while having the *less bounded* failure mode: when the delta head is wrong it
keeps integrating in the same direction and pushes the moving object off the
table, whereas dense's error is bounded jitter spread over every object.

This is consistent with — and sharpens — the oracle-gate diagnostic already in
`RESULTS.md` ("the changed-object bottleneck is the delta *regression*, not the
gate"), but the off-table rate is **not currently stated there**. It is worth a
line in the paper's limitations, and it makes the blog more credible, not less.

### 2. `gate_field_3obj.png` — the Round 1 → Round 2 diagnosis, visualised

Scene held fixed, pusher swept over a grid; background = gate probability for the
target object, arrows = predicted target motion (green when that push sends it
toward the goal, red when away).

The numbers in the summary JSON confirm the prose claims in `RESULTS.md` quantitatively:

| | Round 1 (`global`) | Round 2 (`contact`) |
|---|---|---|
| gate fires on | **100%** of the grid | 25% of the grid |
| gate probability range | 0.56 – 0.81 (never decisive) | **0.02 – 0.97** |
| predicted-direction circular std | **37.7°** (near-constant drift) | 64.4° (geometry-dependent) |
| motion points goalward, pusher *behind* | +0.54 | +0.20 |
| motion points goalward, pusher *on goal side* | **+0.93** (physically wrong) | **−0.64** (correct) |

That last row is the cleanest single number in the whole demo set: Round 1
predicts the target moves *toward* the goal even when the pusher is on the wrong
side — a constant drift with no dependence on contact geometry, which is exactly
why CEM had no gradient to follow. Round 2 flips the sign correctly.

> **Honest note to keep in the caption:** the sweep sets object velocities to zero,
> because that is the state a sampling planner actually visits. That is
> out-of-distribution for the velocity-keyed Round-1 model *by construction* —
> which is the finding, not a handicap invented for the demo. `--keep-velocity`
> runs the comparison with the dataset's true velocities.

### 3. `planning_3obj.gif` + `planning_3obj_episodes.png`

MPC with each model as the forward simulator. The dotted line is where the
planner's chosen action sequence *imagines* the target will go, anchored at the
target's real position.

The dense panel is the strongest single frame in the set: its imagined target
trajectory sails off across the table while the real red box sits **untouched**.
That is what "planning against a fantasy" looks like.

The bar chart shows **every** episode, so the GIF cannot be mistaken for the
average — and the animated episode is explicitly labelled as one of the minority
the sparse model solves. Demo-scale run (12 episodes, single seed) reproduces the
paper's 3-seed numbers closely:

| | demo (12 eps) | RESULTS.md (3 seeds × 20 eps) |
|---|---|---|
| sparse (contact) | 0.25 | 0.23 ± 0.06 |
| dense (mixed) | 0.00 | 0.00 ± 0.00 |
| oracle | 1.00 (7–13 steps) | 1.00 (11.5 steps) |

Sparse beats dense on **every individual episode**, and several sparse failures
sit just above the 0.05 m success line — near-misses, not wrong-way drift.

*Chart note:* oracle bars are ~0 m tall because it lands on the goal; the dot
marker at each bar top is there so a perfect result doesn't read as missing data.

### 4. `count_transfer_from3obj.gif`

One sparse checkpoint trained on 3 objects, stepping 3-, 5- and 8-object scenes.
The dense model is **actually attempted** on each scene and the real exception is
printed under the panel:

```
5 objects: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x51 and 33x256)
8 objects: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x78 and 33x256)
```

Mean 15-step error over ~60 episodes per count, from the 3-object model:
3-obj **0.273**, 5-obj **0.109**, 8-obj **0.109** — performance is set by the test
count, not the training count, exactly as the transfer matrix reports. Panels are
selected at the same motion percentile in every count so scene activity is
comparable across them.

### 5. `interactive.html`

One self-contained file (~200 KB, no network requests, no build step) with a
canvas replay, scrub bar and per-panel readouts for all four demos. Drop it into
a blog post as an `<iframe>`, or link it. Light/dark aware.

Rebuild after re-running any demo:

```bash
python -m experiments.demos.build_interactive
```

---

## Files

| Module | Produces |
|---|---|
| `render2d.py` | Shared top-down painter, GIF writer, JSON dump |
| `demo_rollout_drift.py` | 4-panel rollout GIF + error curve + selection stats |
| `demo_gate_field.py` | Gate/motion field PNG + quantitative field stats |
| `demo_planning.py` | 3-panel MPC GIF + per-episode bar chart |
| `demo_count_transfer.py` | 3-panel transfer GIF + live dense failure probe |
| `build_interactive.py` | Single-file interactive page from the JSON dumps |
