# Change-Detection Benchmarks for Object-Centric World Models Are Degenerate

Code, data generators, and the trivial-baseline **audit battery** behind an ongoing study of
change prediction in object-centric world models. An earlier version of this work appeared at
WORLDS @ IROS 2026; **its central claims did not survive proper baselines**, and this
repository now documents both what failed and what replaced it.

## What this repository claims, as of 2026-08-18

> Change-detection benchmarks in object-centric manipulation are **pervasively degenerate**.
> Three successive benchmarks — motion-filtered, onset-filtered, interaction-filtered — each
> remove their predecessor's shortcut, and each is then won by a *different* one-line rule at
> F1 **0.93–0.96**. Five published object-centric world models plus our own lose to all three.
> The field does not run trivial baselines, and every "our model predicts what changes" claim
> in this literature is unvalidated until it does.

| benchmark | shortcut it removes | one-liner that wins instead | best trivial F1 | best learned gate |
|---|---|---|---|---|
| motion | — | `already_moving` (velocity) | **0.927** | 0.825 |
| onset | velocity | `nearest_to_pusher` (**0 parameters**) | **0.961** | 0.694 |
| interaction | geometry | `moving_or_near_mover` (propagated kinematics) | **0.955** | 0.772 |

The deliverable is `experiments/onset_shortcut_audit.py` — eleven parameter-free or
one-parameter rules, five of them invented *against* the interaction benchmark before it was
built — plus the methodological requirement that any new benchmark be audited with rules
invented against **it**, not against the one it replaces.

The degeneracy is structural rather than a bug in any filter: with a single point-like pusher,
the label is recoverable from one scalar (speed, or distance to the pusher, or distance to the
nearest mover), and a filter only chooses which. See `experiments/RESULTS.md`,
"Why it happens, and what a non-degenerate benchmark would need".

## What survives of the architecture claim

The repository still implements the sparse/residual model the earlier paper was about — a
per-object change **gate** plus a residual **delta head**, so objects the gate leaves off are
carried forward verbatim:

```
p̂ᵢ(t+1) = pᵢ(t) + gᵢ · Δᵢ ,    gᵢ ∈ {0,1} sampled with a Gumbel straight-through estimator
```

One mechanism claim survives contact with real baselines: **error suppression on unchanged
objects**, where the gap over a dense monolith *grows* with object count (dense unchanged-L2
0.271 → 0.438 monotone from N=3 to N=20 while the sparse model stays flat), replicated in a
second environment and unmatched by any of the five published baselines. Two related results
also stand: the learned gate works as a local causal mask for counterfactual splicing, and
velocity-free contact featurisation is what buys genuine onset prediction.

**Three claims from the workshop paper are refuted, and are retained in `RESULTS.md` as
refutations rather than deleted:** that the onset benchmark is a *corrected* benchmark (it
removes the velocity shortcut and nothing more); that only the sparse model plans (four of
five published models plan, and PETS beats it, 0.350 vs 0.250); and every headline number
computed on the leaky splits.

**[`experiments/RESULTS.md`](experiments/RESULTS.md) is the authoritative results
document** — read its "Status map" section before quoting any number from anywhere, including
from this README. [`SUBMISSION_RUNBOOK.md`](SUBMISSION_RUNBOOK.md) is the pre-submission
checklist.

## Run it yourself, in a browser

[`docs/`](docs/) is a GitHub Pages site that runs the **actual trained weights** against live
physics and scores them against the eleven trivial rules, frame by frame. Four tasks: a
sandbox you drive yourself, replay across the four engines, count transfer with one
checkpoint, and CEM planning through the model. No server and no CDN — the JavaScript forward
pass is checked against PyTorch to 1e-4 by `tests/test_web_export.py`.

```bash
python -m http.server 8000 --directory docs     # then http://localhost:8000/
```

A claim as uncomfortable as "a zero-parameter rule beats the model" is better checked than
asserted, so the page lets a reader do exactly that.

## Install

```bash
pip install -r requirements.txt
```

MuJoCo is needed only to *generate* data and to run the planning oracle; training and
evaluation on existing `.npz` files work without it.

## Repository layout

| Path | What's in it |
|---|---|
| `models/` | `sparse_gating.py` (change gate), `sparse_residual.py` (gate + delta head), `dense_predictor.py` (monolithic baseline), `envs/mujoco_tabletop.py` (procedural pushing env), `layout.py` (state layout) |
| `models/literature_baselines.py` | GNS/DPI-Net, C-SWM, SlotFormer, PETS, NPS — published models implemented from their papers |
| `models/envs/box2d_billiards.py`, `models/envs/pymunk_clutter.py` | Third-party-engine domains (Box2D, Chipmunk2D) |
| `models/envs/renderer.py` | State → image, so any existing dataset can be replayed from pixels |
| `models/slot_attention.py`, `models/keypoint_encoder.py` | The two unsupervised perception front ends |
| `models/checkpoints/` | All trained checkpoints referenced by the paper |
| `experiments/` | One script per experiment; see the table below |
| `experiments/RESULTS.md` | Full results, caveats, and reproduction commands |
| `experiments/paper_tables/` | Machine-readable tables (`.md` / `.csv`) behind the paper |
| `experiments/runs/` | Per-run `config.json`, `metrics.csv`, `summary.json`, figures |
| `tests/` | `pytest` suite covering the metric and experiment code |
| `paper/` | ICLR 2027 submission source (`main.tex`) and the official style files |
| `IEEE_Conference_Template/` | **Superseded** — the WORLDS @ IROS 2026 workshop paper, kept for the record; its claims are the refuted ones |

### Experiments

| Script | Paper section |
|---|---|
| `generate_transitions.py`, `create_hard_subset.py`, `split_dataset.py` | Data pipeline (Sec. IV) |
| `build_clean_splits.py` | Episode-disjoint splits — **use these for new work** (see below) |
| `clean_pipeline.sh` | Retrain the sparse/dense baselines on the clean splits (`VARIANT=dense` for the packed variant) |
| `leakage_impact.py` | What the old split leak cost the published numbers |
| `dense_interaction_clean.py` | Dense-interaction control, rebuilt on clean splits |
| `aggregate_delta_head_study.py` | Merge delta-head runs into one multi-seed table |
| `counterfactual_augmentation.py` | CoDA-style augmentation from the learned gate (W3) |
| `scale_series_pipeline.sh` | Object-count series at one geometry, N=3..20 (W4) |
| `train_sparse_model.py`, `train_dense_baseline.py` | Training (Sec. IV) |
| `delta_head_study.py` | Distributional delta heads + multi-step rollout training (W1) |
| `compare_phase4_models.py`, `aggregate_seeds.py` | Prediction accuracy (Sec. V) |
| `param_matched_baseline.py` | "Not just smaller" control (Sec. VI) |
| `gate_ablation.py` | Object-centricity vs. change gate (Sec. VI); `--extended` adds the relational and soft-sparsity baselines |
| `oracle_gate_diagnostic.py` | Detection-vs-regression bottleneck (Sec. VI) |
| `rollout_horizon_error.py` | Multi-step rollout (Sec. VII) |
| `compositional_generalization.py` | Cross-count transfer (Sec. VII) |
| `sample_efficiency.py` | Sample efficiency (Sec. VII) |
| `planning_mpc.py` | Downstream CEM planning (Sec. VIII) |
| `make_paper_figures.py` | Paper figures |
| `domain_characterization.py` | Per-domain shortcut statistics; **derives** each domain's motion threshold |
| `cross_domain_pipeline.sh`, `cross_domain_analysis.py` | The shortcut across four physics engines, both benchmarks |
| `literature_baselines.py` | Five published models on change detection, native objectives, capacity-matched |
| `pixel_benchmark.py` | The whole study from 96×96 images instead of privileged state |
| `statistics.py` | Bootstrap CIs on differences + exact paired permutation tests |
| `onset_planar_pipeline.sh` | Planar onset benchmark at 8× episodes (fixes the underpowered replication) |
| `close_coverage_gaps.sh` | W3 at N=5/8, ratio sweep seeds 1–2, onset scaling regenerated at full episode count |
| `onset_shortcut_audit.py` | **The trivial-rule audit battery — the paper's main deliverable.** Eleven rules × any benchmark |
| `build_interaction_benchmark.py`, `interaction_pipeline.sh` | The third benchmark (indirect, chain-driven onset) and its audit |
| `momentum_shortcut.py` | The original finding: one line of code beats every learned model on the motion benchmark |
| `audit_battery.py` | Standalone entry point for auditing a **third-party** benchmark (see below) |

### The four-domain suite

Every domain exposes the same observation dict and `snapshot`/`restore`/`relocate_object`
API, so the whole pipeline runs against any of them via `--env`:

| `--env` | engine | contact regime | motion threshold |
|---|---|---|---|
| `tabletop` | MuJoCo | impulsive 3D | 0.020 |
| `planar` | ours | quasi-static | 0.010 |
| `billiards` | Box2D | near-elastic | 0.031 |
| `clutter` | Chipmunk2D | high-friction clutter | 0.029 |

Thresholds are **derived**, not hand-tuned: `domain_characterization.py` sets each so the
motion filter retains the same fraction of steps the tabletop's 0.020 retains. The procedure
reproduces 0.020 for the tabletop and 0.010 for planar — both values that were originally
found by hand, one of them only after a shared threshold silently destroyed the planar splits.

## Auditing *your* benchmark

`experiments/audit_battery.py` is the battery with the project dependencies stripped out, so
it runs on any change-detection benchmark. Give it one `.npz` per split holding
`target_mask` (R, N) and `object_xy` (R, N, 2), plus whichever of `object_speed`
(or `object_vel`), `actuator_xy` and `actuator_xy_next` you have — rules whose inputs are
missing are reported as **skipped**, never silently passed.

```bash
python -m experiments.audit_battery     --test mybench_test.npz --val mybench_val.npz --train mybench_train.npz     --model-predictions my_model_test_preds.npz     --report audit_report.md --json audit_report.json
```

Thresholded rules fit their radius on **validation** and apply it unchanged to test, so they
get the same courtesy a learned model gets. The report ends in a verdict — `DEGENERATE`,
`MARGINAL`, or `SURVIVES THESE RULES` — and the last of those is deliberately not called a
pass: these are the eleven rules that broke *our* benchmarks, and the step that would have
saved us two of them is inventing rules against the benchmark under test. `--extra-rules`
takes a Python file defining `rules(data, radius, rest_speed) -> dict` for exactly that.

`tests/test_audit_battery.py` pins the battery against `onset_shortcut_audit.py` on a real
split, so the deliverable and the published experiment cannot drift apart.

## Data splits: read this before running anything new

The pipeline behind the published tables runs `create_hard_subset` **then**
`split_dataset`. That order leaks. `create_hard_subset` sets `done=True` at the end of
every kept chunk, so the later `split_dataset` call treats each chunk as an episode and
fingerprints it by that chunk's *first row* — mid-trajectory poses, not the episode's
initial configuration. Chunks from one trajectory then get different fingerprints and are
assigned to splits independently. The `configuration_leakage: false` guard does not catch
it, because those fingerprints genuinely differ.

Measured at 3 objects / seed 0: **62 of 247 source episodes (25%) have chunks in both
train and test**, and 107 span more than one split. Train and test hold states from the
same rollout a few simulator steps apart.

```bash
python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2
```

reverses the order — split whole episodes first, filter within each split — and verifies
episode-disjointness on read-back. It writes `data/transitions/splits_clean_{N}obj_s{S}/`
containing `_full_` splits (long contiguous runs, needed for rollout training) and `_hard_`
splits (motion-filtered, for the headline metrics), with the latter a subset of the former.

The original `splits_{N}obj_s{S}` directories are left alone so published numbers stay
reproducible. **Numbers computed on clean splits are not comparable to them** — anything
being compared has to be retrained there (`bash experiments/clean_pipeline.sh N SEED`).

## Reproduce

Datasets are not tracked (~165 MB) but regeneration is seeded and deterministic. Build
the three object counts across three seeds:

```bash
bash experiments/run_count_seeds.sh 3
bash experiments/run_count_seeds.sh 5
bash experiments/run_count_seeds.sh 8 0.22 0.09   # 8 boxes need wider bounds / tighter spacing
python experiments/aggregate_seeds.py             # -> experiments/paper_tables/
```

Then any individual study, e.g.:

```bash
python experiments/gate_ablation.py --counts 3 5 8
python experiments/param_matched_baseline.py --counts 3 5 8
python experiments/oracle_gate_diagnostic.py --counts 3 5 8
```

`experiments/RESULTS.md` lists the full command for every table and figure, including the
held-out rollout sets and both rounds of the planning study.

## Tests

```bash
pytest tests/
```

## Scope

A deliberately controlled study: the model consumes structured per-object state (poses
and velocities), not pixels; scenes hold 3–8 rigid boxes; all networks are small MLPs
(< 0.1 M parameters) that train, plan, and evaluate on a single laptop; and the planning
result covers one push-to-goal task. The degeneracy finding is bounded the same way: it is
established for **single-pusher manipulation with local contact**, across four engines, and
the mechanism section says explicitly what a scene would need (multiple simultaneous
actuators, occluded or long-range coupling, delayed effects) for the argument not to apply.

## License

MIT — see [LICENSE](LICENSE).
