# Sparse/Residual World Models — Results (paper-ready)

Everything here is reproducible from `experiments/` scripts. All comparison numbers are
**mean ± std over 3 seeds (0/1/2)** unless stated. Machine-readable versions live in
`experiments/paper_tables/` (`main_results.{md,csv}`, `efficiency.md`, `sparsity_ablation.md`,
`seed_dispersion.json`).

## ⚠⚠ READ FIRST: the headline change-detection metric is dominated by a momentum shortcut

*Found 2026-08-08. `experiments/momentum_shortcut.py`, clean splits, 3 seeds.*

Every change-detection claim in this document compares the gated model against **ungated
regressors**, which are degenerate by construction (they flag every object). The baseline
that was never run is the trivial one:

> predict "this object will change" **iff it is already moving**.

| N | condition | F1 (headline) | **F1 on onset** | F1 on already-moving |
|---|---|---|---|---|
| 3 | **trivial: already moving** | **0.8969** | 0.0000 | 0.9734 |
| 3 | learned gate (`global`) | 0.8577 | 0.1764 | 0.9308 |
| 3 | learned gate (`contact`, velocity-free) | 0.7721 | **0.4127** | 0.9115 |
| 5 | **trivial** | **0.8704** | 0.0000 | 0.9518 |
| 5 | `global` / `contact` | 0.8211 / 0.6606 | 0.0569 / **0.3282** | 0.9353 / 0.8453 |
| 8 | **trivial** | **0.8318** | 0.0000 | 0.9229 |
| 8 | `global` / `contact` | 0.7966 / 0.5970 | 0.0242 / **0.2907** | 0.9062 / 0.8082 |

**One line of code beats every learned model on the headline metric, at every object count.**
P(changed | already moving) is 0.94–0.99 on the hard subset — it is filtered to steps with
real motion, and an object in motion stays in motion — so the metric mostly measures
*continuation*, not prediction. The published "sparse 0.87 vs dense 0.53" gap is against a
straw man.

**Splitting out the part that requires prediction inverts the ranking.** On *onset* — objects
at rest that start moving because something contacted them, where velocity carries no signal
— the velocity-using model collapses (0.02–0.18) while the velocity-free contact
featurisation reaches 0.29–0.41, **3–14× better**, despite scoring *worse* on the
momentum-dominated headline number.

**This unifies three failures previously treated as unrelated.** The model learned a momentum
shortcut, and the shortcut breaks wherever it is stressed:

| stress | symptom | section |
|---|---|---|
| distribution shift | planning collapsed with `global` features; `contact` fixed it | Downstream planning |
| observation noise | `global` loses 19% F1 at 2 cm pose error, `contact` 4.8% | Perception robustness |
| onset sub-task | `global` 0.09, `contact` 0.34 | this section |

**It replicates in the second environment, and is worse there.** Run on the 2D planar domain
(`momentum_shortcut_planar`, 3 seeds), which has different dynamics and a different contact
model:

| environment | trivial "already moving" (N=3/5/8) | learned gate |
|---|---|---|
| tabletop (impulsive) | 0.897 / 0.870 / 0.832 | 0.858 / 0.821 / 0.797 |
| **planar (quasi-static)** | **0.990 / 0.968 / 0.944** | 0.736 / 0.620 / 0.436 |

The prediction going in was that the *quasi-static* planar domain would offer **less**
momentum signal, since objects there stop almost immediately. That was wrong, and instructive:
its contact impulse sets object velocity directly from the pusher's drive, so velocity is an
almost perfect "currently being pushed" indicator and the trivial rule reaches 0.99. The
learned model trails it by 25–51 points. Two environments, two contact models, same
conclusion — this is not a quirk of the MuJoCo tabletop.

**Consequences for the paper.** The trivial velocity rule must be a first-class baseline in
every change-detection table — it is now a rung in `gate_ablation`, and `onset_f1` is reported
alongside `f1` everywhere. Onset F1 should be the primary metric, because it is the only one
that measures prediction rather than continuation. And the claim worth making is no longer
"modelling change beats modelling everything" — it is **"the useful problem is predicting
motion *onset* from contact geometry, and velocity-based featurisation actively harms it by
supplying a shortcut."** That claim is supported here, replicates across environments,
explains the planning and perception results, and does not rest on a metric a one-liner wins.

### The corrected benchmark, and what wins on it

The diagnosis above is only a criticism until the benchmark is fixed. `--filter-mode onset`
(`experiments/create_hard_subset.compute_onset_keep_mask`) keeps only transitions in which a
**stationary** object starts moving — which can only happen through contact — removing the
shortcut from the *training signal* as well as the metric. Onset events are 3.0% of steps
against the motion filter's 21.7%, so `onset_benchmark_pipeline.sh` generates 6× the episodes
to land the onset training split near the motion benchmark's ~950 rows; otherwise the two
benchmarks would differ in sample size as well as task.

Results, 3 counts × 3 seeds, models trained *on* the onset benchmark:

| N | trivial "already moving" | gate (`global`) | **gate (`contact`, velocity-free)** |
|---|---|---|---|
| 3 | 0.109 | 0.653 | **0.752** |
| 5 | 0.174 | 0.493 | **0.677** |
| 8 | 0.280 | 0.394 | **0.654** |

*(onset F1 specifically: trivial 0.000 by construction, `global` 0.639 / 0.459 / 0.315,
`contact` **0.767 / 0.699 / 0.681**.)*

**1. The shortcut is gone.** The trivial rule collapses from 0.897 / 0.870 / 0.832 on the
motion benchmark to 0.109 / 0.174 / 0.280 here. The corrected task cannot be solved by reading
velocity, which is precisely what it was designed to guarantee.

**2. Learned models now beat it decisively**, by 0.54 / 0.32 / 0.11 for `global` and
0.64 / 0.50 / 0.37 for `contact`. On the motion benchmark every learned model *lost* to one
line of code; here the ordering is the one the field assumes and never verified.

**3. Velocity-free contact featurisation wins at every count, and the margin widens with
scene complexity** (+0.099 / +0.184 / +0.260 over `global`). On the motion benchmark `contact`
looked *worse* — 0.772 / 0.661 / 0.597 against `global`'s 0.858 / 0.821 / 0.797 — because it
was being scored on a metric that rewards the shortcut it refuses to take. Under the corrected
task the ranking inverts, which is the strongest available evidence that the original
benchmark was selecting for the wrong thing.

**4. Training on onset data more than doubles onset performance.** `contact` reaches 0.68–0.77
onset F1 here against 0.29–0.41 when trained on motion-filtered data. The shortcut was not just
inflating the metric; it was starving the training signal.

The reciprocal check is worth stating: `contact` is *worse* on already-moving objects
(0.30–0.38 against `global`'s 0.89–0.91). It cannot do continuation, because it cannot see
velocity. The two featurisations solve different halves of the problem, and the motion
benchmark scored only the half that needs no prediction.

### The full ladder on the corrected benchmark — the gate earns its place here

`gate_ablation --extended` re-run on the onset splits, 3 seeds
(`gate_ablation_onsetbench_s{0,1,2}`):

| N | trivial | dense | oc_residual | gnn | set_transf | soft gate | **sparse** |
|---|---|---|---|---|---|---|---|
| **F1** ||||||||
| 3 | 0.109 | 0.526 | 0.526 | 0.526 | 0.526 | 0.647 | **0.653** |
| 5 | 0.174 | 0.369 | 0.369 | 0.368 | 0.369 | 0.490 | **0.493** |
| 8 | 0.280 | 0.271 | 0.271 | 0.271 | 0.271 | **0.401** | 0.394 |
| **unchanged-object L2** ||||||||
| 3 | — | 0.266 | 0.110 | 0.084 | 0.125 | 0.0274 | **0.0139** |
| 5 | — | 0.236 | 0.086 | 0.067 | 0.091 | 0.0331 | **0.0156** |
| 8 | — | 0.274 | 0.089 | 0.057 | 0.086 | 0.0401 | **0.0196** |

**1. The gate is vindicated by the corrected benchmark, having been embarrassed by the old
one.** On the motion benchmark the trivial velocity rule *beat* every learned model. Here the
gated models beat it by 0.54 / 0.32 / 0.11 and beat the ungated models by ~0.12 at every count.
The gate was always doing something; the old metric could not see it because the shortcut was
available to everything.

**2. The ungated degeneracy persists exactly as before.** `dense`, `oc_residual`, `gnn` and
`set_transformer` again produce F1 identical to three decimals at every count (0.526 / 0.369 /
0.271), reproducing the W2 finding on a different task. Full self-attention still does not
help a model notice which objects will move.

**3. The discreteness result survives, and shrinks — which strengthens it.** Soft and hard
gating remain tied on detection (0.647 vs 0.653, 0.490 vs 0.493, 0.401 vs 0.394 — soft
marginally ahead at N=8), while the hard gate is ~2× better on unchanged-object L2 (0.0139 vs
0.0274, 0.0156 vs 0.0331, 0.0196 vs 0.0401). On the motion benchmark that ratio was 8–47×,
which was large enough that the *definitional* objection — a hard gate firing 0 emits exactly
zero, a soft gate at p=0.05 emits 0.05·δ — could not be dismissed. At 2×, on a task whose
unchanged-object population is defined by contact rather than by momentum, it reads as a real
effect of the design rather than a restatement of it.

**4. Relational structure still helps pose accuracy without helping detection.** `gnn` is best
among the ungated rungs on unchanged-object L2 (0.084 / 0.067 / 0.057 against `oc_residual`'s
0.110 / 0.086 / 0.089) while scoring identically on F1 — the same dissociation as before,
reproduced on the corrected task.

*Prediction check:* before running this, the expectation recorded was that the discreteness
result would weaken and the featurisation result would strengthen. Both happened, and one
thing that was not predicted also happened: the *gate itself* looks considerably better here
than on the benchmark it was originally validated on.

### Scaling on the onset benchmark — the mechanism replicates, the F1 numbers do not

The W4 series (N=3→20, 5 seeds, one geometry) onset-filtered directly from the existing `uni`
data rather than regenerated at 6× episodes:

| N | sparse unchanged L2 | dense unchanged L2 | **gap** | test rows / count |
|---|---|---|---|---|
| 3 | 0.0246 ± 0.0107 | 0.350 | 0.326 | 75 |
| 5 | 0.0197 ± 0.0065 | 0.403 | 0.384 | 85 |
| 8 | 0.0245 ± 0.0136 | 0.373 | 0.349 | 87 |
| 12 | 0.0313 ± 0.0104 | 0.507 | 0.476 | 128 |
| 20 | 0.0365 ± 0.0095 | 0.726 | **0.689** | 143 |

**The error-suppression mechanism replicates on the corrected task, and more strongly.** The
dense–sparse gap grows 2.1× from N=3 to N=20 (0.326 → 0.689) against 1.6× on the motion
benchmark, while the sparse model stays two orders of magnitude below the monolith throughout.
Unlike the motion-benchmark version the growth is **not strictly monotone** — it dips at N=8 —
which is what 85 test rows per count predicts.

**The F1 numbers from this series must not be used.** Sparse loses to dense at N=3 (0.438 vs
0.520) and N=8 (0.239 vs 0.258), contradicting the dedicated onset benchmark where sparse wins
at every count by ~0.12. The cause is sample size: onset-filtering 250 episodes leaves
~110–237 training rows per cell, against the dedicated benchmark's 752. This series is
adequate for the *scaling trend* of a large effect and inadequate for a ranking of close ones.
For F1 on the corrected task, use the ladder above.

*If this scaling claim is to carry weight in the paper it should be regenerated at the
dedicated benchmark's episode count across all five object counts — several hours of
generation, most of it at N=20 — rather than reported from this series with a caveat.*

### The delta-head result (W1) does NOT survive the corrected benchmark

Delta heads evaluated on the onset test split (trained on the full unfiltered data, so this is
not sample-size limited), 3 seeds — margin over no-op on changed-object L2:

| N | mse | gaussian | mdn | *(motion benchmark, mdn)* |
|---|---|---|---|---|
| 3 | −0.0017 ± 0.0041 | +0.0012 ± 0.0004 | +0.0035 ± 0.0019 | *+0.046* |
| 5 | −0.0046 ± 0.0039 | −0.0002 ± 0.0087 | −0.0005 ± 0.0050 | *+0.063* |
| 8 | +0.0043 ± 0.0026 | +0.0130 ± 0.0089 | +0.0172 ± 0.0125 | *+0.054* |

**Every head sits at the no-op floor on onset transitions.** The margins span −0.005 to +0.017
where the same MDN reached +0.046 to +0.063 on the motion benchmark — a ~5× collapse. The
*ordering* is preserved at N=3 and N=8 (mdn > gaussian > mse), so the mechanism conclusion
from the K sweep is not contradicted; the **effect** is what disappears.

The interpretation is uncomfortable but clear. W1 was created to fix exactly this — the
oracle-gate diagnostic showed the delta head matching no-op on changed objects — and the
mixture head did fix it *on the momentum-dominated benchmark*. On the corrected task the
problem returns in full. Predicting how far an object moves when contact *initiates* motion is
a materially harder regression than predicting how far an already-sliding object continues,
and the distributional head does not solve it.

So the corrected benchmark validates two of the three W1-W4 stories and refutes the third:
detection (the gate beats the trivial rule and the ungated models), featurisation (velocity-free
`contact` wins with a widening margin), and **delta regression (no head beats no-op)**. The
last is now a limitation to state plainly rather than a contribution — and it identifies the
open problem the paper should name as future work: contact-onset magnitude prediction.

### Cross-environment check on the corrected benchmark — direction replicates, power does not

The onset benchmark rebuilt on the 2D planar domain, which had the **stronger** shortcut
(trivial F1 0.99 there against 0.90 on the tabletop) and is therefore the sharper test:

| N | trivial | gate (`global`) | **gate (`contact`)** |
|---|---|---|---|
| 3 | 0.032 | 0.335 | **0.601** |
| 5 | 0.183 | 0.239 | **0.455** |
| 8 | 0.237 | 0.224 | **0.345** |

Both headline findings replicate in direction. The trivial rule **collapses from 0.990 to
0.032–0.237**, confirming the correction removes the shortcut in a domain where it was nearly
perfect. And velocity-free `contact` beats `global` at every count, by +0.266 / +0.216 / +0.121.

> **✅ SUPERSEDED BY AN ADEQUATELY POWERED RERUN (2026-08-15).** The numbers above came from
> 1500 episodes, which yielded **102 train / 9 val / 10 test rows** at N=3 — a single
> misclassification moved F1 by ~0.1 — and were correctly flagged here as indicative only.
> `experiments/onset_planar_pipeline.sh` regenerated the benchmark at **12,000 episodes and 5
> seeds**, giving **821 / 98 / 87** rows at N=3, comparable to the tabletop's 752 / 90 / 80.
> Run: `experiments/runs/momentum_shortcut_planar12k/`.

**Properly powered planar replication** (12k episodes, 5 seeds):

| N | trivial `already_moving` | gate (`global`) | **gate (`contact`)** | contact − global |
|---|---|---|---|---|
| 3 | 0.085 | 0.600 | **0.779** | +0.179 |
| 5 | 0.143 | 0.590 | **0.769** | +0.179 |
| 8 | 0.243 | 0.479 | **0.769** | +0.290 |

Both findings survive the power increase. The velocity rule collapses (0.085–0.243), so the
onset filter removes the momentum shortcut in this domain too; and velocity-free `contact`
beats `global` at every count with the margin **widening in object count**, matching the
tabletop. The effect sizes are larger and better separated than the underpowered version
suggested, so this is now a genuine cross-environment replication rather than a directional
hint.

**But the full battery, run on the same splits, says this benchmark is degenerate too**
(`experiments/runs/audit_planar12k/`, 3 counts × 5 seeds):

| rule | F1 | onset F1 |
|---|---|---|
| `moving_or_near` | **0.9833** | 0.9833 |
| `near_pusher_or_mover` | 0.9684 | **0.9909** |
| `pusher_near` | 0.9631 | 0.9825 |
| `logistic_on_distance` (one feature) | 0.9611 | 0.9803 |
| `nearest_to_pusher` (zero parameters) | 0.9511 | 0.9789 |
| learned gate (`contact`) | 0.7723 | 0.7644 |
| `already_moving` | 0.1570 | 0.0000 |

So the proximity degeneracy found on the tabletop onset benchmark **replicates on a second,
independent environment** with a different contact model — and is stronger there (0.983 vs
0.961). That is the important reading of this section now: the cross-environment result
confirms the *degeneracy*, not a fix. The `contact`-over-`global` finding above remains a real
comparison between two featurisations, but both are being compared on a task a one-liner wins.

### And on onset, the gate itself is harmful at N ≥ 5

Onset F1 from the extended ladder (`gate_ablation_onset_s{0,1,2}`, 3 seeds, clean splits):

| N | trivial | ungated (dense / gnn / set-transformer) | soft gate | **sparse (hard gate)** | contact-featurised |
|---|---|---|---|---|---|
| 3 | 0.000 | 0.152 ± 0.033 | 0.216 ± 0.086 | 0.176 ± 0.080 | **0.413** |
| 5 | 0.000 | 0.103 ± 0.009 | 0.071 ± 0.019 | **0.057 ± 0.023** | **0.328** |
| 8 | 0.000 | 0.083 ± 0.015 | 0.036 ± 0.051 | **0.024 ± 0.034** | **0.291** |

The ungated column is not a genuine competitor — those models flag every object, so their
onset F1 is exactly the always-flag value `2p/(1+p)` for an onset rate `p` of 5–6%. But the
gated models falling **below** it at N=5 and N=8 is a real result: the gate suppresses
precisely the predictions that onset detection requires. On the sub-task that actually needs
prediction, the paper's central mechanism is a liability at every count above 3.

**The one thing that materially helps onset is the velocity-free `contact` featurisation**
(0.29–0.41, 3–14× the velocity-based gate), and it helps *because* it cannot take the
momentum shortcut. That relocates the contribution: the useful ingredient is **how the model
is given contact geometry**, not the gate. The gate's real and defensible benefit is what W2
isolates — it keeps error out of the static majority (unchanged-object L2 8–47× better than a
soft gate) — which is a claim about *error suppression*, not about detecting change.

## ⚠⚠⚠ READ SECOND: the CORRECTED benchmark has a shortcut too, and it is worse

*Found 2026-08-15. `experiments/onset_shortcut_audit.py`, 3 counts × 3 seeds, thresholds
fitted on validation and applied unchanged to test. Run: `experiments/runs/onset_shortcut_audit/`.*

The onset benchmark was built because the motion benchmark was won by a one-line velocity
rule. It was never audited the same way. It should have been, and this is what that audit
finds:

| rule | params | motion F1 | **onset F1** |
|---|---|---|---|
| `already_moving` (the original shortcut) | 0 | **0.8664** | 0.1875 |
| `pusher_near` (contact radius, fitted on val) | 1 | 0.8046 | **0.9307** |
| `nearest_to_pusher` (the single closest object) | **0** | 0.8230 | **0.9252** |
| `moving_or_near` (both, combined by hand) | 1 | **0.9274** | **0.9605** |
| `always_change` (ungated degeneracy) | 0 | 0.4194 | 0.3887 |
| learned gate (`global`, velocity) | ~7k | 0.8251 | 0.5133 |
| learned gate (`contact`, velocity-free) | ~7k | 0.6766 | 0.6941 |

**1. The correction did not remove the shortcut; it swapped one for a stronger one.** The
velocity rule collapses on the onset benchmark exactly as designed (0.866 → 0.188 — the
filter *works*), but a proximity rule takes its place and scores **0.931**, higher than the
velocity rule ever managed on the motion benchmark. Onset events are contact-driven by
construction, so "which object is the pusher about to touch" is very nearly the whole label.

**2. The strongest trivial rule has zero parameters.** `nearest_to_pusher` — predict change
for the single object closest to the pusher, nothing else, nothing fitted — reaches 0.925 F1
and **0.980 onset F1**. There is no threshold to accuse of being tuned.

**3. Two controls rule out the obvious defences.**

*"You gave the model the wrong features."* It can see contact: the velocity-free `contact`
featurisation exposes *signed contact distance* directly, and that gate still reaches only
**0.694** against the one-liner's 0.925. A learned model with the relevant quantity as an
input feature loses to a parameter-free rule over that same quantity by 0.23 F1.

*"You undertrained the gate."* A **logistic regression on one feature** — the distance from
the object to the post-action pusher position, one weight and one bias, convex objective,
nothing to tune, fitted on the same training rows — reaches **F1 0.916 and onset F1 0.957**.
It nearly matches the hand-tuned proximity rule (0.931) and beats the full learned gate by
0.22 F1. There is no version of "more capacity or more epochs would fix it" that survives a
one-parameter convex model doing the job. The benchmark's label is recoverable from a single
scalar, and that is a fact about the task rather than about any model.

**4. Learned models lose to one-liners on *both* benchmarks.** `moving_or_near` beats the
best learned gate on the motion benchmark (0.927 vs 0.825) and on the onset benchmark (0.961
vs 0.694). The earlier reading — "the old benchmark was broken, the new one restores the
expected ordering" — was itself an artifact of comparing against too small a battery of
trivial baselines. One trivial baseline was run; the right number is several.

### The third benchmark fails too — the trivial ceiling does not move

*Final run: `experiments/runs/audit_interaction_billiards/`, billiards, N=5 and N=8, 3 seeds,
with gates trained on the interaction benchmark itself. N=3 is excluded: at ~250 train / ~35
test rows per cell it is too thin to train on or measure with. Build:
`experiments/build_interaction_benchmark.py` (9 cells, all verified episode-disjoint).*

The interaction benchmark keeps only transitions where a stationary object starts moving
**while some other object is closer to the pusher** -- change that reached the object
indirectly. Construction verified on the built splits: every kept row contains an indirect
onset, the indirect mover sits at median 0.151 m from the pusher against the nearest object's
0.061 m, and the nearest object also moves in 94.4% of rows, so these really are contact
chains.

**It defeats the rule it was built against, and loses to the next one.**

| rule | motion bench | onset bench | **interaction bench** |
|---|---|---|---|
| `already_moving` | **0.8664** | 0.1875 | 0.7830 |
| `nearest_to_pusher` (0 params) | 0.8230 | **0.9252** | **0.4368** |
| `pusher_near` | 0.8046 | 0.9307 | 0.7153 |
| `logistic_on_distance` (1 feature) | -- | 0.9159 | 0.7306 |
| `always_change` | 0.4194 | 0.3887 | 0.6731 |
| `near_a_mover` | 0.0961 | 0.2135 | 0.7124 |
| **`moving_or_near_mover`** | 0.9152 | 0.3149 | **0.9550** |
| **best trivial rule** | **0.9274** | **0.9605** | **0.9550** |
| best learned gate | 0.8251 | 0.6941 | 0.7724 |

`nearest_to_pusher` collapses from 0.925 to 0.437 and the one-feature logistic from 0.916 to
0.731, so the filter did exactly what it was designed to do. But `moving_or_near_mover` --
predict change for anything already moving *or* adjacent to something already moving --
reaches **0.955**, against the best learned gate's 0.772.

This was **predicted before the benchmark was built**. `near_a_mover` and its combinations
were added to the battery specifically because, in a contact chain, the object that starts
moving is by definition adjacent to a mover. Building the benchmark and testing only the rules
it was designed to defeat would have produced a false positive.

*Correction to a preliminary reading.* An earlier partial run (N=3 and N=5, seeds 0-1, no
trained gates) put the best trivial rule at 0.976 and was written up here as the trivial
ceiling **rising** monotonically across the three benchmarks (0.927 -> 0.961 -> 0.976). The
full run does not support that: it gives 0.955, slightly *below* the onset benchmark's 0.961.
The defensible claim is the weaker and still-decisive one below.

### The pattern is the result

Three benchmarks, three filters, each removing the previous shortcut -- and the best trivial
rule stays at **0.93-0.96 every time**, never once dropping to where a learned model could
lead. What changes is only *which* one-liner wins:

| benchmark | what it removes | what wins instead | best trivial |
|---|---|---|---|
| motion | nothing | velocity (`already_moving`) | 0.927 |
| onset | the velocity shortcut | geometry (`nearest_to_pusher`, 0 params) | 0.961 |
| interaction | the geometry shortcut | **propagated** kinematics (`moving_or_near_mover`) | 0.955 |

A second failure mode is visible in the last column: the interaction filter selects
multi-change events, so the positive rate rises and the degenerate `always_change` baseline
jumps to **0.673** -- within 0.04 of the *learned* gate's 0.709. Filtering for harder events
made the class balance easier, which is its own kind of degeneracy and the reason
`always_change` belongs in the battery permanently.

**The honest conclusion.** The degeneracy is not a property of any one filter and it is not
fixable *by* filtering. It is a property of **single-pusher manipulation scenes with local
contact**: the label is always recoverable from local kinematics or local geometry, and a
filter only chooses which of the two. A benchmark in this class that resists the battery would
need a task where change is not locally determined -- multiple simultaneous actuators,
occluded or long-range coupling, delayed effects -- which is a change of *task*, not of filter.

**What the paper should claim.** Not "here is a corrected benchmark". Rather: three successive
corrections, each defeating its predecessor's shortcut and each losing to a different
one-liner at the same ~0.95 level, plus a reusable audit battery and the methodological
requirement that it be run *with rules invented against the benchmark under test*. That is
harder to dismiss than a fourth filter would have been.

### What this costs, and what survives

**Superseded.** Every statement of the form "the corrected onset benchmark restores the
ordering the field assumes" must go. That includes the claim in the earlier ICLR-track
section that "no published model is beaten by the one-liner here" — true of `already_moving`,
false of `pusher_near` and `nearest_to_pusher`, which beat all five published models on the
onset benchmark by an even wider margin than the velocity rule beats them on the motion one.
The onset filter is a *valid removal of the velocity shortcut* and nothing more.

**Strengthened.** The paper's actual contribution gets sharper and harder to dismiss. It is
not "here is a corrected benchmark". It is:

> Change-detection benchmarks in object-centric world modelling are **pervasively degenerate**.
> The motion-filtered benchmark is won by velocity; the onset-filtered benchmark is won by
> proximity; five published models plus our own lose to both. The field does not run trivial
> baselines, and every "our model detects what changes" claim in this literature is unvalidated
> until it does.

That is a benchmark-methodology result with a concrete deliverable — the battery of six
trivial rules in `onset_shortcut_audit.py` — rather than a claim about one architecture.

### Why it happens, and what a non-degenerate benchmark would need

The mechanism is structural, not a bug in the filter. With **one point-like pusher** and
objects placed at least 0.09 m apart, contact is spatially unambiguous: at any step the
pusher is near at most one object, and that object is the one that moves. `nearest_to_pusher`
is then close to an oracle. Both shortcuts are symptoms of the same thing — the label is
recoverable from a single scalar (speed, or distance) because the scene never presents a
genuine attribution problem.

Removing it requires scenes where *which* object moves is not determined by any one scalar:

* **dense packing, so several objects are equidistant from the pusher.** Partially tested:
  on the packed dense-interaction variant (`experiments/runs/audit_dense_packed/`)
  `nearest_to_pusher` falls 0.823 → 0.768 and `pusher_near` 0.805 → 0.757, i.e. packing does
  erode the proximity shortcut — but not enough, since `moving_or_near` still leads the
  learned gate 0.909 to 0.822.
* **contact chains, so change propagates to objects the pusher never touches.** Untested at
  the time of writing; this is the condition proximity cannot represent even in principle.
* **both at once, with the onset filter applied.** No dataset in the project currently
  combines dense packing *and* onset filtering; that combination is the first candidate for a
  benchmark that resists both shortcuts, and the `clutter` domain (Chipmunk2D,
  `min_object_separation` 0.07) was built for exactly it.

**The honest status is that the audit is a finished negative result and the fix is a
hypothesis.** The battery must be run on every new benchmark before it is claimed to be
non-degenerate, this one included — which is the whole lesson of this section.

#### 2026-08-18: dense packing + onset was built, and the first attempt is UNDERPOWERED

*`experiments/runs/audit_onset_clutter/`. **Do not quote these numbers.***

The combination named above as "the first candidate for a benchmark that resists both
shortcuts" now exists: the clutter domain (Chipmunk2D) generated at its designed
`min_object_separation` of **0.07** rather than the 0.09 the cross-domain pipeline overrode
it with, then onset-filtered. The audit ran at 3 counts x 3 seeds. It returned the best
trivial rule at **0.851** (`moving_or_near`) against a learned gate at **0.324 / 0.355** --
which would be the largest trivial-over-learned margin in the project.

**It is not reportable, because the splits are too small to train on or measure with.** Onset
events are ~0.065% of clutter steps against the tabletop's ~3%, a 46x difference: in dense
high-friction clutter objects are mostly already touching and jostling, so a clean
stationary-to-moving transition is rare. At 1500 episodes per cell that yields

| N | train rows (s0/s1/s2) | test rows |
|---|---|---|
| 3 | 38 / 47 / 43 | 6 / 6 / 10 |
| 5 | 80 / 83 / 102 | 17 / 11 / 19 |
| 8 | 133 / 174 / 133 | 24 / 20 / 9 |

against the other benchmarks' ~750-950 training rows. This project already excludes N=3 from
the interaction benchmark for being thinner than *250 train / 35 test*; every cell here is
thinner than that. A learned gate at 0.32 trained on 38 rows is at least as likely to be
starvation as a finding, and publishing it would be the same error this document exists to
document, pointed the other way.

**One observation does survive the sample size, and it is worth stating as a hypothesis to
test:** the *trivial ceiling itself* fell for the first time -- 0.927 / 0.961 / 0.955 on the
three tabletop benchmarks against **0.851** here. Trivial rules need no training rows, so
that number is not affected by the split size the way the learned gate's is (its ~110 test
rows still make it noisy). If it holds at full sample size, dense packing is the first
manipulation this project has found that *erodes* the shortcut rather than swapping it.

`experiments/clutter_onset_rebuild.sh` regenerates N=5 and N=8 at **12000 episodes per cell**
to land near the other benchmarks' row count. N=3 is excluded and stays excluded: it would
need ~27000 episodes per cell, and the exclusion is reported rather than quietly applied.

## ICLR-track additions (2026-08-15) — external validity of the shortcut

*Everything in this block is new work aimed at the two objections the momentum-shortcut
finding could not answer as of 2026-08-08: that the shortcut is a property of **our
simulators**, and that our baselines are **stand-ins we wrote ourselves**. Both are now
tested directly. Run artifacts: `experiments/runs/domain_characterization/`,
`.../literature_baselines_{motion,onset}/`.*

### The domain suite — four contact regimes on four independent engines

The shortcut was previously measured in two environments, both ours. Two third-party
engines are now added, chosen to bracket the manipulation regime from both ends:

| domain | engine | contact regime | post-contact motion |
|---|---|---|---|
| `tabletop` | MuJoCo | impulsive 3D | short slide |
| `planar` | ours | quasi-static | stops immediately |
| **`billiards`** | **Box2D** | near-elastic (restitution 0.85) | very long |
| **`clutter`** | **Chipmunk2D** | high-friction, dense packing | short, chained |

Box2D and Chipmunk share no code with us or with each other, so agreement across the suite
cannot be a shared-implementation artefact. `models/envs/box2d_billiards.py`,
`models/envs/pymunk_clutter.py`; both expose the existing observation dict and
`snapshot`/`restore`/`relocate_object` API, so the whole pipeline runs against them unchanged.

**Measured directly on the simulators, no model trained** (3 counts × 3 seeds × 60 episodes,
`experiments/domain_characterization.py`):

| domain | engine | P(chg \| moving) | P(chg \| rest) | ratio | changed frac | motion threshold |
|---|---|---|---|---|---|---|
| billiards | Box2D | **0.9817** | 0.0198 | 50× | 0.394 | 0.031 |
| planar | ours | 0.9500 | 0.0113 | 84× | 0.152 | 0.010 |
| clutter | Chipmunk2D | 0.8778 | 0.0247 | 36× | 0.358 | 0.029 |
| tabletop | MuJoCo | **0.7144** | 0.0149 | 48× | 0.124 | 0.020 |

**The shortcut's existence condition holds in every domain.** P(changed | already moving)
exceeds P(changed | at rest) by 36–84× on four engines and four contact models. The shortcut
is a property of *physical pushing tasks*, not of any one simulator.

**Pre-registered predictions, scored.** (a) *billiards has the strongest shortcut* —
**confirmed**, 0.9817, the highest of the four, as the near-elastic regime predicts.
(c) *the shortcut exists in every domain* — **confirmed**. (b) *clutter has the weakest* —
**refuted**: the weakest is the MuJoCo tabletop at 0.7144, with clutter second at 0.8778. The
reasoning behind (b) was that high friction stops objects fastest, and that is true, but it
missed that the tabletop's 3D solver lets a box come to rest through tilt and settling that a
2D engine cannot represent, so more of its "moving" objects stop within one step. Recorded as
refuted rather than quietly rewritten.

**Per-domain motion thresholds are derived, not guessed.** Each is set so its filter retains
the same fraction of steps the tabletop's hand-tuned 0.020 retains under identical settings.
Two self-checks say the procedure is sound: it returns **0.020 for the tabletop**,
reproducing the value it was anchored to, and **0.010 for planar**, independently recovering
the value that was found by hand only after a shared threshold destroyed the planar splits.
The `retention @ 0.02` column shows why that mattered — applying the tabletop threshold
unchanged to planar retains **0.6%** of steps.

*An earlier version of this calibration anchored on the ratio between threshold and median
moved-object displacement, using the tabletop's documented 0.0254. That was circular: 0.0254
was measured on the already-filtered hard split, so it is conditioned on passing the very
threshold being derived. The tell was that the tabletop's own derived threshold came back as
0.010 rather than its actual 0.020.*

### Five published models on change detection — the degeneracy is not ours

`models/literature_baselines.py`, `experiments/literature_baselines.py`. Each implemented
from its paper rather than adapted from our ladder, trained with its **native objective**, on
identical splits and features, capacity-matched to the sparse model's total parameter count
(~8.5–8.8k):

* **GNS / DPI-Net** (Sanchez-Gonzalez 2020; Li 2019) — encode-process-decode, **3
  message-passing steps**, explicit geometric edge features, residual processor updates. The
  existing `gnn` rung is a single step with no edge features, so it was never a fair test of
  "a real GNN would fix this".
* **C-SWM** (Kipf, van der Pol & Welling, ICLR 2020) — contrastive energy objective, no
  decoder. Pose columns come from a probe trained on **frozen** latents and are labelled as
  such; its native ranking metrics are reported separately.
* **SlotFormer** (Wu et al., ICLR 2023) — temporal transformer over object tokens with a
  history window, so it is the one baseline that could recover velocity by differencing.
* **PETS** (Chua et al., NeurIPS 2018) — ensemble of heteroscedastic Gaussians, NLL on
  bootstrapped batches.
* **NPS** (Goyal et al., NeurIPS 2021) — sparse learned rules, one applied per object per
  step. The closest published relative of our claim, and therefore the sharpest test of it.

**Motion benchmark** (3 counts × 3 seeds, clean splits):

| model | F1 | onset F1 | recall | unchanged L2 |
|---|---|---|---|---|
| **trivial: already moving** | **0.8664** | 0.0000 | 0.8319 | — |
| gns | 0.4193 | 0.1124 | 0.9995 | 0.0673 |
| cswm (probe) | 0.4193 | 0.1121 | 0.9986 | 0.0616 |
| slotformer | 0.4194 | 0.1126 | 1.0000 | 0.0889 |
| nps | 0.4195 | 0.1126 | **1.0000** | 0.0757 |
| pets | 0.4196 | 0.1127 | 1.0000 | 0.0656 |

**1. One line of code beats five published object-centric world models**, at every object
count: 0.897 / 0.870 / 0.832 against ~0.539 / 0.405 / 0.313. Paired bootstrap on the
*difference* excludes zero at every count (e.g. N=8: −0.519, CI [−0.565, −0.494]).

**2. Every published model is degenerate**, recall 0.9986–1.0000. They differ only on
unchanged-object L2 (0.062–0.089), i.e. on pose accuracy, never on knowing *which* objects
move. This is the strongest available form of the W2 finding: it is a property of ungated
regression as a class, not of the rungs we happened to build.

**3. NPS does not substitute for a change gate** — recall exactly 1.000, indistinguishable
from the rest. Sparsity over *which rule transforms an object* is a different axis from
sparsity over *whether an object is transformed*, and having the first does not give the
second. This was pre-registered as prediction (b) and is the single most load-bearing check
on the paper's attribution.

**Onset benchmark — the ranking inverts, exactly as the correction predicts:**

| model | F1 | onset F1 | recall |
|---|---|---|---|
| **trivial: already moving** | **0.1875** | 0.0000 | 0.1112 |
| all five published models | 0.3887 | 0.3662–0.3663 | 1.0000 |

The trivial rule collapses 0.866 → 0.188 while the learned models roughly triple their onset
F1 (0.112 → 0.366). So the onset filter does remove the *velocity* shortcut, which is what it
was built to do.

> **⚠ SUPERSEDED — read the audit section above.** This paragraph originally continued "no
> published model is beaten by the one-liner here; the corrected benchmark restores the
> ordering the field assumes." That is true only of `already_moving`. The shortcut audit
> (`experiments/onset_shortcut_audit.py`) found that a **proximity** rule scores 0.931 on the
> onset benchmark and a parameter-free nearest-object rule 0.925 — so all five published
> models are beaten here too, by a *wider* margin than the velocity rule beats them on the
> motion benchmark. The onset benchmark removes one shortcut and exposes another. Quote it as
> a velocity-shortcut removal, never as a corrected benchmark.

**Two findings that were not predicted.**

*PETS ensemble disagreement also rides the shortcut.* Per-object epistemic variance was
included as the gate-free way to read change out of a standard probabilistic model, and it
works on the motion benchmark (AUC **0.766**, well above chance). On the onset benchmark it
falls to **0.527** — essentially chance. So ensemble disagreement is a change detector only
where change means continuation; it does not detect contact-driven onset. Prediction (c) said
the AUC would be above chance but below a gate's, which held on the motion benchmark; that it
would *vanish* under the correction was not anticipated.

*C-SWM's native metric moves the other way.* Hits@1 goes **0.305 → 0.884** and MRR
0.527 → 0.919 from the motion to the onset benchmark. Its contrastive latents rank successors
far better on onset data even while its change-detection probe stays degenerate — a clean
illustration that a model can be good on its own objective and still carry no information
about which objects will move.

### Statistical treatment — intervals on differences, not overlapping error bars

`experiments/statistics.py`. Every table above quotes `mean ± std` over 3–5 seeds and
conclusions were previously drawn by whether bars appeared to overlap. Two fixes:

* **Bootstrap CIs on the difference**, not on each condition separately. Two overlapping
  marginal intervals are entirely compatible with a difference interval that excludes zero,
  and the difference is what every claim is about.
* **Paired sign-flip permutation tests**, exact by enumeration at these sample sizes. Seeds
  are paired (same split, same episode set), and unpaired comparison throws that away.

**The n=3 floor, stated plainly.** With 3 seeds a two-sided sign-flip test has 2³ = 8
outcomes, so the smallest attainable p-value is **0.25** — every comparison above reports
`p = 0.250` *at the floor*, which is not a null result and must never be read as one. This is
reported explicitly as `min_attainable_p`. It is also the concrete argument for the 5-seed
runs now standard in the new pipelines: the floor drops to 0.0625 at n=5 and 0.031 at n=6.
Where a claim needs significance rather than an effect size, quote the difference CI.

## Pixels — built, and BLOCKED on perception, with the cause measured

*2026-08-15. `models/envs/renderer.py`, `models/slot_attention.py`,
`models/keypoint_encoder.py`, `experiments/pixel_benchmark.py`. Status: infrastructure
complete and tested; no pixel result yet.*

The sharpest objection to the momentum-shortcut finding is that it is an artefact of feeding
models a velocity channel and would evaporate in a system that had to *see*. Answering it
needs the whole study re-run from images, which the renderer makes possible without
regenerating any episode: it renders directly from the stored state vector, so every existing
dataset — four domains, both benchmarks, every seed — becomes an image dataset by a pure
function of data already on disk. 96x96 RGB, rotated squares so yaw stays observable, no
motion blur or trails so a single frame carries **no** velocity information at all.

**Both perception front ends fail to localise these objects, and the reason is measurable.**

| front end | match distance | chance | healthy |
|---|---|---|---|
| Slot Attention (Locatello 2020) | **33.4 px**, unchanged across epochs | ~33 px | <10 px |
| spatial-softmax keypoints (Finn 2016) | **21.6 px** after 20 epochs, plateaued | ~33 px | <10 px |

Slot Attention does not merely converge slowly — it **collapses**. Its match distance was
identical to two decimal places at epochs 0 and 2 while reconstruction loss fell, which is the
signature of every slot mask sitting at the image centre: the Hungarian matching then returns
the same constant cost no matter where the objects are. The keypoint encoder does better and
its keypoints do separate (spread 0.01 → 7.6 px), but 21.6 px on a 96 px image is still not
finding objects that are 8 px across.

**Cause, measured rather than guessed.** A 5 cm object on a 60 cm table covers about **0.7% of
the frame**; three objects give ~2% foreground. A decoder that reproduces the dark background
and nothing else already achieves most of the attainable reconstruction loss, so plain MSE
supplies almost no gradient pressure to represent the objects. This is a property of the scene
geometry, not of either architecture — and note that the sparsity of the *image* here is a
direct consequence of the sparsity of *change* that the whole project is about, so it is not
an incidental nuisance.

**Fix implemented, not yet validated.** `KeypointAutoencoder.loss(foreground_weight=...)`
scales each pixel's contribution by its deviation from the image's own **median colour**,
computed per batch from the pixels themselves — no labels, no masks, and no use of the known
background constant, so the front end stays unsupervised. It changes which errors the
objective cares about, not what information it is given. A unit test pins the property
(errors on object pixels count >10x those on background). Training runs to validate it were
repeatedly killed by memory pressure from the concurrently-running data pipelines, so **no
pixel number is reported here and none should be quoted.**

**What this costs the paper.** The privileged-state limitation stands unaddressed. It should
be stated as a limitation rather than papered over, together with this diagnosis — which is
itself worth reporting, because "standard object-centric perception front ends fail on
sparse-foreground manipulation scenes" is a useful negative for anyone attempting the same
thing. The remaining work is a single uninterrupted training run per condition on a machine
that is not simultaneously generating data.

## ⚠ Known issue: the splits behind every number below leak (found 2026-08-08)

The pipeline runs `create_hard_subset` **then** `split_dataset`. `create_hard_subset` sets
`done=True` at the end of every kept chunk, so `split_dataset` subsequently treats each
chunk as an episode and fingerprints it by that chunk's *first row* — mid-trajectory object
poses rather than the episode's initial configuration. Chunks carved from a single
trajectory therefore receive different fingerprints and are assigned to splits
independently. The `configuration_leakage: false` guard does not detect this: it checks
only that one fingerprint never appears in two splits, and these fingerprints do differ.

Measured at 3 objects / seed 0: 765 chunks come from 247 source episodes, and **62 of those
episodes (25%) have chunks in both the train and the test split**; 107 span more than one
split. Train and test therefore contain states from the same rollout a few simulator steps
apart.

**Scope of the damage — measured, 3 seeds.** `experiments/leakage_impact.py` scores three
conditions: `published` (old checkpoint, old test split — reproduces the tables below),
`old_on_clean` (the *same* model on the episode-disjoint test split, so the gap is pure
memorisation with training held fixed), and `clean` (retrained on clean splits — the honest
number). Overall per-object L2, mean over seeds 0/1/2:

| N | model | published | old_on_clean | clean | inflation | ratio |
|---|---|---|---|---|---|---|
| 3 | sparse | 0.1363 | 0.1477 | 0.1484 | +0.0115 | 1.08× |
| 3 | dense | 0.3475 | 0.3156 | 0.3657 | −0.0319 | 0.91× |
| 5 | sparse | 0.1013 | 0.1000 | 0.1053 | −0.0013 | 0.99× |
| 5 | dense | 0.3191 | 0.2664 | 0.3453 | −0.0527 | 0.83× |
| 8 | sparse | 0.0705 | 0.0914 | 0.0931 | +0.0209 | **1.30×** |
| 8 | dense | 0.3286 | 0.2831 | 0.3863 | −0.0455 | 0.86× |

**The leak flattered the sparse model specifically.** Its error rises once the memorised
rows are removed — by 30% at N=8, the count with the smallest published margin — while the
dense monolith's *falls*. That asymmetry is what you would expect: a model that copies
unchanged objects and regresses a small residual can memorise a repeated state; a monolith
whose error is dominated by systematic hallucination cannot, so for it the clean split is
simply a slightly different draw.

**The headline claim survives.** Sparse still beats dense at every count and every seed on
clean splits, and the ratio barely moves: 2.55× → 2.46× (N=3), 3.15× → 3.28× (N=5),
4.66× → 4.15× (N=8). So the ordering and the order-of-magnitude are safe; the individual
figures are not, and every number quoted in a paper should come from the clean splits.
Full table: `experiments/runs/leakage_impact/leakage_impact.md`.

**Fix.** `experiments/build_clean_splits.py` reverses the order — split whole episodes
first, then filter within each split — and verifies episode-disjointness by reading back a
`source_episode` tag rather than trusting construction. Output lives in
`data/transitions/splits_clean_{N}obj_s{S}/`, with `_full_` splits (long contiguous runs,
required for multi-step rollout training) and `_hard_` splits (motion-filtered, for the
headline metrics) where the latter is a subset of the former. Baselines retrained there via
`bash experiments/clean_pipeline.sh N SEED`. The original `splits_{N}obj_s{S}` directories
are untouched so the numbers below remain reproducible as historical record.

## Elevator claim — SUPERSEDED, and its replacement

> **⚠⚠ THIS SECTION PREDATES THE 2026-08-15 SHORTCUT AUDIT AND IS ITSELF PARTLY SUPERSEDED.**
> It was written on 2026-08-08, when the onset benchmark was still believed to be a
> *correction*. Point 1 below ("the gate works — but the old benchmark could not show it")
> rests on the onset benchmark restoring the expected ordering, and READ SECOND shows it does
> not: a zero-parameter proximity rule scores 0.925 there against the learned gate's 0.694.
> **The current elevator claim is the one in READ SECOND → "What the paper should claim".**
> What still stands from this section, unaffected by the audit, is the *error-suppression*
> mechanism in point 1's second half (hard vs soft gating on unchanged-object L2, which is
> measured with detection held fixed and does not depend on any benchmark's difficulty) and
> point 3 (the gate as a causal mask). Point 2 stands as a statement about featurisation, not
> as evidence that the onset benchmark is sound.

> **The original claim, retained for the record:** *"Explicitly modeling what changes — a
> per-object change gate plus a residual delta head — predicts tabletop-push dynamics far more
> accurately than a dense monolithic predictor, at a fraction of the parameters… The advantage
> is concentrated exactly where it should be: detecting and localizing the sparse set of
> objects that actually move."*
>
> The last sentence is **false as stated**. Detection is won by a one-line velocity rule (see
> the momentum-shortcut section at the top), and on the sub-task requiring genuine prediction
> the gate is a liability at N ≥ 5. The comparison "against a dense monolith" is also a straw
> man: the monolith is degenerate by construction.

**The corrected benchmark exists and the models were re-run on it** (`--filter-mode onset`).
There the trivial rule scores 0.11–0.28 instead of 0.90, learned models beat it decisively,
and the velocity-free `contact` featurisation wins at every count with the margin *widening*
in object count (+0.10 / +0.18 / +0.26) — the exact reverse of its ranking on the old metric.
So this is a corrected benchmark with results on it, not only a criticism of the old one.

**What the evidence actually supports, after the 2026-08-08 audit:**

1. **The gate works — but the old benchmark could not show it.** On the corrected onset
   benchmark the gated models beat the trivial velocity rule by 0.54 / 0.32 / 0.11 and beat
   every ungated model by ~0.12, where on the motion benchmark they *lost* to one line of
   code. Separately, with detection held fixed (soft and hard gating tie: 0.647 vs 0.653 at
   N=3) the hard gate is ~2× better on unchanged-object L2 — down from 8–47× on the motion
   benchmark, which makes it more credible rather than less, since at 47× the definitional
   objection (a hard gate firing 0 emits exactly zero) could not be ruled out. The
   error-suppression mechanism also **compounds with object count** (dense unchanged-error
   0.271 → 0.438 monotone from N=3 to N=20 while the sparse model stays flat), replicated in
   a second environment.

2. **Velocity-free contact featurisation is what buys genuine prediction.** Trained on the
   corrected onset benchmark it reaches F1 0.752 / 0.677 / 0.654 against the velocity-based
   model's 0.653 / 0.493 / 0.394, and the *same* design change independently fixes planning
   (0.23 vs 0.00) and resists observation noise (−4.8% vs −19% F1 at 2 cm pose error). One
   ingredient, three failure modes — because it cannot take the momentum shortcut.

3. **The learned gate is a usable local causal mask.** It matches the ground-truth mask for
   counterfactual splicing (0.948 vs 0.952 validity against the simulator), and data generated
   with it helps where causally-blind splicing actively hurts.

### Status map — read this before quoting anything below

*Rewritten 2026-08-15. Three claims that were "current" the previous time this table was
written have since failed against proper baselines. They are listed as refuted rather than
removed, because what refuted each of them is itself a result.*

| section | status |
|---|---|
| **Shortcut audit** (READ SECOND) | **current, and the primary result**, 11 trivial rules × 3 counts × 3 seeds |
| **Five published models on change detection** | **current**, 3 counts × 3 seeds, native objectives, capacity-matched |
| **Four-engine domain suite + characterisation** | **current**, 4 engines × 3 counts × 3 seeds |
| Momentum shortcut / onset (READ FIRST) | **current as a diagnosis**; its *remedy* is superseded by the audit |
| "Corrected onset benchmark" | **REFUTED as a correction** — proximity rule 0.925 (0 params) vs learned gate 0.694. It removes the velocity shortcut and nothing more |
| W5 planning: "only the sparse model plans" | **REFUTED** — 4 of 5 published models plan; PETS 0.350 > sparse 0.250 |
| Full ladder on the onset benchmark | **current but reframed** — it ranks models on a benchmark a one-liner wins, so quote it for the *ungated degeneracy* only |
| Extended ablation (W2), motion benchmark | **current**, and its degeneracy finding is now the strongest thing here — reproduced on 5 published architectures |
| Counterfactual augmentation (W3) | **current, extended** — now 3 counts × 3 seeds; mask advantage grows with N (+0.046/+0.085/+0.089) |
| Scaling at one geometry (W4) | **current**, 5 seeds, clean splits |
| Delta-head study (W1) | **current**, 3 seeds — and now *corroborated from control*: PETS, the only likelihood-trained baseline, is the best planner |
| Unchanged-object L2 / error suppression | **current** — the one mechanism claim no baseline has matched |
| Headline numbers, Findings 1–4 | **superseded** — leaky splits, straw-man baseline |
| Dense-interaction control | **retracted** — does not reproduce |
| Parameter efficiency | **de-emphasised** — true but uninteresting; invites the "just smaller" objection |
| Oracle-gate, param-matched, transfer, sample efficiency | **re-verified** on clean splits, 3 seeds |
| Interaction benchmark | **current, and negative** — built, trained on, and audited (billiards, N=5/8 x 3 seeds). It defeats `nearest_to_pusher` (0.925 -> 0.437) and is then won by `moving_or_near_mover` at 0.955 vs the learned gate's 0.772 |
| Pixel benchmark | **built, not run** — Slot Attention collapses on these scenes (objects ≈0.7% of frame); a keypoint front end with a foreground-weighted loss is implemented as the alternative but **not yet validated**. No pixel number may be quoted |
| Audit battery coverage | **clutter onset built 2026-08-18, but UNDERPOWERED and not reportable** (38–174 train rows/cell; see the boxed note in READ SECOND). Rebuilding at 12000 episodes via `clutter_onset_rebuild.sh` is the outstanding item. Otherwise: tabletop-motion, tabletop-onset, planar, dense-packed and interaction-billiards are covered |
| Seed count on the audit tables | **n=3, at the significance floor** (min attainable p = 0.250). 5 seeds are required before any audit comparison is quoted as significant |
| Planning vs published baselines | **1 training seed, 20 episodes (± 0.10)** — the PETS-over-sparse ordering is within noise for a single comparison and must not be quoted as an ordering until re-run at 3 seeds |

**What a reader should take from this table.** The architecture claims have narrowed to one
(error suppression on unchanged objects) plus one methodological contribution (the audit).
Every claim that rested on comparison against baselines we wrote ourselves has either failed
or been reframed once real baselines were run. That pattern is the paper.

## Setup

- **Env**: MuJoCo tabletop pushing, procedural XML, `num_objects` free boxes (5 cm), end-effector
  delta-xy actions, scripted pushing policy. `models/envs/mujoco_tabletop.py`.
- **Data**: 250 episodes × 100 steps per (object-count, seed); `(s_t, a_t, s_{t+1})` with
  ground-truth per-object changed mask + delta. Filtered to a **hard** subset (steps with real
  motion, `min-max-xy-delta 0.02`) so the metric isn't dominated by trivially static steps.
  80/10/10 split with an explicit configuration-leakage guard (`configuration_leakage: false`).
- **Models**: dense `DenseStatePredictor` (MLP, 256×3) regressing all poses; sparse
  `SparseResidualHead` (per-object gate + residual delta, Gumbel straight-through, auto-balanced
  BCE, `sparsity_weight 0.2`). State dims auto-infer from data — no per-count retuning.
- **Baseline**: no-op ("predict no change").
- **Scale**: 3, 5, 8 objects. 8-obj placement uses wider bounds (±0.22 m) / tighter separation
  (0.09 m) to fit 8 boxes; 3/5-obj use env defaults. See caveat below.

## Headline numbers (mean ± std, 3 seeds)

| N | sparse F1 | sparse precision | sparse changed-obj L2 | dense overall L2 | sparse overall L2 | no-op overall L2 | param ratio (dense/sparse) |
|---|---|---|---|---|---|---|---|
| 3 | 0.867 ± 0.021 | 0.953 ± 0.046 | 0.359 ± 0.116 | 0.347 ± 0.053 | 0.136 ± 0.046 | 0.149 ± 0.045 | 11.1× |
| 5 | 0.802 ± 0.024 | 0.923 ± 0.024 | 0.408 ± 0.024 | 0.319 ± 0.007 | 0.101 ± 0.004 | 0.108 ± 0.001 | 9.8× |
| 8 | 0.829 ± 0.008 | 0.966 ± 0.009 | 0.398 ± 0.093 | 0.329 ± 0.023 | 0.071 ± 0.016 | 0.071 ± 0.016 | 8.6× |

Full per-model table incl. recall / unchanged-L2 / FLOPs / latency: `paper_tables/main_results.md`
and `paper_tables/efficiency.md`.

## Findings (each is a paper claim with support)

> *Computed on the leaky splits — see the warning above. Findings 1–4 below are the original
> claims and are retained as the historical record. Their re-measured forms live in
> "Scaling at one geometry" (W4), "Extended ablation" (W2) and the leakage-impact table; the
> **ordering** in each survives, the individual values do not.*

1. **Sparse ≫ dense on prediction accuracy, at every scale.** Overall per-object L2 is ~2.5–4.6×
   lower than dense across 3/5/8 objects, with non-overlapping error bars. The dense MLP must
   regress every object and injects error into the many unchanged ones (dense unchanged-L2 grows
   0.21 → 0.27 with N). *Core claim — robust.*

2. **The durable win over no-op is change detection, not raw regression.** Sparse holds gate
   F1 0.80–0.87 and precision 0.92–0.97 across all counts (no-op is 0 by construction). On the
   *overall* L2 metric, sparse's margin over no-op shrinks as scenes get sparser (0.013 → 0.007 →
   0.0003) because a larger fraction of objects genuinely don't move — but **no-op never wins on
   any seed at any count** (`no_op_trivially_wins: false` × 9). This is why we report
   change-detection F1 + changed-object L2 as primary and overall L2 as secondary.

3. **Parameter efficiency is the honest efficiency story.** Sparse uses 8.6–11.1× fewer
   parameters than dense at equal-or-better accuracy. We explicitly do **not** claim a
   wall-clock or FLOP win: the FLOP ratio erodes from 3.8× (3 obj) to 1.1× (8 obj) as per-object
   heads scale with N, and dense's single matmul is faster in latency than per-object gating.
   Reported transparently in `paper_tables/efficiency.md`.

4. **Sparsity penalty trades recall for precision, as intended.** A clean 5-point sweep
   (0.0–1.0, 3 obj, seed 0) shows the penalty makes the gate more conservative: recall falls
   0.81 → 0.71 while precision rises 0.94 → 0.97; F1 degrades gently (0.87 → 0.82) and pose L2 is
   essentially flat. The chosen 0.2 sits in the stable middle. `paper_tables/sparsity_ablation.md`.

## Scaling at one geometry, N=3→20 (W4) — and the 8-object confound retired

*`experiments/scale_series_pipeline.sh` + `experiments/scale_analysis.py`. **Mean over 5
seeds (0–4)**, clean episode-disjoint splits, bounds ±0.26 and separation 0.09 identical for
**every** count — so N is the only variable, unlike the published series which switched
geometry at N=8.*

The geometry choice matters and was constrained. Holding object *density* constant would need
bounds up to ±0.735 at N=50, against a table of half-size 0.34 and a pusher reaching only
±0.26 — objects would sit off the table and out of reach, and most would never be touched,
flattering the sparse model for trivial reasons. Fixed bounds instead let density rise with
N, which is the axis the "copying unchanged objects compounds" prediction is actually about.
Probing confirmed N=3,5,8,12,20 all place reliably at one setting; N=30/50 need separation
0.075/0.055 and would reintroduce a geometry shift, so they are excluded.

| N | changed frac | sparse F1 | dense F1 | sparse unchanged L2 | dense unchanged L2 | sparse overall | no-op overall |
|---|---|---|---|---|---|---|---|
| 3 | 0.364 | 0.865 | 0.534 | 0.0011 | 0.271 | 0.1373 | 0.1487 |
| 5 | 0.230 | 0.841 | 0.374 | 0.0014 | 0.280 | 0.0970 | 0.1035 |
| 8 | 0.160 | 0.809 | 0.275 | 0.0010 | 0.325 | 0.0723 | 0.0766 |
| 12 | 0.140 | 0.812 | 0.245 | 0.0009 | 0.385 | 0.0663 | **0.0653** |
| 20 | 0.102 | 0.749 | 0.186 | 0.0022 | 0.438 | 0.0447 | **0.0423** |

**1. The mechanism compounds with object count, exactly as predicted.** The dense monolith's
error on objects that did not move climbs monotonically, 0.271 → 0.438 (+62%), while the
sparse model stays flat at ~0.001–0.002 — it is copying them verbatim, so it has nothing to
accumulate. The gap `dense − sparse` is monotone across all five counts: 0.270 / 0.279 /
0.324 / 0.384 / 0.435.

**2. The change-detection advantage widens.** Sparse F1 declines gently (0.865 → 0.749) while
dense collapses (0.534 → 0.186), so the gap grows 0.332 → 0.563. The ungated model degrades
faster than the gated one as scenes get sparser, which is the same degeneracy the W2 ladder
isolates, now traced across a 7× range of object count.

**3. The pre-registered check failed, and the statistic was at fault.** The prediction was
encoded as "does the dense/sparse unchanged-L2 **ratio** grow with N?" It does not: 249 →
204 → 332 → 449 → 200, neither monotone nor larger at the end than the start. The reason is
that the ratio's denominator is the sparse model's unchanged-object error, ~0.001, which is
essentially the gate's false-positive rate — a near-zero, noisy quantity that swings the
ratio by 2× between adjacent counts while meaning almost nothing. The **difference** is the
quantity that tracks the mechanism, and it is cleanly monotone. Recorded here rather than
quietly swapped: the ratio statistic (used in the published tables) should be retired in
favour of the gap.

**4. The overall-L2 saturation gets worse, and now crosses over.** The changed-object
fraction falls 0.364 → 0.102 as the fixed table gets more crowded, so "predict no change"
strengthens with N. Sparse leads no-op at N=3/5/8 but **loses at N=12 and N=20** (0.0663 vs
0.0653; 0.0447 vs 0.0423). Combined with the dense-interaction rescue failing to reproduce on
clean splits, this settles the metric question: overall per-object L2 is not usable at high
object count, and the paper's primary metrics must be change-detection F1 and
unchanged-object L2 — on which the advantage is not close and *grows* with N.

## Dense-interaction control (resolves the high-N saturation)

To test whether the overall-L2 saturation at N=8 is a metric artifact of scene *sparsity*
rather than a model failure, we regenerated a **dense-interaction** variant at each count:
objects are packed at generation (per-count `--object-bound`/`--min-object-separation`) so pushes
cascade into neighbors, roughly doubling the multi-object-change fraction (e.g. 8-obj: 0.34 → 0.50).
Same models, training, and 3 seeds.

Sparse advantage over no-op on overall L2 (larger = better):

| N | sparse-scene margin | dense-interaction margin |
|---|---|---|
| 3 | 0.0131 ± 0.0013 | 0.0083 ± 0.0051 |
| 5 | 0.0067 ± 0.0048 | 0.0074 ± 0.0035 |
| 8 | **0.0003 ± 0.0002** (tied) | **0.0042 ± 0.0020** (~12×) |

The margin no longer decays toward zero at high object count; change-detection F1 stays strong
(0.80–0.85) and sparse remains ~2–4.4× better than dense. **Conclusion: the saturation is scene
sparsity, not a sparse-model weakness.** Full detail: `paper_tables/dense_vs_sparse_scene.md`,
`paper_tables/main_results_dense.md`.

> ### ⚠ This control does NOT reproduce on clean splits — do not use it
>
> Rebuilt end to end on the episode-disjoint splits (`experiments/dense_interaction_clean.py`,
> same 3 seeds, packed-object data regenerated through `build_clean_splits`):
>
> | N | sparse-scene margin | dense-interaction margin | positive at every seed? |
> |---|---|---|---|
> | 3 | +0.0091 ± 0.0050 | +0.0098 ± 0.0025 | yes / yes |
> | 5 | +0.0095 ± 0.0023 | +0.0083 ± 0.0016 | yes / yes |
> | 8 | **+0.0041 ± 0.0030** | **+0.0011 ± 0.0033** | **no / no** |
>
> At N=8 the two regimes essentially **swap** relative to the table above: the sparse-scene
> margin comes out *larger* than published (0.0003 → 0.0041) and the dense-interaction margin
> *smaller* (0.0042 → 0.0011), with neither positive at every seed. The claimed ~12× restoration
> is an artefact of the leaked splits.
>
> The manipulation itself worked — the multi-object-change fraction still roughly doubles at
> N=8 (0.370 → 0.512), so this is a failed *rescue*, not a failed intervention. Packing the
> scene does not save overall L2 at high object count.
>
> **What this does and does not cost.** It removes the answer to high-N overall-L2 saturation,
> so the N=8 overall-L2 column should be reported as *saturated in both regimes* rather than
> rescued. It costs nothing on the metrics the paper already designates primary, which are not
> close at N=8: change-detection F1 is 0.774 for sparse against 0.304 for every ungated model
> including the relational ones, and unchanged-object L2 is 0.0006 against 0.2849 for dense.
> The correct response is to lean harder on the framing RESULTS.md already argues for — overall
> L2 is a weak metric in sparse scenes and F1 / unchanged-object L2 carry the claim — rather
> than to look for a different rescue.

## Parameter-matched dense baseline — removing the "sparse is just smaller" confound

The headline table gives dense 8.6–11.1× more parameters than sparse, so "sparse wins"
is confounded with "sparse is smaller." We remove the confound by shrinking the dense
MLP to the *same* parameter budget (`hidden_dim=64, num_layers=3` matches sparse to
**1.00×** at every count) and retraining it. `experiments/param_matched_baseline.py`.

At equal parameter count (test split, seed s0):

| N | sparse overall-L2 | dense-matched overall-L2 | sparse F1 | dense-matched F1 | params |
|---|---|---|---|---|---|
| 3 | **0.116** | 0.363 | **0.885** | 0.538 | ~6.9k each |
| 5 | **0.104** | 0.363 | **0.777** | 0.388 | ~8.5k each |
| 8 | **0.081** | 0.415 | **0.837** | 0.294 | ~10.8k each |

**Sparse still wins on every metric at every count** (overall-L2 ~3–5× lower, and it
actually detects change while dense stays degenerate). Shrinking dense to sparse's budget
does *not* help it — the matched dense is if anything slightly worse than the full dense
on overall-L2 (e.g. N=8: 0.415 vs 0.354), and its change-detection F1 is unchanged (the
degeneracy is architectural, not a capacity issue). The advantage is the object-centric
structure, not the parameter count. Figure:
`experiments/runs/param_matched_baseline/param_matched.png`.

## Gate ablation — is the win object-centricity, or actually modeling change?

The parameter-matched control removes the "sparse is just smaller" confound, but one
confound survives it, and it is the first one a reviewer reaches for: the sparse model is
both **object-centric** (per-object features, weights shared across objects) *and*
**change-modeling** (a discrete gate over a residual), while the dense monolith is neither.
The headline gap therefore cannot say which ingredient earns the win. We interpolate the
two designs with a **capacity-matched ladder** where each rung adds exactly one ingredient
(`experiments/gate_ablation.py`, seed 0, test split; the `oc_*` rungs are width-matched to
the sparse model's *total* parameter count to within 0.04%):

| rung | object-centric | residual | change gate |
|---|---|---|---|
| `dense` | — | — | — |
| `oc_absolute` | ✓ | — | — |
| `oc_residual` | ✓ | ✓ | — |
| `sparse` | ✓ | ✓ | ✓ |

Overall per-object L2 (lower better) and change-detection F1:

| N | dense | oc_absolute | oc_residual | **sparse** | no-op | ungated F1 | **sparse F1** |
|---|---|---|---|---|---|---|---|
| 3 | 0.367 | 0.239 | 0.221 | **0.116** | 0.128 | 0.538 | **0.885** |
| 5 | 0.318 | 0.194 | 0.171 | **0.104** | 0.107 | 0.388 | **0.777** |
| 8 | 0.354 | 0.187 | 0.159 | **0.081** | 0.081 | 0.294 | **0.837** |

**Three findings, and the third is the important one.**

1. *Object-centric featurization is worth real accuracy but not the whole gap.* Moving from
   the monolith to a shared-weight per-object MLP at 1/8th the parameters cuts overall L2
   by 35–47% (e.g. N=3: 0.367 → 0.239). The residual parameterization adds a further
   modest gain (0.239 → 0.221).

2. *The change gate is the single largest step.* `oc_residual` → `sparse` is a **1.65–1.96×**
   reduction in overall L2 (N=3 1.91×, N=5 1.65×, N=8 1.96×) — larger than either preceding
   rung — and it is the *only* rung that pulls the model below the no-op reference at N=3
   and N=5.

3. *Change detection is not degenerate because the model is monolithic; it is degenerate
   because the model is ungated.* `dense`, `oc_absolute`, and `oc_residual` produce
   **byte-identical** detection metrics at every count (F1 0.538 / 0.388 / 0.294, recall
   exactly 1.000): every ungated regressor flags every object as changed, no matter how
   object-centric it is. Only the gated model detects change (F1 +0.35 / +0.39 / +0.54).
   This is a sharper statement than the earlier "the dense degeneracy is architectural" —
   the degeneracy belongs to *ungated regression*, and object-centric structure alone does
   not cure it.

The mechanism shows up cleanly in the **unchanged**-object column: 0.211 (dense) → 0.111
(`oc_absolute`) → 0.086 (`oc_residual`) → **0.0013** (sparse) at N=3. The gate is what stops
error being injected into the static majority; that is ~70× better than the best ungated
object-centric model, and it is the whole ballgame. Full table:
`experiments/runs/gate_ablation/gate_ablation.md`, figure `.../gate_ablation.png`.

*Caveat: seed 0. Re-running with `--width-mode identical` — which gives the `oc_*` rungs a
delta head architecturally identical to sparse's, so they carry ~half sparse's parameters
(3.6k / 4.4k / 5.5k vs 6.9k / 8.5k / 10.8k) rather than being width-matched — reproduces
the same ordering and near-identical gate steps (1.93× / 1.59× / 1.98×), so the conclusion
does not depend on how the ablations are sized:
`experiments/runs/gate_ablation_identical/gate_ablation.md`.*

## Extended ablation (W2) — relational baselines and soft sparsity

*Clean episode-disjoint splits, **mean ± std over 3 seeds (0/1/2)**, 15 epochs,
`sparse`/`dense` retrained there too so no rung is scored on data it partly memorised.
`experiments/gate_ablation.py --extended`.*

The published ladder cannot answer the two objections a reviewer reaches for first. Its
ungated rungs process each object *independently*, so "ungated" is confounded with "no
interaction modelling"; and it never tries the obvious cheap alternative to a discrete
gate, which is to penalise the deltas. Four rungs close both holes: `gnn` (interaction
network) and `set_transformer` (multi-head self-attention over the object set), both
permutation-equivariant with an always-applied residual; and `dense_l1` (monolith + L1 on
implied deltas) and `soft_gate` (the sparse architecture with a continuous sigmoid instead
of a Gumbel straight-through sample). All are capacity-matched to the sparse model.

| N | dense | dense_l1 | oc_residual | gnn | set_transformer | soft_gate | **sparse** | no-op |
|---|---|---|---|---|---|---|---|---|
| **overall L2** ||||||||
| 3 | 0.3657±0.0250 | 0.3660±0.0248 | 0.2322±0.0121 | 0.2209±0.0214 | 0.2394±0.0079 | 0.1580±0.0050 | **0.1484±0.0058** | 0.1576±0.0080 |
| 5 | 0.3453±0.0148 | 0.3663±0.0166 | 0.1895±0.0156 | 0.1584±0.0188 | 0.1870±0.0203 | 0.1181±0.0224 | **0.1053±0.0240** | 0.1149±0.0218 |
| 8 | 0.3863±0.0431 | 0.3937±0.0363 | 0.1799±0.0153 | 0.1389±0.0219 | 0.1699±0.0288 | 0.1108±0.0230 | **0.0931±0.0221** | 0.0972±0.0238 |
| **change-detection F1** ||||||||
| 3 | 0.5394±0.0076 | 0.5394±0.0076 | 0.5394±0.0076 | 0.5387±0.0085 | 0.5394±0.0076 | **0.8634±0.0134** | 0.8577±0.0143 | 0.000 |
| 5 | 0.4054±0.0075 | 0.4054±0.0075 | 0.4054±0.0075 | 0.4050±0.0070 | 0.4054±0.0075 | **0.8217±0.0025** | 0.8211±0.0023 | 0.000 |
| 8 | 0.3134±0.0181 | 0.3134±0.0181 | 0.3134±0.0181 | 0.3135±0.0182 | 0.3134±0.0181 | 0.7954±0.0136 | **0.7966±0.0181** | 0.000 |
| **unchanged-object L2** ||||||||
| 3 | 0.2403±0.0255 | 0.2406±0.0254 | 0.0897±0.0116 | 0.0750±0.0190 | 0.0992±0.0096 | 0.0171±0.0015 | **0.0006±0.0002** | 0.000 |
| 5 | 0.2425±0.0081 | 0.2608±0.0165 | 0.0841±0.0131 | 0.0455±0.0041 | 0.0849±0.0027 | 0.0191±0.0016 | **0.0024±0.0004** | 0.000 |
| 8 | 0.3069±0.0206 | 0.3129±0.0152 | 0.0894±0.0135 | 0.0473±0.0062 | 0.0848±0.0065 | 0.0236±0.0017 | **0.0005±0.0002** | 0.000 |

**1. The ungated degeneracy survives relational modelling — this is the sharpest form of
the paper's third finding.** `dense`, `dense_l1`, `oc_absolute`, `oc_residual` and
`set_transformer` produce **identical detection metrics to four decimal places at every
count and every seed** (recall exactly 1.000; F1 0.5394 / 0.4054 / 0.3134), and the
interaction network is within 0.001 of them. An architecture that can attend to every other
object, weight-shared and fully permutation-equivariant, *still* flags everything as
changed. Interaction modelling does not cure the degeneracy; only gating does (+0.32 /
+0.42 / +0.48 F1).

**2. But relational structure is worth real accuracy, and the ladder's ordering changes.**
The interaction network cuts overall L2 well below `oc_residual` at every count —
0.1799 → 0.1389 at N=8, and 0.1895 → 0.1584 at N=5 — a larger step than the residual
parameterisation contributes. So the published claim that the *gate* is "the single largest
step" needs qualifying: it is the largest step **among the ingredients originally tested**,
but message passing is worth more than the residual parameterisation, and the honest ladder
is dense → oc_absolute → oc_residual → gnn → sparse.

**3. Soft sparsity is not a substitute for a gate.** `dense_l1` is indistinguishable from
plain `dense` (0.3660 vs 0.3657 at N=3, well inside ±0.025) and *worse* at N=5 and N=8, with
detection metrics identical to the unpenalised model. An L1 penalty on implied displacements
changes nothing about the monolith's behaviour. The "why not just regularise the deltas"
objection has a clean empirical answer.

**4. Discreteness matters, and it matters exactly where the paper says it should — this is
the cleanest result in the study.** The soft sigmoid gate is **statistically tied** with the
hard gate on *detection*: F1 0.8634±0.0134 vs 0.8577±0.0143, 0.8217±0.0025 vs 0.8211±0.0023,
0.7954±0.0136 vs 0.7966±0.0181 — soft marginally ahead at N=3 and N=5, sparse at N=8, all
overlapping. The entire difference is on the **unchanged** objects, and there the gap is
enormous and the error bars do not come close to touching:

| N | soft gate | hard gate | ratio |
|---|---|---|---|
| 3 | 0.0171 ± 0.0015 | **0.0006 ± 0.0002** | 28× |
| 5 | 0.0191 ± 0.0016 | **0.0024 ± 0.0004** | 8× |
| 8 | 0.0236 ± 0.0017 | **0.0005 ± 0.0002** | 47× |

A continuous gate leaks a small residual into every static object; a hard gate copies them
verbatim. That propagates straight into overall L2, where sparse leads soft at every count.
This isolates the paper's central mechanism — *not injecting error into the objects that did
not move* — to the single design choice that produces it, with detection held fixed.

So `hard_gate_beats_soft_sparsity_everywhere` is recorded as **False**, and correctly so:
the hard gate does not win on detection, it wins on *not injecting error into the static
majority*. The precise claim W2 supports is narrower and better than the one it set out to
test.

**Two honest problems this run surfaces.**

- **At N=8 sparse no longer beats no-op on overall L2** (0.0741 vs 0.0739 at this seed;
  +0.0041 ± 0.0030 over 3 seeds, not positive at every seed). The dense-interaction control
  was the existing answer to high-N saturation — it has now been rerun on clean splits and
  **does not reproduce** (+0.0011 ± 0.0033 at N=8; see the boxed note in that section). So
  the N=8 overall-L2 column is saturated in both regimes and should be reported that way.
- **`gnn` and `set_transformer` lose to no-op at N=8 outright** (0.1389 / 0.1699 vs 0.0972),
  by margins far outside the error bars. Every ungated model is beaten by "predict no change"
  at high object count. This reinforces that overall L2 is a weak metric in sparse scenes;
  change-detection F1 and unchanged-object L2 carry the argument, and on those the gap is
  enormous.

Full tables: `experiments/runs/gate_ablation_extended_clean{,_s1,_s2}/gate_ablation.md`.

## Oracle-gate diagnostic — the bottleneck is regression, not detection

Finding 2 noted that sparse's *changed-object* L2 sits close to the no-op reference. That
is ambiguous: is the gate missing changed objects (detection), or is the delta head
regressing them poorly even when it fires (regression)? We separate the two by feeding the
**ground-truth changed mask** to the delta head. `experiments/oracle_gate_diagnostic.py`.

Changed-object L2 (restricted to objects that truly moved), test split s0:

| N | predicted-gate | oracle-gate (perfect detection) | no-op |
|---|---|---|---|
| 3 | 0.312 | 0.314 | 0.348 |
| 5 | 0.426 | 0.432 | 0.446 |
| 8 | 0.466 | 0.474 | 0.470 |

**Perfect detection barely moves the number** (the detection gap is ≈0, even slightly
negative — the gate is weakly *helpful* in suppressing deltas it would get wrong). Even
with the true mask, the delta head only shaves ~10% off no-op at N=3 (0.348→0.314) and is
essentially at no-op by N=8 (0.474 vs 0.470). **So the changed-object bottleneck is the
delta *regression*, not the gate** — tabletop push deltas (especially rotation, and
contact-driven motion) are simply hard to predict one step ahead. This reframes the value
proposition cleanly: the sparse model wins by *detecting* what changed and *not injecting
error into the unchanged majority*, not by superior changed-object regression — and it
points at the delta head (richer contact features, distributional/heteroscedastic output)
as the highest-leverage place to improve next.

## Delta-head study (W1) — lifting the regression bottleneck

*Run on the clean episode-disjoint splits; not comparable to the tables above.
`experiments/delta_head_study.py`, **mean ± std over 3 seeds (0/1/2)**, 25 epochs, evaluated
on the clean hard test split. The controls further down (capacity matching, component-count
sweep) are seed 0 only and say so.*

The oracle-gate diagnostic above concluded the bottleneck is delta *regression*, not
detection — with the ground-truth mask the head still only matched no-op, and at N=8 came
in slightly worse. The hypothesis was that squared error on **multimodal** contact
converges to a near-zero conditional mean. This study replaces the point regressor with a
conditional density trained by NLL: `gaussian` (heteroscedastic diagonal) and `mdn`
(5-component mixture), each in a one-step and a 5-step unrolled training regime.

Margin over no-op on changed-object L2 with the oracle gate (**positive = the head finally
beats declaring the object stationary**):

| N | mse onestep | mse rollout | gaussian onestep | gaussian rollout | mdn onestep | mdn rollout |
|---|---|---|---|---|---|---|
| 3 | +0.0333 ± 0.0107 | −0.1018 ± 0.0277 | +0.0229 ± 0.0203 | +0.0412 ± 0.0112 | +0.0451 ± 0.0082 | **+0.0479 ± 0.0116** |
| 5 | +0.0523 ± 0.0074 | −0.1203 ± 0.0434 | +0.0380 ± 0.0086 | +0.0588 ± 0.0081 | +0.0653 ± 0.0050 | **+0.0734 ± 0.0083** |
| 8 | +0.0209 ± 0.0080 | −0.1394 ± 0.0228 | +0.0311 ± 0.0054 | +0.0497 ± 0.0099 | +0.0538 ± 0.0103 | **+0.0603 ± 0.0084** |

**The head clears the no-op floor at every count, and the MDN is the best cell at every
count.** The decisive comparison is at N=8, where the published diagnostic had the head
*below* the floor at −0.004: the MDN reaches +0.0538 ± 0.0103 one-step and +0.0603 ± 0.0084
under rollout, against +0.0209 ± 0.0080 for squared error — a 2.6× separation with
non-overlapping error bars. The same ordering holds at N=5 (also non-overlapping); at N=3
the MDN/MSE bars touch, which is the weakest cell in the grid.

Aggregated table: `experiments/runs/delta_head_study_multiseed/delta_head_multiseed.md`.

**But the stated mechanism is wrong, and the test that refutes it is the useful result.**
Scoring the *same* trained MDN with the mixture mean instead of the highest-weight
component gives, over 3 seeds, −0.0002 ± 0.0017 / +0.0019 ± 0.0016 / −0.0011 ± 0.0014
one-step and +0.0073 ± 0.0055 / +0.0013 ± 0.0014 / +0.0017 ± 0.0009 under rollout at
N=3/5/8 — every cell within about one standard deviation of zero. Committing to a mode buys
nothing, which is not what the multimodality-plus-mode-selection story predicts.

**Capacity control, and what it says about the mechanism.** The MDN carries 1.38–1.60× the
deterministic head's parameters at equal width, so the margin needed an explicit control:
`--capacity-match` widens the smaller heads until every head's delta-head budget is equal
(matched to within 0.2% — 11032 / 11054 / 11044 at N=3). One-step, seed 0:

**Mean ± std over 3 seeds:**

| N | mse | gaussian | **mdn** |
|---|---|---|---|
| 3 | +0.017 ± 0.012 | +0.013 ± 0.018 | **+0.046 ± 0.008** |
| 5 | +0.043 ± 0.014 | +0.040 ± 0.015 | **+0.063 ± 0.004** |
| 8 | +0.024 ± 0.002 | +0.030 ± 0.011 | **+0.054 ± 0.011** |

Two conclusions, the second of which corrects the first reading of this study.

1. **The MDN's edge is not bought with parameters.** At matched budget it holds +0.055 /
   +0.066 / +0.048, essentially unchanged from its default-width numbers, while the widened
   deterministic head does not improve.

2. **Heteroscedasticity alone is not the ingredient — the mixture is.** At matched capacity
   the unimodal Gaussian is *not* better than squared error (worse at N=5 and N=8, a hair
   better at N=3). Since the Gaussian has the learned per-object scale and gains nothing
   from it, the "likelihood objective discounts unpredictable deltas" explanation does not
   survive. The only thing separating the MDN from the Gaussian is having more than one
   component, and that is where the entire one-step gain lives.

**Component-count sweep — the mechanism, settled.** Every K is pinned to one parameter
budget via `--capacity-target-components 5` (matched to within 0.4% inside each count), so
this measures component count and not capacity. Margin over no-op, one-step, seed 0:

**Mean ± std over 3 seeds:**

| N | K=1 | K=2 | K=3 | K=5 | K=10 |
|---|---|---|---|---|---|
| 3 | +0.019 ± 0.001 | +0.036 ± 0.005 | **+0.047 ± 0.007** | +0.046 ± 0.008 | +0.046 ± 0.004 |
| 5 | +0.047 ± 0.010 | +0.063 ± 0.012 | **+0.070 ± 0.009** | +0.063 ± 0.004 | +0.069 ± 0.010 |
| 8 | +0.031 ± 0.012 | +0.035 ± 0.005 | +0.041 ± 0.005 | **+0.054 ± 0.011** | +0.047 ± 0.012 |

The same shape at all three counts: **monotone gain from K=1 to K=3, then a plateau.** At N=3
the K=1 → K=3 step is 0.019 ± 0.001 → 0.047 ± 0.007, comfortably outside the seed spread. K=1
— a heteroscedastic Gaussian by construction, landing close to the separately-trained
`gaussian` rung, which is a useful implementation check — captures under half the available
margin. Roughly doubling it requires three components.

*Correction from the seed-0 reading:* that run showed a decline at K=10 and it was attributed
to over-parameterised mixtures splitting on noise. **It does not replicate.** Across 3 seeds
K=10 is flat with K=3/K=5 (0.046 / 0.069 / 0.047), so the honest shape is "rises to K≈3, then
plateaus", with no evidence of harm from extra components.

That resolves the mechanism, and the resolution is that **the original multimodality
diagnosis was right about the data and wrong about the remedy.** A gain that scales with
component count and saturates at three is the signature of a genuinely multimodal
conditional with a small number of modes; nothing else explains why K=3 beats K=1 at equal
parameters. But it does not act through *choosing* a mode at inference — the mode/mean
comparison is null. The consistent account is that a unimodal density fit to a multimodal
target is **biased**, so its mean is pulled off the truth, whereas a mixture that fits the
modes has an unbiased mean. Modelling the multimodality is what matters; which point
estimate you then read off it does not. The decline at K=10 is the usual
over-parameterised-mixture behaviour, components splitting on noise.

So the corrected chain is: squared error is biased on multimodal contact (original diagnosis,
**upheld**) → the fix is not mode selection (**refuted**, mode/mean null) → nor
heteroscedasticity alone (**refuted**, K=1 and the matched Gaussian capture under half) →
it is fitting the multimodal density, which needs about three components.

*Now 3 seeds (was seed 0).* The MDN's advantage survives capacity matching at every count
with non-overlapping or near-non-overlapping spreads, and **Gaussian is indistinguishable from
MSE everywhere** (0.013 vs 0.017, 0.040 vs 0.043, 0.030 vs 0.024) — the finding that rules
heteroscedasticity out as the operative ingredient. The seed-0 reading put widened MSE at
+0.0058 at N=3; across seeds it is +0.017 ± 0.012, so that particular number was on the low
side of the spread while the ordering held. This control remains one-step only.

**Rollout training splits the heads sharply, and this is the largest effect in the study.**
It *destroys* the deterministic head — **−0.1018 ± 0.0277 / −0.1203 ± 0.0434 /
−0.1394 ± 0.0228** at N=3/5/8, the only cells in the entire grid that fail to beat no-op,
and nowhere near overlapping zero — while *helping* both probabilistic heads at every count.
The MDN's best cell is its rollout cell at all three counts.

This is also where the *Gaussian* clearly earns its keep (+0.0497 ± 0.0099 against
−0.1394 ± 0.0228 at N=8), unlike the one-step regime where it is no better than squared
error. So heteroscedasticity buys **stability under unroll**, not one-step accuracy — a
cleaner division of labour between the two probabilistic ingredients than we expected.

**The obvious confound was tested, and the finding survives it.** `run_rollout_epoch`
supervises `true_next_pose − rolled_pose`, a target whose magnitude *grows with accumulated
drift* — squared error chases such targets while a learned-scale likelihood can discount
them, so the split could have been manufactured by the target rather than intrinsic to the
objectives. `--rollout-target recorded` repeats the rollout cells supervising the dataset's
own one-step delta, whose magnitude does not grow. Margin over no-op, seed 0:

| N | mse (correcting) | mse (**recorded**) | gaussian (recorded) | mdn (recorded) |
|---|---|---|---|---|
| 3 | −0.0683 | **−0.0611** | +0.0522 | +0.0602 |
| 5 | −0.0816 | **−0.0314** | +0.0284 | +0.0775 |
| 8 | −0.1444 | **−0.0678** | +0.0467 | +0.0506 |

**The deterministic head still falls below the no-op floor at every count**, and both
probabilistic heads stay solidly above it. The drift-correcting target *amplifies* the
effect (−0.144 → −0.068 at N=8) but does not create it: the sign is unchanged everywhere
under both formulations. So the interaction is real, and its magnitude — not its direction —
depends on how the unrolled target is defined.

The Gaussian-vs-MSE half of this comparison is also effectively capacity-controlled without
further work, since the two heads differ by only 1.04–1.06× in parameters, and the Gaussian
wins decisively under both targets. *The MDN's rollout numbers carry 1.38–1.60× the
parameters and are not capacity-matched* — the `--capacity-match` control covered one-step
only.

**On the second success criterion.** The pre-registered criterion was "the oracle-gate
detection gap becomes informative rather than ≈0". It did not: gaps are +0.0016 / +0.0037 /
+0.0000 at N=3/5/8. On reflection that criterion was specified backwards. A near-zero gap
means the predicted gate is already as good as the ground-truth mask *for this metric* —
which is a property of a good gate, not evidence of a bad head. Whether the head is still
the bottleneck is measured by the margin over no-op, which now passes decisively. The
criterion is recorded as failed and withdrawn as ill-posed rather than quietly dropped.

**One confound found and fixed mid-study.** Weighting the delta loss by the predicted gate
probability (`--delta-supervision predicted_probs`, the default) lets the gate reduce its
own loss by *turning itself down* on objects the head finds hard. At squared-error
magnitudes this is negligible; at NLL magnitudes it is not — gate F1 collapsed from 0.869
(MSE) to 0.430 (Gaussian) and 0.640 (MDN), which would have scored every head through a
gate of different quality. `--detach-delta-gate` (default `auto`: on for probabilistic
heads) cuts that gradient path; with it, F1 is uniform across heads at each count
(0.866–0.869 at N=3), so the table above compares heads and not gates.

## Sample efficiency — a stronger prior needs less data

If explicitly modelling *what changes* is a genuinely stronger inductive bias, it should pay off
most when data is scarce. We retrain both models from scratch on **10 / 25 / 50 / 100%** of the
training split (a shuffled prefix, so both models see the *same* subset) and evaluate each on the
*full, fixed* test split. Same hyperparameters as the headline runs; single seed s0 per count.
`experiments/sample_efficiency.py`.

Change-detection F1 on the test split vs training-set size:

| N | sparse @10% | sparse @25% | sparse @100% | dense (any %) |
|---|---|---|---|---|
| 3 | 0.675 (113) | 0.833 (282) | 0.885 (1129) | 0.538 |
| 5 | 0.502 (120) | 0.753 (300) | 0.777 (1202) | 0.388 |
| 8 | 0.351 (129) | 0.743 (322) | 0.837 (1289) | 0.294 |

*(sample counts in parentheses)*

**Sparse reaches ~90% of its full-data F1 with a quarter of the data.** At 25% of the training
split sparse is already at 94% / 97% / 89% of its 100%-data F1 for 3 / 5 / 8 objects, and it beats
dense at *every* budget including the smallest. **Dense F1 is flat across all budgets** because its
change-detection is degenerate (predicts every object changed, recall = 1.0 — see caveat below), so
more data cannot help it on this metric — its F1 is fixed by the test-set class balance. The same
picture holds on change-detection accuracy (sparse 0.66→0.92 at N=3; dense pinned at ~0.37/0.24/0.17
by its all-changed predictions; no-op at the base rate). Figures:
`experiments/runs/sample_efficiency_{3,5,8}obj_s0/sample_efficiency.png`, curves in
`sample_efficiency_curves.csv`.

One honest nuance on the *pose* metric: at the very smallest budget the sparse changed-object L2 can
sit at or slightly above no-op (e.g. N=8, 129 samples: 0.579 vs 0.470) — the delta head needs a
little data before it improves moving objects rather than the gate alone — but it crosses below
no-op by ~25% and keeps dropping, while dense's changed-object L2 is 1.5–2.9× worse throughout.

## Counterfactual augmentation (W3) — the gate is a usable causal mask

*Clean episode-disjoint splits, 3 objects, seed 0. `experiments/counterfactual_augmentation.py`,
`models/counterfactual.py`.*

Everything above treats the gate as a modelling device. This asks whether it is what the
framing claims — a **local causal mask** — by using it for something a mere change detector
could not support: generating data. If object *i* is genuinely unaffected this step, its pose
is causally independent of everything else that happened, so relocating it anywhere that
creates no new interaction leaves the transition valid. That is the CoDA argument, and it
turns the gate into a data generator.

### Stage 1: are the synthesized transitions actually physically valid?

Measured, not argued. Each splice is checked against MuJoCo: take an exact `snapshot`, run
the real step for ground truth, `restore`, relocate **only** the objects under test (via
`relocate_object`, which rewrites one object's qpos and leaves every other degree of freedom
bit-identical), then step again with the same action. Any difference is attributable to the
relocation alone.

Per-object validity, **3 counts × 3 seeds** (`w3_validity_{3,5,8}obj_s{0,1,2}`):

| N | oracle mask (ground truth) | **learned gate** | no mask (causally blind) | mask advantage |
|---|---|---|---|---|
| 3 | 0.951 ± 0.008 | **0.949 ± 0.013** | 0.867 ± 0.027 | +0.082 |
| 5 | 0.896 ± 0.022 | **0.895 ± 0.014** | 0.756 ± 0.006 | +0.138 |
| 8 | 0.880 ± 0.019 | **0.887 ± 0.019** | 0.713 ± 0.011 | **+0.174** |

**The learned gate is indistinguishable from the ground-truth mask at every count** — within
0.007 everywhere, and marginally *ahead* at N=8. That is the claim W3 needed, and unlike the
unchanged-object-L2 result it is not definitional: nothing about the architecture guarantees
that a gate trained for change detection produces placements the simulator agrees with.
`others_match_fraction` is exactly 1.000 in every condition, confirming the relocation
perturbs nothing it should not.

**The advantage over blind splicing grows monotonically with object count** (+0.082 / +0.138 /
+0.174). An earlier note in this document predicted the opposite, reasoning that at high N
almost every object is inert so blind relocation would be right by default. That was wrong:
more objects means more opportunities for a blind placement to land somewhere it changes the
dynamics, and the causal mask is what avoids them. Recorded because the prediction was written
down before the measurement.

Read the gap on the **per-object** column, not per-splice: the blind condition relocates every
object where the masked conditions relocate 1.4–1.7, so a per-splice "all stayed put" rate
would penalise it for volume rather than for being wrong.

> **Methodological warning, recorded because it nearly produced an inverted result.** The
> first version of this check reconstructed the synthetic scene from the reduced planar state
> and replayed it. That does not work: the reduced state drops z, tilt, contact history and
> velocities, so re-entering a mid-contact configuration makes the solver eject objects.
> **Unmodified** transitions scored 0% valid with 6 m errors — the instrument had no
> resolution — and the causally-blind condition scored *best*, because scattering objects
> apart reduced the interpenetration the reconstruction itself had introduced. Never verify a
> counterfactual by rebuilding the scene from a lossy state; restore an exact snapshot and
> perturb only what is under test.

### Stage 2: does the synthesized data help?

Self-bootstrapped, the only honest protocol: train the gate on a data budget, use *that* gate
to augment *that* budget, retrain from scratch on the union. No ground-truth mask and no extra
real data enter anywhere. `no_mask` adds the same volume of splices with the causal reasoning
removed.

**Mean ± std over 3 seeds** (`w3_efficiency_3obj{,_s1,_s2}`):

| budget | F1 real-only | F1 **learned gate** | F1 no-mask | margin real-only | margin **gate** | margin no-mask |
|---|---|---|---|---|---|---|
| 10% | 0.755 ± 0.021 | **0.797 ± 0.019** | 0.743 ± 0.031 | −0.124 ± 0.032 | −0.069 ± 0.005 | −0.055 ± 0.027 |
| 25% | 0.813 ± 0.035 | 0.819 ± 0.030 | 0.781 ± 0.034 | −0.047 ± 0.014 | **−0.017 ± 0.015** | −0.039 ± 0.016 |
| 50% | 0.860 ± 0.010 | 0.865 ± 0.012 | 0.816 ± 0.013 | −0.028 ± 0.014 | **−0.012 ± 0.010** | −0.013 ± 0.015 |
| 100% | 0.874 ± 0.006 | 0.882 ± 0.013 | 0.836 ± 0.014 | −0.021 ± 0.032 | **+0.030 ± 0.006** | +0.007 ± 0.014 |

**The robust claim is that the causal mask is what makes augmentation safe.** Gate-spliced data
beats blind splicing on F1 at *every* budget (+0.054 / +0.038 / +0.049 / +0.046, against seed
spreads of 0.01–0.03), and blind splicing is **worse than not augmenting at all** at three of
four budgets. Adding the same volume of physically invalid transitions actively damages change
detection. That is the result worth reporting: not "more data helps", but "data generated
without the causal mask hurts, and the gate supplies a mask good enough to avoid that".

**What is *not* robust:** gate-augmentation versus *no* augmentation. Its F1 edge is +0.042 at
the 10% budget but only +0.005 to +0.008 at 25/50/100%, inside the seed spread. The one place it
clearly separates is the delta-regression margin at full budget, where augmentation flips the
head above the no-op floor (**+0.030 ± 0.006** against −0.021 ± 0.032) — non-overlapping, and
something training on real data alone never achieves. Splice acceptance is 0.93–1.00, so the
clearance test is not the bottleneck.

**Dose-response separates the two conditions, which the level comparison alone cannot.**
Sweeping the augmentation ratio at full budget (3 objects, seed 0):

| synthetic per real | **gate** F1 | no-mask F1 | gate margin |
|---|---|---|---|
| 0.5× | 0.887 | 0.857 | −0.010 |
| 1.0× | 0.897 | 0.853 | +0.033 |
| 2.0× | **0.904** | 0.858 | **+0.038** |

Gate-spliced data improves **monotonically** with volume while blind splicing is flat
(0.857 / 0.853 / 0.858). If the gate's synthetic transitions were merely noise that happened
not to hurt, adding more of them would not help monotonically; if blind splices were merely
low-quality rather than *invalid*, more of them would eventually help too. Neither holds. This
is the cleanest evidence that what the mask supplies is validity rather than volume.

**✅ RESOLVED 2026-08-15 — the efficiency stage now covers 3/5/8 × 3 seeds**
(`experiments/close_coverage_gaps.sh`). It was previously N=3 only, which mattered because
the claim is about causal masking and the validity stage had already shown the mask's
advantage *growing* with object count (+0.082 / +0.138 / +0.174). Change-detection F1, mean
over 3 seeds:

| N | budget | real only | **learned gate** | no mask | gate − no-mask |
|---|---|---|---|---|---|
| 3 | 10% | 0.7552 | **0.7974** | 0.7431 | +0.0543 |
| 3 | 100% | 0.8741 | **0.8823** | 0.8361 | +0.0462 |
| 5 | 10% | 0.5960 | **0.7824** | 0.7571 | +0.0253 |
| 5 | 100% | 0.8296 | **0.8525** | 0.7680 | +0.0845 |
| 8 | 10% | 0.4712 | **0.7340** | 0.6541 | +0.0799 |
| 8 | 100% | 0.7969 | **0.8196** | 0.7304 | +0.0892 |

**The mask advantage grows with object count, as the validity stage predicted.** At full
budget it is +0.046 / +0.085 / +0.089 for N=3/5/8 — monotone, and the ordering matches the
per-object validity gap measured independently in stage 1. Gate-spliced data beats blind
splicing in **all 12 cells**, so the central W3 claim ("data generated without the causal
mask hurts") now rests on three object counts rather than one.

A second effect appears only at higher counts: at the 10% budget `real_only` collapses as
scenes get more complex (0.755 → 0.596 → 0.471) while gate-augmented training barely moves
(0.797 → 0.782 → 0.734). Augmentation matters most exactly where data is scarce and the
scene is crowded, which is the regime the method was motivated by.

*Still outstanding:* the augmentation-**ratio** sweep. Seeds 1 and 2 were run
(`w3_ratio_{0.5,1.0,2.0}_3obj_s{1,2}`) but the dose-response claim should be re-read off all
three seeds together before the "monotone with volume" wording is kept.

The margins are negative in most cells because this trains on the *hard* split (≤1121 rows)
rather than the full split the W1 study used (5326 rows) — a deliberately low-data regime, so
these margins are **not** comparable to the W1 table.

## Compositional generalization — transfer across object counts

The sharpest object-centric claim: because the sparse model is a *per-object* gate +
residual head with weights **shared across objects** (no layer is sized to the object
count — verified in `models/sparse_gating.py` / `models/sparse_residual.py`), a model
trained on `N`-object scenes should work on `M`-object scenes with **zero retraining**.
The dense monolith cannot even be *run* off-count: its input and output layers are
sized to a specific count.

The one obstacle is featurization — the default per-object features append a flattened
all-object pose whose width grows with the count (24 / 30 / 39 for 3 / 5 / 8). We
replace that single term with a permutation-invariant neighbour aggregate (mean
relative position of other objects + nearest-neighbour relative position and distance),
giving a **fixed width of 20 for any count** (`--feature-mode invariant`,
`build_object_features_invariant`). We retrain the sparse models with it and evaluate
every (train N, test M) pair. `experiments/compositional_generalization.py`.

Change-detection F1, count-invariant sparse (rows = train count, cols = test count):

| train ↓ / test → | 3 | 5 | 8 |
|---|---|---|---|
| **3** | *0.846* | 0.793 | 0.832 |
| **5** | 0.846 | *0.801* | 0.839 |
| **8** | 0.850 | 0.801 | *0.848* |

*(italic = in-distribution diagonal)*

**Performance is set by the test count, not the training count.** Every row is nearly
identical: a model that never saw a 5-object scene scores 0.79–0.80 F1 on 5-object test
data — indistinguishable from the model trained on 5 objects. Off-diagonal (transfer)
mean F1 is **0.827 vs 0.832 on the diagonal → 99.4% retention**; changed-object L2 shows
the same column-uniformity (e.g. test-3: 0.313 / 0.312 / 0.307 regardless of train
count). **Dense transfers zero:** we *attempt* each off-diagonal dense evaluation and it
raises a dimension `RuntimeError` every time (recorded in `transfer_matrix.csv`) — the
diagonal is the only place a monolith can run at all. Figure/heatmap:
`experiments/runs/compositional_generalization/transfer_matrix.png`.

Two honest notes. (1) The count-invariant featurizer costs essentially nothing at the
native count — its diagonal F1 (0.846 / 0.801 / 0.848) is within noise of the original
count-specific `global` model (0.885 / 0.777 / 0.837), so transfer is bought for free,
not by degrading the base model. (2) The 8-object data uses a different table geometry
(wider bounds, tighter spacing), so any transfer touching 8 crosses a geometry shift as
well as a count shift; the **3↔5 transfer is the cleanest count-only evidence** (0.846
and 0.793, both ≈ their diagonals) and 8 corroborates it.

## Multi-step rollout (horizon error) — the world-model test

All numbers above score a *single* one-step prediction. The world-model question is what happens
when we **close the loop**: reconstruct the next state (`state + gate·delta` for sparse, absolute
poses for dense, unchanged for no-op), feed it back in, and repeat. Only the predicted object-pose
slice is rolled forward; exogenous state (pusher, velocities, goal) is taken from ground truth at
each step so all three models are driven identically. Rollouts launch from every timestep of every
episode; per-object L2 is aggregated per horizon (1→20) over all starts that reach it.
`experiments/rollout_horizon_error.py`.

Evaluated on a **strictly held-out trajectory set**: 250 fresh scripted episodes per count
generated with an unseen seed (100; training used 0/1/2), same geometry per count, verified to
share **zero** identical state rows with any training set. Mean per-object full-pose L2 at
**horizon 20**, **mean ± std over the 3 training seeds** (the no-op reference is
seed-independent):

| N | sparse | dense | no-op | sparse beats dense |
|---|---|---|---|---|
| 3 | 0.373 ± 0.033 | 1.241 ± 0.072 | 0.277 | ✓ (3.4×, every seed) |
| 5 | 0.293 ± 0.038 | 1.097 ± 0.096 | 0.180 | ✓ (3.8×, every seed) |
| 8 | 0.182 ± 0.006 | 1.170 ± 0.022 | 0.108 | ✓ (6.4×, every seed) |

Per-seed dense/sparse ratios: N=3 [3.98, 2.79, 3.33], N=5 [2.94, 4.64, 3.90], N=8 [6.74, 6.09, 6.46].
The earlier single-seed numbers, measured on the *full* `scale_{N}obj_s0.npz` datasets (the superset
the hard training subset was drawn from), were 0.360 / 0.358 / 0.164 sparse vs 1.290 / 0.967 / 1.181
dense — same ordering, so the conclusion never depended on the overlap; it is now **removed** rather
than disclosed.

**The structural prior compounds far less error.** The dense baseline perturbs every object's pose
every step, so error accumulates on the many objects that never moved and the rollout drifts. The
sparse gate copies unchanged objects verbatim, so its curve hugs the no-op reference (the true
"how much does the world move over H steps" floor) while still tracking the objects that do move —
it never blows up the way dense does. This is the clearest single demonstration that *modeling what
changes* is what makes the model usable as a world model, not just as a one-step regressor.

Per-horizon curves (full-pose + translation-only) per count and a combined overlay:
`experiments/runs/rollout_heldout_s{0,1,2}/rollout_horizon_{3,5,8}obj.png`,
`.../rollout_horizon_combined.png`; per-object breakdown in `rollout_per_object.csv`.

## Downstream planning (Phase 5) — the sparse model's advantage extends to control

Everything above scores the models as *predictors*. The decision-making test puts each model
**inside a planner**: sampling-based Model-Predictive Control (CEM, 256 samples × 3 refit iterations,
horizon 15, receding — replan every step) using the world model as the forward simulator, on the
tabletop **push-the-target-to-the-goal** task (3 objects, success = target within 0.05 m of goal).
The *only* component swapped between conditions is the model; planner, cost, seeds, and
imagined-state reconstruction are identical. `experiments/planning_mpc.py`. Two references bound the
result: an **oracle** planner uses the *true simulator* as its model (exact snapshot/rollout/rewind),
and **scripted** (the hand controller from data generation) / **random** are the model-free upper and
lower references.

### Round 1 — the prediction-trained models (Phases 2–4) cannot plan

| condition | forward model | success | mean steps | final target→goal (m) | plan ms/step |
|---|---|---|---|---|---|
| oracle | true simulator | **1.00** (10/10) | 11.5 | 0.0008 | 6907 |
| scripted | hand controller | **0.95** (19/20) | 28.6 | 0.0447 | 1.0 |
| random | uniform actions | 0.15 (3/20) | 21.7 | 0.2815 | 1.0 |
| sparse (global feats) | sparse/residual | **0.00** (0/20) | — | 0.3385 | 283 |
| dense | dense monolith | **0.00** (0/20) | — | 0.3261 | 223 |

**The planner is sound (oracle 1.00, ~11 steps) but the prediction-trained models are not
control-grade** — both fail completely, worse than random. The mechanism is exactly what the
one-step diagnostics predict, realised in closed loop:

- **Dense hallucinates motion.** With a zero action and no contact it still moves every object
  5–10 cm/step (the recall≈1.0 "everything changed" degeneracy). CEM plans against a fantasy.
- **Sparse is out-of-distribution under planning.** Its `global` features key change prediction on
  object *velocity* and the scripted policy's contact context. When CEM queries teleported, near-rest
  states from arbitrary approach angles the gate never fires and the delta head returns a near-constant
  drift *regardless of the pusher's position* — no usable gradient toward pushing. The failure is the
  **featurization + training distribution, not the sparse architecture**.

### Round 2 — fixing the diagnosed cause makes the sparse model plan, and beat dense

Two changes aimed straight at the Round-1 diagnosis, planner unchanged: (1) a **contact-aware,
velocity-free** feature mode (`--feature-mode contact`, `build_object_features_contact`) exposing the
pusher's *post-action* position, signed contact distance, and push direction — quantities well-defined
for any state a planner visits; (2) retrain on **diverse mixed-policy data** (scripted + random,
26 k transitions) covering the approach angles CEM samples. Dense is retrained on the same diverse
data (it consumes raw state and cannot use the per-object features).

Sparse/dense success is **mean ± std over 3 training seeds** (0/1/2, same data split, same 20 episode
configs); scripted/random are seed-independent (deterministic policies on fixed configs).

| condition | forward model | success | final target→goal (m) | plan ms/step |
|---|---|---|---|---|
| scripted | hand controller | **0.95** (19/20) | 0.045 | 1.0 |
| **sparse (contact)** | sparse/residual, contact feats + diverse data | **0.23 ± 0.06** (5/3/6 of 20) | **0.204 ± 0.019** | 215 |
| random | uniform actions | 0.15 (3/20) | 0.281 | 1.0 |
| **dense (mixed)** | dense monolith, diverse data | **0.00 ± 0.00** (0/20 every seed) | 0.318 ± 0.021 | 204 |

**The sparse world model now enables planning and the sparse-vs-dense gap is decisive and robust.**
Sparse goes **0.00 → 0.23 ± 0.06** success and cuts its final-distance error nearly in half
(0.34 → 0.20). The advantage over dense holds at **every seed**: dense is **0/20 at all three**
(diverse data does not cure the monolith's hallucination — the degeneracy is architectural), and the
final-distance error bars do not overlap (0.20 ± 0.02 vs 0.32 ± 0.02). Probing the retrained model
confirms the fix: the gate now fires (prob 0.7–0.94) when the pusher is positioned to push, and the
predicted target motion points *toward* the goal when the pusher is behind the object and *away* once
it overshoots — physically correct, geometry-dependent signal that was entirely absent in Round 1.
Many remaining sparse "failures" are near-misses (final distance ~0.09–0.16, i.e. pushed most of the
way) rather than wrong-way drift.

**Honest gaps.** (1) Sparse remains well below the scripted expert (0.23 vs 0.95): the delta head lands
*near* the goal but not reliably inside the 5 cm radius, and a few episodes still knock the object
away. (2) The margin over random (0.15) is real *on average* but not at every seed — the weakest
sparse seed (0.15) merely ties random, while the other two (0.25, 0.30) clear it; the *dense*
comparison is the robust one. This is a **promising** result (3 seeds, but one task/cost, 3 objects),
not a solved control benchmark. The concrete path to close the gap: a **distributional/heteroscedastic
contact delta head** (contact is multimodal; MSE averages it toward zero), **DAgger** on
planner-visited states, and **multi-step rollout training**. See "What it would take" below.

**Go/no-go decision (checklist Aug 3): KEEP, reframed.** Round 1 alone would have been a cut (both
0%); Round 2 makes it a genuine, if modest, positive: *the sparse/residual structure's advantage over
the dense monolith extends from prediction to control* (0.23 vs 0.00), once the model is featurized
and trained for the planner's state distribution. Recommended framing in the paper: report both rounds
— the honest negative and the diagnosed fix — as evidence that the object-centric prior is what carries
over to control, with the sim-to-expert gap stated plainly as future work.

**What it would take (future work).** Heteroscedastic/mixture delta head for multimodal contact;
DAgger loop (run MPC → label planner-visited states with the true sim → retrain — the oracle and env
`snapshot`/`restore` needed for this are already in the repo); multi-step rollout training so the model
is accurate over the horizon MPC actually uses; and planning in a learned latent rather than raw pose.

Reproduce:

```bash
# Round 1 — prediction-trained checkpoints + anchors:
python -m experiments.planning_mpc --conditions sparse dense scripted random \
  --num-episodes 20 --max-steps 60 --num-samples 256 --cem-iters 3 --horizon 15 \
  --run-name planning_models_3obj_v1
python -m experiments.planning_mpc --conditions oracle \
  --num-episodes 10 --max-steps 60 --num-samples 256 --cem-iters 3 --horizon 15 \
  --run-name planning_oracle_3obj_v1
# Round 2 — contact-aware sparse + diverse-data dense:
#   data: generate scripted + random, concat, split
python -m experiments.generate_transitions --policy scripted --episodes 200 --max-steps 80 \
  --num-objects 3 --seed 0 --output data/transitions/plan_scripted_3obj.npz
python -m experiments.generate_transitions --policy random --episodes 350 --max-steps 60 \
  --num-objects 3 --seed 1 --output data/transitions/plan_random_3obj.npz
python -m experiments.concat_transitions \
  --inputs data/transitions/plan_scripted_3obj.npz data/transitions/plan_random_3obj.npz \
  --output data/transitions/plan_mixed_3obj.npz
python -m experiments.split_dataset --input data/transitions/plan_mixed_3obj.npz \
  --output-dir data/transitions/splits_plan_mixed_3obj --seed 0
#   train (contact-aware sparse + diverse-data dense):
python -m experiments.train_sparse_model \
  --train data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_train.npz \
  --val   data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_val.npz \
  --run-name sparse_contact_3obj_v1 --feature-mode contact --epochs 25 \
  --sparsity-weight 0.05 --auto-balance-bce --seed 0
python -m experiments.train_dense_baseline \
  --train data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_train.npz \
  --val   data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_val.npz \
  --run-name dense_mixed_3obj_v1 --epochs 25 --hidden-dim 256 --num-layers 3 --seed 0
#   plan (seed 0 = _v1; repeat train+plan with --seed 1,2 and checkpoints _s1,_s2 for the 3-seed mean):
python -m experiments.planning_mpc --conditions sparse dense scripted random \
  --sparse-checkpoint models/checkpoints/sparse_contact_3obj_v1.pt \
  --dense-checkpoint  models/checkpoints/dense_mixed_3obj_v1.pt \
  --num-episodes 20 --max-steps 60 --num-samples 256 --cem-iters 3 --horizon 15 \
  --run-name planning_contact_3obj_v1
# Seeds 1,2 (same data split; sparse/dense only — scripted/random are seed-independent):
#   train_sparse_model ... --run-name sparse_contact_3obj_s{1,2} --seed {1,2}
#   train_dense_baseline ... --run-name dense_mixed_3obj_s{1,2} --seed {1,2}
#   planning_mpc --conditions sparse dense --sparse-checkpoint ..._s{1,2}.pt \
#     --dense-checkpoint ..._s{1,2}.pt --run-name planning_contact_3obj_s{1,2}
```

### ⚠⚠ W5's central claim is REFUTED: other learned models do plan, and one beats ours

*Found 2026-08-15. `experiments/runs/planning_literature_3obj/`. Held-out episode seeds
5000-5019, 20 episodes, 3 objects; identical planner, cost, and episode configurations for
every condition -- only the forward model changes. The five published baselines are trained on
the same planner-distribution (scripted + random) data as the sparse and dense conditions,
with the contact featurisation, and saved by `literature_baselines.py --checkpoint-dir`.*

| condition | forward model | success | final distance | plan ms/step |
|---|---|---|---|---|
| scripted | hand controller | **1.000** | 0.042 | 0.3 |
| **pets** | Chua et al. 2018 ensemble | **0.350** | **0.162** | 109 |
| **sparse (contact)** | ours | 0.250 | 0.156 | 68 |
| slotformer | Wu et al. 2023 | 0.150 | 0.263 | 110 |
| nps | Goyal et al. 2021 | 0.150 | 0.249 | 109 |
| gns | Sanchez-Gonzalez et al. 2020 | 0.100 | 0.309 | 213 |
| dense | monolith | 0.000 | 0.331 | 26 |
| latent | TD-MPC2-style | 0.000 | 0.327 | 59 |
| cswm | Kipf et al. 2020 | 0.000 | 0.314 | 80 |
| random | uniform actions | 0.000 | 0.316 | 0.2 |

**What this refutes.** The W5 section below states that "no learned model except the sparse
one plans at all" and that "the object-centric structured model is the **only** learned model
that plans at all (0.25 vs 0.00 for both a dense monolith and a latent-dynamics model)". Both
are false. **Four of the five published models plan above zero**, and **PETS reaches 0.350
against the sparse model's 0.250**.

The earlier claim was not wrong about its own evidence -- the dense monolith and the
latent-dynamics baseline really do score 0.000, and both reproduce here. It was wrong because
two baselines, both of which we wrote, were treated as standing in for "learned models" as a
class. The same error the momentum-shortcut finding exposed on the prediction side, repeated
on the control side.

**What survives, stated precisely.**

* The sparse model still beats the dense monolith decisively (0.250 vs 0.000), and the
  latent-dynamics baseline still fails at 39x the parameters. Those comparisons are intact.
* The oracle at 1.000 (below) still proves the planner and cost are sound, so every deficit
  here is model quality.
* What does **not** survive is any claim of the form "object-centric change modelling is what
  makes a world model usable for control". A probabilistic ensemble with no object-centric
  structure and no change gate plans better than our model does.

**One caution against over-reading the ordering.** At 20 episodes a success rate carries
roughly +/-0.10, so PETS-over-sparse (0.35 vs 0.25) is **within noise for a single
comparison** and rests on one training seed for the published baselines. The robust
statements are the qualitative ones -- four published models plan, three of the ten conditions
including ours cluster in the 0.10-0.35 band, and the sparse model is not distinguishable
from the best of them. Re-running the published baselines across three training seeds is the
obvious next step and is not yet done.

**Why PETS in particular is a plausible winner rather than a fluke.** It is the only baseline
trained by likelihood rather than squared error, and W1 already established on the prediction
side that a distributional delta head is what clears the no-op floor on contact-driven motion
(MSE +0.021 vs MDN +0.054 at N=8). Contact is multimodal; a squared-error model averages the
modes toward zero and gives a planner nothing to climb. That the same ingredient shows up as
the best planner is consistent with the prediction-side result rather than in tension with it
-- and it points at the honest reframing: **the ingredient that transfers to control is
probabilistic contact modelling, not the change gate.**

## Planning, properly tested (W5) — SUPERSEDED in part: see the refutation directly above

*Held-out episode seeds 5000–5019, disjoint from the 1000–1014 range DAgger collected on
(`planning_mpc`'s default base seed is 1000, so the default would have silently evaluated on
the training configurations). 20 episodes, 3 objects, seed 0. `experiments/dagger_planning.py`,
`models/latent_dynamics.py`.*

Two things were added to make this a real control result rather than a comparison against
non-learned anchors: a **DAgger loop** that labels planner-visited states with true
transitions via `snapshot`/`restore` (the correct form here labels the *transition*, not an
expert action, because the world model is what needs correcting), and a **latent dynamics
baseline** — the TD-MPC2/Dreamer core of encoder, latent transition and decoder, planned by
the *same* CEM with the same cost and seeds so only the representation varies.

| condition | success | mean final distance |
|---|---|---|
| scripted expert | 1.000 | 0.042 |
| **sparse (no DAgger)** | **0.250** | **0.120** |
| sparse (DAgger ×4 rounds) | 0.200 | 0.208 |
| dense monolith | 0.000 | 0.331 |
| **latent dynamics (274k params)** | **0.000** | 0.327 |
| random | 0.000 | 0.316 |

**1. DAgger does not help — it hurts.** Four rounds aggregating 38.5k simulator-labelled
transitions from planner-visited states left success at 0.20 against 0.25 without it, and
roughly doubled the final distance (0.120 → 0.208). Success at n=20 carries ±0.10, so that
difference alone is not significant; the distance is, and both point the same way. Collection
success across rounds (0.20 / 0.40 / 0.27 / 0.33) shows no trend either. The most likely
cause is that DAgger data ends up ~46% of the training set and is dominated by uniform
actions in states an *evolving* policy visited, so early rounds contribute transitions from a
worse policy — but that is a hypothesis, not something this experiment establishes.

**2. No learned model except the sparse one plans at all.** The latent baseline scores 0.000,
identical to the dense monolith and to random, despite carrying **274k parameters against the
sparse model's ~7k** — a 39× capacity advantage, trained on the same data and planned by the
same CEM. That makes the negative more credible, not less: this is not a case of an
undertrained straw man.

**3. Applying the pre-registered criterion: planning is demoted.** The bar set in advance was
~0.7 success on multiple tasks. The best learned model reaches 0.25 on one task. Planning
therefore becomes a short *"the prior transfers to control"* subsection, not a headline
section. What it can honestly claim:

  * the object-centric structured model is the **only** learned model that plans at all
    (0.25 vs 0.00 for both a dense monolith and a latent-dynamics model);
  * the oracle at 1.00 proves the planner and cost are sound, so the deficit is model quality;
  * and DAgger, the obvious fix, does not close it — which is worth reporting precisely
    because it is the first thing a reviewer would suggest.

That last point is the useful contribution here. "Sparse beats dense" was already known; "a
TD-MPC2-style latent model with 39× the capacity also fails, and DAgger does not rescue
either" is a substantially more informative negative, and it makes the case that the gap is
about *what these models get wrong at contact*, not about training distribution.

## Qualitative figures

Predicted diff vs ground-truth diff, 2 examples per count:
`experiments/runs/phase4_{3obj_s0,5obj_s0,8obj_s0}/qualitative_example_{0,1}.png`.

## Caveats to state in the paper

*Substantially revised 2026-08-08. Several entries below previously described problems that
are now **fixed**, and two described solutions that turned out **not to reproduce**. Read
this list rather than the older prose above it wherever the two disagree.*

- **⚠ SUPERSEDED — the splits leak.** Every number computed on `splits_{N}obj_s{S}` has train
  and test states from the same trajectory (25% of source episodes span both). Quote only
  figures recomputed on `splits_clean_*`. The ordering survives (sparse still beats dense at
  every count and seed) but individual values do not — the sparse model's own error rises up
  to 1.30× once memorised rows are removed. See the boxed section at the top.
- **✅ FIXED — 8-object geometry.** The published series switched geometry at N=8. The W4
  series uses bounds ±0.26 / separation 0.09 for **every** count from 3 to 20, so N is now the
  only variable. Use that series for any cross-N claim.
- **Ungated change-detection is degenerate** (recall = 1.000): every model without a gate
  flags everything as changed, with detection metrics identical to four decimal places across
  dense, dense+L1, both object-centric ablations **and the set transformer**, with the
  interaction network within 0.001. Relational modelling does not cure it — only gating does.
  This is now the stronger, better-controlled form of the claim.
- **✅ FIXED — single-seed analyses, and all four survived.** The gate ablation is 3-seed
  (with four new rungs), the delta-head study 3-seed, the W4 scale series 5-seed. The four
  remaining tables were re-run on clean splits across 3 seeds
  (`experiments/rerun_single_seed_tables.sh`) and **every published conclusion held**:

  | table | published | clean, 3 seeds |
  |---|---|---|
  | cross-count transfer retention | 99.4% | **99.5% ± 0.4%** (dense still transfers never) |
  | sample efficiency at 25% data | ~90% of full-data F1 | **94%** (0.804 vs 0.858) |
  | dense F1 vs data budget | flat | **flat**, 0.539 at every budget |
  | parameter-matched dense | shrinking dense does not help | **worse** than full dense at every N |

  One correction the reruns produced: the oracle-gate diagnostic at N=8 gives
  **+0.0131 ± 0.0218** margin over no-op across seeds, i.e. statistically indistinguishable
  from zero — *not* the negative value seed 0 alone reports (−0.0165). The motivation for W1
  stands (a perfect mask buys essentially nothing) but "worse than no-op" was a single-seed
  artifact and must not be written that way.

  Still seed-0-only: the W1 capacity and K-sweep controls, and all of W3.
- **⚠ WORSE THAN REPORTED — overall-L2 saturation.** Not a tie at N=8 but a **crossover**: on
  clean splits sparse loses to no-op at N=12 (0.0663 vs 0.0653) and N=20 (0.0447 vs 0.0423),
  and every ungated model loses there too. The dense-interaction control that previously
  rescued this **does not reproduce on clean splits** (+0.0011 ± 0.0033 at N=8, not positive
  at every seed). Report overall per-object L2 as unusable at high object count and lead with
  change-detection F1 and unchanged-object L2, where the advantage instead *grows* with N.
- **The mechanism ratio is a bad statistic.** The dense/sparse unchanged-L2 *ratio* used in
  the published tables is noise-dominated — its denominator is the gate's false-positive rate
  (~0.001) — and is non-monotone in N (249/204/332/449/200). The *difference* is monotone
  across all five counts. Quote the gap.
- **Planning is preliminary, scoped to 3 objects / one cost.** The prediction-trained models plan at
  0% (Round 1); with contact-aware features + diverse data the sparse model reaches **0.23 ± 0.06**
  (3 seeds) and beats dense (**0.00** at every seed) and random (0.15 on average) (Round 2), while the
  oracle confirms the planner is sound (1.00). Sparse is still far below the scripted expert (0.95),
  the margin over random is not significant at the weakest seed, and a few episodes knock the object
  away — report it as a *promising directional* result (the object-centric advantage extends to
  control), not a solved control benchmark. One CEM cost (target→goal + small proximity shaping); no
  cost/object-count sweep.

## Reproduce

```bash
# One (object-count, seed) pipeline end-to-end:
bash experiments/scale_pipeline.sh <N> <SEED> [OBJECT_BOUND] [MIN_SEP]
# All 3 seeds for a count (8 obj needs geometry args):
bash experiments/run_count_seeds.sh 3
bash experiments/run_count_seeds.sh 5
bash experiments/run_count_seeds.sh 8 0.22 0.09
# Dense-interaction variant (packed objects; VARIANT tag keeps runs separate):
VARIANT=dense bash experiments/run_count_seeds.sh 3 0.10 0.08
VARIANT=dense bash experiments/run_count_seeds.sh 5 0.13 0.085
VARIANT=dense bash experiments/run_count_seeds.sh 8 0.16 0.08
# Aggregate to paper tables (main + dense variant):
python experiments/aggregate_seeds.py
python experiments/aggregate_seeds.py --variant dense
# Sparsity ablation:
bash experiments/run_sparsity_ablation.sh
# Parameter-matched dense baseline (defensibility) across counts:
python experiments/param_matched_baseline.py --counts 3 5 8
# Gate ablation (object-centricity vs change gate), capacity-matched ladder + robustness check:
python experiments/gate_ablation.py --counts 3 5 8
python experiments/gate_ablation.py --counts 3 5 8 --width-mode identical \
  --run-name gate_ablation_identical
# Oracle-gate diagnostic (detection vs regression bottleneck):
python experiments/oracle_gate_diagnostic.py --counts 3 5 8
# Sample-efficiency sweep (10/25/50/100% of train) per object count:
bash experiments/run_sample_efficiency.sh 3
bash experiments/run_sample_efficiency.sh 5
bash experiments/run_sample_efficiency.sh 8
# Compositional generalization: cross-object-count transfer matrix (sparse vs dense):
python experiments/compositional_generalization.py --counts 3 5 8
# Multi-step rollout / horizon error (world-model test): held-out trajectories x 3 seeds.
#   1) held-out eval sets (generation seed 100, unseen at training):
python -m experiments.generate_transitions --policy scripted --episodes 250 --max-steps 100 \
  --num-objects 3 --seed 100 --output data/transitions/scale_3obj_heldout.npz
python -m experiments.generate_transitions --policy scripted --episodes 250 --max-steps 100 \
  --num-objects 5 --seed 100 --output data/transitions/scale_5obj_heldout.npz
python -m experiments.generate_transitions --policy scripted --episodes 250 --max-steps 100 \
  --num-objects 8 --seed 100 --object-bound 0.22 --min-object-separation 0.09 \
  --output data/transitions/scale_8obj_heldout.npz
#   2) roll out each training seed's checkpoints on them:
for S in 0 1 2; do
  python -m experiments.rollout_horizon_error \
    --manifest experiments/rollout_manifest_heldout_s$S.json --run-name rollout_heldout_s$S
done
# Downstream planning (Phase 5): MPC with each model as forward simulator + oracle/scripted/random.
python -m experiments.planning_mpc --conditions sparse dense scripted random \
  --num-episodes 20 --max-steps 60 --num-samples 256 --cem-iters 3 --horizon 15 \
  --run-name planning_models_3obj_v1
python -m experiments.planning_mpc --conditions oracle \
  --num-episodes 10 --max-steps 60 --num-samples 256 --cem-iters 3 --horizon 15 \
  --run-name planning_oracle_3obj_v1
```

Checkpoints: `models/checkpoints/{dense,sparse}_{3,5,8}obj[_dense]_s{0,1,2}.pt`.
Per-run artifacts: `experiments/runs/phase4_{3,5,8}obj[_dense]_s{0,1,2}/`.
