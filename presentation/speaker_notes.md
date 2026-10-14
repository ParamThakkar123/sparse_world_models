# Speaker script: Modeling What Changes (WORLDS @ IROS 2026)

Generated from the notes pane of `WORLDS_IROS2026_talk.pptx` by `build_deck.py`. Slides 1-13 are the talk (about 10 minutes), 14 is an optional postscript, 15-16 are backup.

## 1. Title

Good afternoon. I'm going to talk about a very simple idea for world models of manipulation
scenes: model what changes, and copy everything else.

(Title, authors and venue. Repository: https://github.com/ParamThakkar123/sparse_world_models)

## 2. Physical scenes are sparse in change

Start with the observation. When a robot nudges one block on a cluttered table, exactly one
pose changes. The rest of the scene is not approximately the same; it is identical.

The dominant world-model recipe ignores that. It encodes the scene and regresses the whole
next state, monolithically, at every step. So it spends its capacity reproducing objects that
did nothing.

## 3. Re-predicting everything has three costs

Three costs. Capacity is the obvious one. Error injection is the insidious one: a regressor
that has to output every pose will inevitably perturb the objects it should have left alone,
and in a rollout that error compounds precisely on the objects that never moved. And third,
interpretability: a monolithic model has no explicit belief about what will change.

We ask whether explicitly modelling what changes addresses all three.

## 4. Gate what changes, copy the rest

The model is deliberately simple and object-centric. For each object we build a feature
vector from its pose, velocity, relative goal and pusher positions, the action, and a
permutation-invariant aggregate over the other objects.

A gate network emits a change decision, zero or one. A delta head emits a pose residual in
R^3. The next pose is the current pose plus the masked residual. If the gate is off, the
object is copied verbatim: not approximately, exactly.

The gate is discrete, trained with a Gumbel straight-through estimator, so it is hard at
inference. And because the heads are shared and nothing is sized to the object count, a
model trained on 3 objects runs on 8 with no retraining.

## 5. One loss, three terms

Three terms. A class-balanced BCE on the gate against the ground-truth changed mask. An L2 on
the residual, supervised only where something actually changed. And a sparsity penalty on the
mean gate activation.

The sparsity weight is a precision-recall knob. A five-point sweep moves precision from 0.94
to 0.97 and recall from 0.81 to 0.71; F1 degrades only gently. We use 0.2 for prediction.
(Appendix D of the paper.)

Everything is tiny: two-layer, width-128 heads, under 0.1 million parameters.

## 6. A controlled tabletop pushing benchmark

The setup is deliberately controlled and I want to state its bounds up front: structured
per-object state, not pixels; three to eight rigid boxes; small MLPs.

Two baselines. A dense MLP that regresses every pose. And no-op, which predicts that nothing
moves. No-op matters: on a scene that is mostly static it is a strong floor, and I will come
back to it.

The metric that matters most today is per-object pose L2 split into changed and unchanged
objects. That split is what lets us say where a win comes from.

## 7. Lower error at a fraction of the parameters

Here is the headline comparison, Table I of the paper. At every object count the sparse
model's overall L2 is 2.5 to 4.6 times lower than the dense MLP, with 8.6 to 11.1 times fewer
parameters. Change detection holds F1 0.80 to 0.87 at precision 0.92 to 0.97, where the dense
model is degenerate: it says everything moves.

Now look at the blue bars. That is no-op, and sparse essentially matches it. Hold that
thought; it is the most important thing on this slide.

Efficiency caveat, stated honestly: the win is parameters, not wall-clock. Per-object heads
scale with N, and the dense model's single matmul is faster in latency (Appendix C).

## 8. Sparse moves only what moves

Qualitatively this is what it looks like. Two held-out scenes, arrows are predicted
displacement. Ground truth on the left: one object moves. Sparse in the middle: the gate fires
on that object and only that object. Dense on the right: it hallucinates motion on all three.

That is error injection, and it is what the gate removes.

## 9. Where the win actually comes from

This is the slide I care most about. Why does it win?

First, it is not just smaller: a dense MLP shrunk to the same parameter budget is still 3 to
5 times worse (Appendix E). Second, the ladder on the left adds one ingredient at a time:
object-centric featurization, then an always-on residual, then the gate. Featurization earns a
third to a half of the gap, residual a bit more, but the gate is the largest single step and
the only rung that gets below no-op. The unchanged-object column shows the mechanism: error on
static objects goes from 0.211 to 0.001.

Now the table on the right, our oracle-gate diagnostic (Appendix F). If we give the delta head
the ground-truth changed mask, changed-object error barely moves. And it sits right at the
no-op floor. So, stated plainly: the win is not accurate dynamics on movers. It is change
detection that avoids injecting error into everything else. One-step contact deltas,
especially rotation, are simply hard, and the delta head is the highest-leverage thing to fix.

One missing rung we flag in the paper: a dense, non-object-centric model predicting residuals.
That is immediate future work.

## 10. Rollout, transfer across counts, sample efficiency

Three world-model properties, Figure 3 of the paper.

Rollout: close the loop for twenty steps. Dense drifts, sparse compounds far less error, 3.4 to
6.4 times less at horizon 20 (Table III). But be honest with the blue line: sparse is above
no-op at every horizon. The decomposition says why: sparse wins on unchanged objects and only
ties no-op on movers, consistent with the one-step result.

Transfer: because nothing is sized to N, a model trained on 3 objects runs on 8 with zero
retraining. Off-diagonal F1 retains 99.4 percent of the diagonal. The dense model cannot even
be executed off-count; its input dimension is wrong.

Sample efficiency: sparse reaches about 90 percent of full-data F1 with a quarter of the data.
(b and c are seed 0.)

## 11. Can it plan? Separated from dense, not yet from random

Does any of this help control? Each model is the forward simulator inside the same CEM
planner on a push-to-goal task; the model is the only thing that changes.

Round 1 was a clean negative result. The oracle solves every episode, so the planner and task
are fine. Both learned models score zero, worse than random. The cause is out-of-distribution
querying: the planner visits near-rest, teleported states the models never saw. Dense
hallucinates motion with no contact; sparse's velocity features leave its gate silent.

Round 2 fixes the diagnosed cause: contact-aware, velocity-free features and diverse data. With
the planner unchanged, sparse goes from zero to 0.23 and halves its final distance, while dense
stays at zero at every seed.

But I will not oversell it. 0.23 against random's 0.15 is not significant; the weakest seed
is exactly 0.15. The robust claim is separation from dense at every seed, not that sparse
"begins to plan".

## 12. What the headline is, and what it is not

Let me put both halves side by side, because the paper does.

It is a change detector that avoids error injection. That is a real and durable win over
dense, one-step, in rollout, and in planning, at a tenth of the parameters, with a mask you
can inspect.

It is not an accurate predictor of the objects that move. On those, it is at no-op quality.
Overall it matches no-op; in rollout it is worse than no-op; in planning it is not
significantly better than random. Phrases like "predicts dynamics far more accurately" invite
a reading the data do not support, and we retired them from the paper.

Also: F1 is partly a property of the metric (thresholding a dense regressor forces recall to
1), so we foreground the changed / unchanged L2 split instead. And the ladder is missing a
dense-residual rung, which confounds residual against object-centricity.

## 13. Takeaways

Three takeaways. Model what changes: the gate plus masked residual is a one-line bias that
stops error injection. Decompose the metric: the changed / unchanged split and the no-op
column are what turned a headline into an honest claim. And the delta head is the lever:
detection is solved, regression of contact deltas is not.

Code, data generators, checkpoints and machine-readable tables are at https://github.com/ParamThakkar123/sparse_world_models. The QR code
also reaches an in-browser demo that runs the real checkpoints on live physics.

Thank you.

## 14. Since the camera-ready: run the trivial baselines

OPTIONAL SLIDE. Use it if you want to end on the strongest possible note of scientific
honesty; drop it if the talk is short or you prefer to keep the follow-up for its own venue.

After the camera-ready we asked a question we should have asked earlier: what does a trivial
rule score on our own benchmark? "The object is moving, or it is near the pusher": one
parameter, F1 0.927. Our learned gate: 0.823.

The reason is structural. With a single point-like pusher the changed label is recoverable
from one scalar, so the benchmark is degenerate. Filtering harder only changes which one-liner
wins. The error-suppression mechanism in this paper still holds; what does not hold is any
claim that a learned gate is needed to get it on this benchmark. Our follow-up is a
trivial-baseline audit battery, and our recommendation to this community is: run it.

(Numbers: README / experiments/RESULTS.md, motion benchmark, best trivial rule vs best learned gate.)

## 15. Backup: efficiency in full

Backup. Appendix C, Table V: parameters, FLOP estimate and CPU latency (Intel i7-12650H).

## 16. Backup: sparsity weight and per-seed planning

Backup. Appendix D (Table VI), Appendix G (Table IX, 20 episodes per seed) and Appendix E (Table VII).
