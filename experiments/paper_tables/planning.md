<!-- Phase 5 downstream planning: 3-object tabletop push, target -> goal, success radius 0.05 m.
     Identical CEM-MPC (256 samples x 3 iters, horizon 15, receding) for every planner row; the
     only swapped component is the forward model. Model/scripted/random over 20 episodes
     (seeds 1000-1019); oracle over 10 episodes (seeds 1000-1009). Reproduce: RESULTS.md
     (## Downstream planning). Sources: experiments/runs/planning_{models,oracle,contact}_3obj_v1/. -->

# Downstream planning (Phase 5) — MPC with each world model as the forward simulator

## Round 1 — the models trained for prediction (Phases 2–4) cannot plan

Checkpoints trained on the scripted **hard subset** with the pose/velocity (`global`) features.

| condition | forward model | success | mean steps (success) | final target→goal (m) | plan ms/step |
|---|---|---|---|---|---|
| oracle    | true simulator               | **1.00** | 11.5 | 0.0008 | 6907 |
| scripted  | none (hand controller)       | **0.95** | 28.6 | 0.0447 | 1.0 |
| random    | none (uniform actions)       | 0.15 | 21.7 | 0.2815 | 1.0 |
| sparse    | sparse/residual (global feats) | **0.00** | — | 0.3385 | 283 |
| dense     | dense monolith               | **0.00** | — | 0.3261 | 223 |

The identical planner solves the task with a perfect model (oracle 1.00) and the task is solvable
(scripted 0.95), yet **both learned models fail completely** — worse than random. Diagnosis: on the
states a planner queries (arbitrary approach angles, near-rest, teleported), the models are
out-of-distribution — the dense model hallucinates 5–10 cm of motion with no contact, and the sparse
gate never fires / the delta head returns a near-constant drift regardless of the pusher. The failure
is *featurization + data distribution*, not the sparse architecture.

## Round 2 — fixing the diagnosed cause makes the sparse model plan (and beat dense)

Same planner, same seeds. Two changes, both aimed straight at the Round-1 diagnosis: (1) a
**contact-aware, velocity-free** per-object feature mode (`--feature-mode contact`) that exposes the
pusher's *post-action* position, signed contact distance, and push direction — well-defined for any
state; (2) retrain on **diverse mixed-policy data** (scripted + random) covering the approach angles
CEM samples. Dense is retrained on the same diverse data (it takes raw state, so it cannot use the
contact features).

Sparse/dense are **mean ± std over 3 training seeds** (0/1/2, same data split, same 20 episode
configs); scripted/random are seed-independent.

| condition | forward model | success | final target→goal (m) | plan ms/step |
|---|---|---|---|---|
| scripted           | none (hand controller)          | **0.95** | 0.045 | 1.0 |
| **sparse (contact)** | **sparse/residual, contact feats + diverse data** | **0.23 ± 0.06** (5/3/6 of 20) | **0.204 ± 0.019** | 215 |
| random             | none (uniform actions)          | 0.15 | 0.281 | 1.0 |
| **dense (mixed)**  | dense monolith, diverse data    | **0.00 ± 0.00** (0/20 every seed) | 0.318 ± 0.021 | 204 |

**The sparse world model now enables planning, and the sparse-vs-dense gap is decisive and robust.**
Sparse goes 0.00 → **0.23 ± 0.06** success and cuts its final-distance error nearly in half
(0.34 → 0.20). The advantage over dense holds at **every seed** (dense 0/20 all three; the
final-distance error bars do not overlap): diverse data does not cure the monolith's hallucination —
it is architectural. Sparse beats random on average, though its weakest seed (0.15) ties random.
Many sparse "failures" are near-misses that pushed the target most of the way to the goal
(final distance ~0.09–0.16) rather than the wrong-way drift of Round 1. It remains well below the
scripted expert (0.95): the delta head lands *near* the goal but not reliably inside the 5 cm radius,
and a few episodes still knock the object away. Closing that gap is the future work in RESULTS.md
(distributional/contact delta head, DAgger on planner-visited states, multi-step rollout training).

Sources: `experiments/runs/planning_{models,oracle}_3obj_v1` (Round 1),
`experiments/runs/planning_contact_3obj_{v1,s1,s2}` (Round 2, seeds 0/1/2). Checkpoints:
`models/checkpoints/sparse_contact_3obj_{v1,s1,s2}.pt`, `models/checkpoints/dense_mixed_3obj_{v1,s1,s2}.pt`.
