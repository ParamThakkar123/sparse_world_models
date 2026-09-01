# Phase 4 — Object-Count Scaling (3 → 5 → 8)

> **Superseded by `experiments/RESULTS.md`** — this doc reports the original **single-seed
> (seed 0)** exploration. The paper-ready numbers are the **3-seed mean ± std** in `RESULTS.md`
> and `paper_tables/`. The single-seed values below match the multi-seed means and the narrative
> still holds; kept for the detailed per-metric discussion.

Sparse/residual vs dense vs no-op, evaluated on the held-out **hard** test split at each
object count. All runs use the identical recipe (250 scripted episodes × 100 steps, hard
subset with `min-max-xy-delta 0.02`, 80/10/10 split with the configuration-leakage guard,
dense 25ep / sparse 15ep `sparsity-weight 0.2` `auto-balance-bce`, seed 0). State dimensions
auto-infer from the data, so no per-count model retuning.

> **Geometry caveat (8 objects only):** 8 objects do not fit the default 0.36×0.36 m
> placement area at 0.12 m separation, so the 8-obj data was generated with a wider bound
> (±0.22 m) and tighter separation (0.09 m) — still far above the 5 cm box size, no physical
> overlaps. The *within-count* sparse/dense/no-op comparison is unaffected; the 8-obj density
> point is not a perfectly controlled continuation of the 3→5 trend. 3- and 5-obj data use the
> original geometry unchanged.

## Combined results

| N | changed-obj frac | hard transitions | sparse L2 | no-op L2 | **margin vs no-op** | dense L2 | sparse F1 | precision | recall | sparse/dense params | sparse/dense FLOPs | sparse lat (ms) | dense lat (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.369 | 1,421 | 0.1157 | 0.1283 | **−0.0126** | 0.3671 | 0.885 | 0.965 | 0.817 | 6.9k / 76.8k (11.1×) | 40k / 153k (3.8×) | 0.21 | 0.09 |
| 5 | 0.240 | 1,474 | 0.1038 | 0.1074 | **−0.0035** | 0.3180 | 0.777 | 0.904 | 0.681 | 8.5k / 83.0k (9.8×) | 82k / 165k (2.0×) | 1.02 | 0.30 |
| 8 | 0.176 | 1,596 | 0.0807 | 0.0811 | **−0.0004** | 0.3540 | 0.837 | 0.974 | 0.734 | 10.8k / 92.2k (8.6×) | 168k / 183k (1.1×) | 0.52 | 0.08 |

`no_op_trivially_wins: false` at all three counts (sparse overall L2 < no-op at every N).

## Key findings

1. **Sparse beats dense decisively and consistently.** Overall per-object L2 is ~4× lower than
   dense at every object count (0.08–0.12 vs 0.32–0.37). The dense MLP regresses *every* object's
   pose and is dominated by the many unchanged objects it perturbs (dense unchanged-L2 grows
   0.21 → 0.28 with N). This is the core claim and it holds across scale.

2. **The no-op margin collapses as scenes get sparser** (−0.0126 → −0.0035 → −0.0004). As N
   grows, a larger fraction of objects genuinely don't move (changed-obj fraction 0.37 → 0.18),
   so "predict no change" gets *stronger* on overall L2. At 8 objects sparse only edges out
   no-op on overall L2 (0.0807 vs 0.0811). **Overall per-object L2 is a weak headline metric for
   sparse scenes** — it is dominated by unchanged objects and flatters the no-op baseline.

3. **The durable advantage over no-op is change detection, not regression.** Sparse holds
   F1 0.78–0.89 (precision 0.90–0.97) across all counts, while no-op is 0 by construction. On
   *changed*-object L2 alone, sparse barely beats no-op at 8 obj (0.466 vs 0.470) — the delta
   head's regression on moved objects is weak in absolute terms; its win is knowing *which*
   objects moved. **Recommendation: foreground change-detection F1 + changed-object L2 as the
   primary metrics; treat overall L2 as secondary.**

4. **The efficiency win is parameters, not FLOPs or latency.** Param advantage persists
   (8.6–11× fewer than dense) but the FLOP advantage erodes with N (3.8× → 1.1×), because the
   per-object gate+delta heads scale with object count while the dense MLP is fixed-width. Sparse
   wall-clock latency is actually *higher* than dense (per-object gating overhead vs one big
   matmul). **Frame the efficiency claim around parameter count / capacity, not inference speed.**

## Implications for the writeup

- Report change-detection F1 and changed-object L2 as primary; note overall-L2 degeneracy in sparse regimes.
- Either report on the hard subset (done) or generate denser-interaction data (more objects moved
  per step) so the regression signal doesn't vanish at high N.
- Reframe the efficiency story as parameter efficiency; drop or heavily qualify latency claims.

## Artifacts

- 3 obj: `experiments/runs/phase4_comparison_hard_v1/`
- 5 obj: `experiments/runs/phase4_comparison_5obj_v1/`
- 8 obj: `experiments/runs/phase4_comparison_8obj_v1/`

Each contains `results_table.{md,csv}`, `detailed_results.json`, `summary.json`, and two
`qualitative_example_*.png` (predicted diff vs ground-truth diff). Checkpoints:
`models/checkpoints/{dense_baseline,sparse_model}_{5,8}obj_v1.pt`.
