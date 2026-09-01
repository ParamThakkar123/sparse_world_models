#!/usr/bin/env bash
# Close the coverage gaps RESULTS.md lists as outstanding.
#
# These are not new claims. Every one is an existing result that rests on fewer seeds or
# fewer object counts than the conclusion drawn from it needs, and each is listed in
# RESULTS.md's own caveats. They are grouped here so the whole set can be run and checked
# off together rather than rediscovered one reviewer at a time.
#
# GAP 1 -- W3 efficiency stage covers only N=3.
#   The counterfactual-augmentation *validity* stage runs at 3/5/8 x 3 seeds, but the stage
#   that shows the augmentation actually helps ("does the synthesized data improve the
#   model?") was only ever run at 3 objects. The claim it supports -- that a causally-blind
#   splice is worse than not augmenting at all -- is object-count-sensitive on its face,
#   since more objects means more ways for a blind placement to change the dynamics. The
#   validity stage already shows that advantage GROWING with count (+0.082 / +0.138 /
#   +0.174), so the efficiency stage should be measured where that gradient is largest.
#
# GAP 2 -- the W3 augmentation-ratio sweep is seed 0 only.
#   The dose-response result (gate-spliced data improves monotonically with volume while
#   blind splicing stays flat) is the cleanest evidence that the mask supplies validity
#   rather than volume. It currently rests on one seed, where "monotone across three points"
#   can happen by chance.
#
# GAP 3 -- the W4 onset scaling series is unusable for F1.
#   That series was onset-FILTERED from existing motion-benchmark data rather than generated
#   at the onset benchmark's episode count, leaving 110-237 training rows per cell against
#   the dedicated benchmark's 752. RESULTS.md marks its F1 numbers as "must not be used"
#   (sparse loses to dense at N=3 and N=8, contradicting the dedicated benchmark) while
#   keeping its unchanged-L2 scaling trend, which is a large enough effect to survive the
#   sample size. Regenerating at full episode count is the fix RESULTS.md itself prescribes;
#   until then the scaling claim has to be reported with a caveat that swallows it.
#
# Usage: bash experiments/close_coverage_gaps.sh
set -euo pipefail

# Small thread cap: this runs alongside the cross-domain and planar pipelines, and these are
# tiny MLPs where extra threads buy nothing but do steal cores from the generation jobs.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "############ GAP 1: W3 efficiency at N=5 and N=8 ############"
for N in 5 8; do
  # The 8-object clean splits were generated at the wider bounds; passing the env default
  # here makes placement of 8 boxes impossible and the validity replay dies at reset.
  if [[ "$N" == "8" ]]; then BOUND=0.22; SEP=0.09; else BOUND=0.26; SEP=0.09; fi
  for S in 0 1 2; do
    python -m experiments.counterfactual_augmentation \
      --count "$N" --seed "$S" --stages efficiency \
      --object-bound "$BOUND" --min-object-separation "$SEP" \
      --run-name "w3_efficiency_${N}obj_s${S}"
  done
done

echo "############ GAP 2: W3 augmentation-ratio sweep at seeds 1 and 2 ############"
for S in 1 2; do
  for RATIO in 0.5 1.0 2.0; do
    python -m experiments.counterfactual_augmentation \
      --count 3 --seed "$S" --stages efficiency --augment-ratio "$RATIO" --budgets 1.0 \
      --object-bound 0.26 --min-object-separation 0.09 \
      --run-name "w3_ratio_${RATIO}_3obj_s${S}"
  done
done

echo "############ GAP 3: regenerate the onset scaling series at full episode count ############"
# Same episode count as the dedicated onset benchmark (1500), at every count in the W4
# series, so the onset F1 numbers become usable instead of sample-size-limited. Bounds and
# separation are held identical across ALL counts -- that is the whole point of the W4 series
# and the fix for the published series' geometry switch at N=8.
BOUND=0.26
SEP=0.09
EPISODES=1500
for N in 3 5 8 12 20; do
  for S in 0 1 2 3 4; do
    OUT="data/transitions/onsetscale_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT"; continue; fi
    echo "  gen N=${N} s=${S}"
    python -m experiments.generate_transitions --policy scripted \
      --episodes "$EPISODES" --max-steps 100 --num-objects "$N" --seed "$S" \
      --object-bound "$BOUND" --min-object-separation "$SEP" \
      --output "$OUT" --run-name "onsetscale_${N}obj_s${S}"
  done
done

python -m experiments.build_clean_splits --counts 3 5 8 12 20 --seeds 0 1 2 3 4 \
  --filter-mode onset --min-max-xy-delta 0.020 \
  --input-template "data/transitions/onsetscale_{n}obj_s{seed}.npz" \
  --output-template "data/transitions/splits_onsetscale_{n}obj_s{seed}"

for N in 3 5 8 12 20; do
  for S in 0 1 2 3 4; do
    D="data/transitions/splits_onsetscale_${N}obj_s${S}"; ST="onsetscale_${N}obj_s${S}_hard"
    [[ -f "$D/${ST}_train.npz" ]] || { echo "  MISSING $D/${ST}_train.npz"; continue; }
    python -m experiments.train_dense_baseline \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "dense_onsetscale_${N}obj_s${S}" --epochs 25 \
      --hidden-dim 256 --num-layers 3 --seed "$S"
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_onsetscale_${N}obj_s${S}" --epochs 15 \
      --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ COVERAGE_GAPS_DONE ############"
