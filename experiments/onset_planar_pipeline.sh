#!/usr/bin/env bash
# Rebuild the PLANAR onset benchmark at adequate statistical power.
#
# Why this exists. The published planar cross-environment replication ran on 1500 episodes
# and yielded 102 train / 9 val / 10 test rows at N=3 -- against the tabletop onset
# benchmark's 752 / 90 / 80. With 10 test rows a single misclassification moves F1 by ~0.1,
# so the direction of the effect was visible but no individual number was trustworthy.
# RESULTS.md flagged it as "indicative only" and prescribed this fix.
#
# Onset events are far rarer in the quasi-static planar domain than on the impulsive
# tabletop -- 0.15% of steps against 3.0% -- because objects there move only while in
# contact, so a rest->moving transition that also clears the displacement threshold is
# uncommon and the scripted policy finishes episodes quickly. The fix is more episodes, NOT
# a lower displacement threshold: lowering the threshold would change the task definition
# and make the two environments' benchmarks non-comparable, which is exactly what the
# cross-environment claim needs to avoid.
#
# 12000 episodes (8x) lands the planar onset training split near the tabletop's ~750 rows.
# Seeds are raised 3 -> 5 at the same time, because the whole point of this rerun is
# statistical power and generation here is cheap (~50 episodes/s).
#
# Usage: bash experiments/onset_planar_pipeline.sh [COUNTS] [SEEDS]
set -euo pipefail

COUNTS="${1:-3 5 8}"
SEEDS="${2:-0 1 2 3 4}"
BOUND=0.26
SEP=0.09
EPISODES=12000
MAX_STEPS=100
# Tuned for the planar domain's smaller per-step displacements; see planar_push.py's
# docstring. Deliberately identical to the value the motion-filtered planar runs used, so
# onset and motion benchmarks differ in FILTER only.
MIN_DELTA=0.010

echo "############ GENERATE (${EPISODES} episodes; planar onset events are ~0.15% of steps) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    OUT="data/transitions/onsetplanar12k_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    echo "  generating N=${N} seed=${S}"
    python -m experiments.generate_transitions --policy scripted \
      --episodes "$EPISODES" --max-steps "$MAX_STEPS" --num-objects "$N" --seed "$S" \
      --object-bound "$BOUND" --min-object-separation "$SEP" --env planar \
      --output "$OUT" --run-name "onsetplanar12k_${N}obj_s${S}"
  done
done

echo "############ ONSET SPLITS (episode-disjoint) ############"
python -m experiments.build_clean_splits --counts $COUNTS --seeds $SEEDS \
  --filter-mode onset --min-max-xy-delta "$MIN_DELTA" \
  --input-template "data/transitions/onsetplanar12k_{n}obj_s{seed}.npz" \
  --output-template "data/transitions/splits_onsetplanar12k_{n}obj_s{seed}"

echo "############ TRAIN (global + contact featurisation + dense) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    D="data/transitions/splits_onsetplanar12k_${N}obj_s${S}"
    ST="onsetplanar12k_${N}obj_s${S}_hard"
    python -m experiments.train_dense_baseline \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "dense_onsetplanar12k_${N}obj_s${S}" --epochs 25 \
      --hidden-dim 256 --num-layers 3 --seed "$S"
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_onsetplanar12k_${N}obj_s${S}" --epochs 15 \
      --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_onsetplanar12k_contact_${N}obj_s${S}" --feature-mode contact \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ ONSET_PLANAR_DONE ############"
