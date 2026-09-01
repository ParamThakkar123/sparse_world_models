#!/usr/bin/env bash
# Build the ONSET benchmark: change prediction with the momentum shortcut removed.
#
# Why this exists. The motion-filtered benchmark ("keep any step where something moved") is
# won by a one-line rule -- predict change iff the object is already moving -- at every object
# count and in both environments (F1 0.90 tabletop, 0.99 planar, against the learned gate's
# 0.86 and 0.44-0.74). P(changed | already moving) is 0.94-0.99 there, so the metric measures
# momentum continuation rather than prediction. See experiments/momentum_shortcut.py.
#
# The onset filter keeps only transitions where a *stationary* object starts moving, which can
# only happen through contact. That removes the shortcut from the training signal as well as
# the metric.
#
# The catch this script exists to solve: onset events are ~3% of steps, so the standard 250
# episodes yield only ~156 training rows -- far too few to train on, and ~20 test rows is too
# few to measure with. EPISODES is therefore raised ~6x so the onset training split lands near
# the motion benchmark's ~950 rows, making the two benchmarks comparable on data quantity
# rather than differing on both task and sample size at once.
#
# Usage: bash experiments/onset_benchmark_pipeline.sh [COUNTS] [SEEDS]
set -euo pipefail

COUNTS="${1:-3 5 8}"
SEEDS="${2:-0 1 2}"
BOUND=0.26
SEP=0.09
EPISODES=1500
MAX_STEPS=100

echo "############ GENERATE (${EPISODES} episodes; onset events are ~3% of steps) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    OUT="data/transitions/onset_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    echo "  generating N=${N} seed=${S}"
    python -m experiments.generate_transitions --policy scripted \
      --episodes "$EPISODES" --max-steps "$MAX_STEPS" --num-objects "$N" --seed "$S" \
      --object-bound "$BOUND" --min-object-separation "$SEP" \
      --output "$OUT" --run-name "onset_${N}obj_s${S}"
  done
done

echo "############ ONSET SPLITS (episode-disjoint) ############"
python -m experiments.build_clean_splits --counts $COUNTS --seeds $SEEDS \
  --filter-mode onset \
  --input-template "data/transitions/onset_{n}obj_s{seed}.npz" \
  --output-template "data/transitions/splits_onset_{n}obj_s{seed}"

echo "############ TRAIN (global + contact featurisation, and the dense baseline) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    D="data/transitions/splits_onset_${N}obj_s${S}"; ST="onset_${N}obj_s${S}_hard"
    python -m experiments.train_dense_baseline \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "dense_onset_${N}obj_s${S}" --epochs 25 --hidden-dim 256 --num-layers 3 --seed "$S"
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_onset_${N}obj_s${S}" --epochs 15 --sparsity-weight 0.2 \
      --auto-balance-bce --seed "$S"
    # The velocity-free featurisation, which is the one that helps on onset.
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_onset_contact_${N}obj_s${S}" --feature-mode contact \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ ONSET_BENCHMARK_DONE ############"
