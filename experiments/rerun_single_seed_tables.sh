#!/usr/bin/env bash
# Re-run the four remaining single-seed analyses on the clean splits, across seeds.
#
# RESULTS.md flags these as seed-0-only: the parameter-matched dense control, the oracle-gate
# diagnostic, the cross-count transfer matrix, and the sample-efficiency sweep. They are also
# all computed on the ORIGINAL splits, which leak (25% of source episodes have chunks in both
# train and test -- see build_clean_splits.py), so they need re-running on clean splits
# regardless of the seed question. This does both at once.
#
# The gate ablation was the fifth such table and is already done at 3 seeds
# (gate_ablation_extended_clean{,_s1,_s2}).
#
# Requires the clean baselines: bash experiments/clean_pipeline.sh N SEED for each cell.
#
# Usage: bash experiments/rerun_single_seed_tables.sh [SEEDS] [COUNTS]
set -euo pipefail

SEEDS="${1:-0 1 2}"
COUNTS="${2:-3 5 8}"
SPLIT_TEMPLATE='data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz'
SPARSE_TEMPLATE='models/checkpoints/sparse_clean_{n}obj_s{seed}.pt'
DENSE_TEMPLATE='models/checkpoints/dense_clean_{n}obj_s{seed}.pt'

for S in $SEEDS; do
  echo "############ seed ${S}: oracle-gate diagnostic ############"
  python -m experiments.oracle_gate_diagnostic --counts $COUNTS --seed "$S" \
    --split-template "$SPLIT_TEMPLATE" --sparse-template "$SPARSE_TEMPLATE" \
    --run-name "oracle_gate_clean_s${S}"

  echo "############ seed ${S}: parameter-matched dense control ############"
  python -m experiments.param_matched_baseline --counts $COUNTS --seed "$S" \
    --split-template "$SPLIT_TEMPLATE" --sparse-template "$SPARSE_TEMPLATE" \
    --dense-template "$DENSE_TEMPLATE" --run-name "param_matched_clean_s${S}"

  echo "############ seed ${S}: cross-count transfer matrix ############"
  python -m experiments.compositional_generalization --counts $COUNTS --seed "$S" \
    --split-template "$SPLIT_TEMPLATE" --run-name "compositional_clean_s${S}"

  echo "############ seed ${S}: sample-efficiency sweep ############"
  for N in $COUNTS; do
    DIR="data/transitions/splits_clean_${N}obj_s${S}"
    STEM="scale_${N}obj_s${S}_hard"
    python -m experiments.sample_efficiency \
      --train "${DIR}/${STEM}_train.npz" --val "${DIR}/${STEM}_val.npz" \
      --test "${DIR}/${STEM}_test.npz" --seed "$S" \
      --run-name "sample_efficiency_clean_${N}obj_s${S}"
  done
done

echo "############ DONE ############"
