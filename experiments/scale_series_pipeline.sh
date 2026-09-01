#!/usr/bin/env bash
# W4a: regenerate the object-count series at ONE geometry, extended to N=20, 5 seeds.
#
# Fixes the standing geometry confound. The published series used bounds +-0.18 /
# separation 0.12 for N=3,5 but +-0.22 / 0.09 for N=8, because 8 boxes did not fit the
# default packing -- so any cross-N trend touching 8 crossed a geometry shift as well as a
# count shift. Probing shows N=3,5,8,12,20 all place reliably at a single setting
# (bounds +-0.26, separation 0.09), so the whole series can share one geometry and N
# becomes the only variable.
#
# Why fixed bounds rather than fixed density: holding objects-per-area constant would need
# bounds up to +-0.735 at N=50, against a table of half-size 0.34 and a pusher that only
# reaches +-0.26. Objects would sit off the table and out of reach, and most would never be
# touched -- which would flatter the sparse model for trivial reasons. Fixed bounds instead
# let density rise with N (0.028 -> 0.185 from N=3 to N=20), which is exactly the axis the
# "copying unchanged objects compounds" prediction is about, and keeps every object
# reachable.
#
# N=30 and N=50 need separation 0.075 / 0.055 to place at all, so they do NOT belong in this
# series; run them separately and report them as geometry-shifted.
#
# ENV selects the simulator. 'planar' is the dependency-free 2D domain, which runs ~45x
# faster than MuJoCo and so affords higher counts and more seeds; it reproduces the same
# change sparsity (0.267 changed-object fraction against the tabletop's ~0.27) with genuinely
# different dynamics, which is what makes it a breadth result rather than a re-run.
#
# Usage: bash experiments/scale_series_pipeline.sh [COUNTS] [SEEDS] [ENV] [TAG]
#        bash experiments/scale_series_pipeline.sh "3 5 8 12 20" "0 1 2 3 4"
#        bash experiments/scale_series_pipeline.sh "3 5 8 12 20 30" "0 1 2 3 4" planar planar
set -euo pipefail

COUNTS="${1:-3 5 8 12 20}"
SEEDS="${2:-0 1 2 3 4}"
ENV_NAME="${3:-tabletop}"
TAG="${4:-uni}"
BOUND=0.26
SEP=0.09
# "Real motion" is environment-relative. The 0.02 default was tuned for the tabletop and
# keeps only 0.13% of planar steps (15 training rows at N=3); 0.010 retains a comparable
# fraction there. See models/envs/planar_push.py.
if [[ "$ENV_NAME" == "planar" ]]; then MOTION_THRESHOLD=0.010; else MOTION_THRESHOLD=0.020; fi
EPISODES=250
MAX_STEPS=100

echo "############ GENERATE env=${ENV_NAME} tag=${TAG} (bound=${BOUND} sep=${SEP}, identical for every count) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    OUT="data/transitions/scale_${N}obj_${TAG}_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    echo "  generating N=${N} seed=${S}"
    python -m experiments.generate_transitions --policy scripted --env "$ENV_NAME" \
      --episodes "$EPISODES" --max-steps "$MAX_STEPS" --num-objects "$N" --seed "$S" \
      --object-bound "$BOUND" --min-object-separation "$SEP" \
      --output "$OUT" --run-name "scale_${N}obj_${TAG}_s${S}"
  done
done

echo "############ CLEAN SPLITS (episode-disjoint; see build_clean_splits.py) ############"
python -m experiments.build_clean_splits --counts $COUNTS --seeds $SEEDS \
  --min-max-xy-delta "$MOTION_THRESHOLD" \
  --input-template "data/transitions/scale_{n}obj_${TAG}_s{seed}.npz" \
  --output-template "data/transitions/splits_clean_{n}obj_${TAG}_s{seed}"

echo "############ TRAIN BASELINES ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    SPLITDIR="data/transitions/splits_clean_${N}obj_${TAG}_s${S}"
    STEM="scale_${N}obj_${TAG}_s${S}_hard"
    # Same hyperparameters as scale_pipeline.sh, so the only change is the data.
    python -m experiments.train_dense_baseline \
      --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
      --run-name "dense_${TAG}_${N}obj_s${S}" --epochs 25 --hidden-dim 256 --num-layers 3 --seed "$S"
    python -m experiments.train_sparse_model \
      --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
      --run-name "sparse_${TAG}_${N}obj_s${S}" --epochs 15 --sparsity-weight 0.2 \
      --auto-balance-bce --seed "$S"
  done
done

echo "############ DONE ############"
