#!/usr/bin/env bash
# Sample-efficiency sweep (10/25/50/100% of the training split) for one object
# count and seed, using the canonical per-tag hard splits.
# Usage: run_sample_efficiency.sh N [SEED]
set -euo pipefail

N="$1"
SEED="${2:-0}"
TAG="${N}obj_s${SEED}"
SPLITDIR="data/transitions/splits_${TAG}"
STEM="scale_${TAG}_hard"

python -m experiments.sample_efficiency \
  --train "${SPLITDIR}/${STEM}_train.npz" \
  --val   "${SPLITDIR}/${STEM}_val.npz" \
  --test  "${SPLITDIR}/${STEM}_test.npz" \
  --run-name "sample_efficiency_${TAG}" \
  --seed "${SEED}"
