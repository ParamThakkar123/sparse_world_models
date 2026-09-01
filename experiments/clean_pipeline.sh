#!/usr/bin/env bash
# Retrain the canonical sparse/dense baselines on the EPISODE-DISJOINT clean splits.
#
# Why this exists: the checkpoints in models/checkpoints/{sparse,dense}_{N}obj_s{S}.pt were
# trained on splits where 25% of source episodes have chunks in both train and test (see
# experiments/build_clean_splits.py for the mechanism). Any new baseline evaluated on the
# clean test split must be compared against models that never saw it, or the old
# checkpoints get a free advantage.
#
# Hyperparameters mirror scale_pipeline.sh exactly -- dense 25 epochs / 256 wide / 3 layers,
# sparse 15 epochs / sparsity 0.2 / auto-balanced BCE -- so the ONLY difference from the
# published runs is which rows are in the training set.
#
# Usage: bash experiments/clean_pipeline.sh N SEED
#        VARIANT=dense bash experiments/clean_pipeline.sh N SEED   # dense-interaction data
#
# VARIANT mirrors scale_pipeline.sh: it is woven into the tag so variant runs never collide
# with the main ones. The dense-interaction variant is needed because on clean splits the
# sparse model marginally *loses* to no-op on overall L2 at N=8 (0.0741 vs 0.0739), and the
# packed-object variant is the existing answer to that saturation -- but it was built through
# the leaky pipeline, so it has to be rebuilt here before the N=8 column can be used.
set -euo pipefail

N="$1"
SEED="$2"
VARIANT="${VARIANT:-}"
TAG="${N}obj${VARIANT:+_${VARIANT}}_s${SEED}"
SPLITDIR="data/transitions/splits_clean_${TAG}"
STEM="scale_${TAG}_hard"

if [[ ! -f "${SPLITDIR}/${STEM}_train.npz" ]]; then
  echo "Missing ${SPLITDIR}/${STEM}_train.npz -- run: python -m experiments.build_clean_splits" >&2
  exit 1
fi

echo "############ [clean ${TAG}] 1/2 TRAIN DENSE ############"
python -m experiments.train_dense_baseline \
  --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
  --run-name "dense_clean_${TAG}" --epochs 25 --hidden-dim 256 --num-layers 3 --seed "${SEED}"

echo "############ [clean ${TAG}] 2/2 TRAIN SPARSE ############"
python -m experiments.train_sparse_model \
  --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
  --run-name "sparse_clean_${TAG}" --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "${SEED}"

echo "############ [clean ${TAG}] DONE ############"
