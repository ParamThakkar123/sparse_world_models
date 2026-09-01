#!/usr/bin/env bash
# Full Phase-4 pipeline for a given object count and seed.
# Mirrors the 3-object "hard" recipe exactly (see phase4_comparison_hard_v1).
# Usage: scale_pipeline.sh N SEED [OBJECT_BOUND] [MIN_SEP]
set -euo pipefail

N="$1"
SEED="$2"
OBJECT_BOUND="${3:-}"
MIN_SEP="${4:-}"
# Optional variant label (env var), woven into the tag so variants don't collide
# with the main runs. E.g. VARIANT=dense -> tag "8obj_dense_s0".
VARIANT="${VARIANT:-}"

TAG="${N}obj${VARIANT:+_${VARIANT}}_s${SEED}"
NAME="scale_${TAG}"
FULL="data/transitions/${NAME}.npz"
HARD="data/transitions/${NAME}_hard.npz"
SPLITDIR="data/transitions/splits_${TAG}"
STEM="${NAME}_hard"

GEN_GEOM=()
if [[ -n "${OBJECT_BOUND}" ]]; then GEN_GEOM+=(--object-bound "${OBJECT_BOUND}"); fi
if [[ -n "${MIN_SEP}" ]]; then GEN_GEOM+=(--min-object-separation "${MIN_SEP}"); fi

echo "############ [${TAG}] 1/6 GENERATE ############"
python -m experiments.generate_transitions --policy scripted --episodes 250 --max-steps 100 \
  --num-objects "${N}" --seed "${SEED}" "${GEN_GEOM[@]}" --output "${FULL}" --run-name "${NAME}"

echo "############ [${TAG}] 2/6 HARD SUBSET ############"
python -m experiments.create_hard_subset --input "${FULL}" --output "${HARD}" --seed "${SEED}"

echo "############ [${TAG}] 3/6 SPLIT ############"
python -m experiments.split_dataset --input "${HARD}" --output-dir "${SPLITDIR}" --seed "${SEED}"

echo "############ [${TAG}] 4/6 TRAIN DENSE ############"
python -m experiments.train_dense_baseline \
  --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
  --run-name "dense_${TAG}" --epochs 25 --hidden-dim 256 --num-layers 3 --seed "${SEED}"

echo "############ [${TAG}] 5/6 TRAIN SPARSE ############"
python -m experiments.train_sparse_model \
  --train "${SPLITDIR}/${STEM}_train.npz" --val "${SPLITDIR}/${STEM}_val.npz" \
  --run-name "sparse_${TAG}" --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "${SEED}"

echo "############ [${TAG}] 6/6 PHASE4 COMPARE ############"
python -m experiments.compare_phase4_models \
  --data "${SPLITDIR}/${STEM}_test.npz" \
  --dense-checkpoint "models/checkpoints/dense_${TAG}.pt" \
  --sparse-checkpoint "models/checkpoints/sparse_${TAG}.pt" \
  --run-name "phase4_${TAG}"

echo "############ [${TAG}] DONE ############"
