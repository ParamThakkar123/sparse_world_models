#!/usr/bin/env bash
# Clean sparsity-weight ablation on 3-obj seed-0 hard data, standard 15-epoch config.
set -euo pipefail
TR="data/transitions/splits_3obj_s0/scale_3obj_s0_hard_train.npz"
VA="data/transitions/splits_3obj_s0/scale_3obj_s0_hard_val.npz"
for SW in 0.0 0.05 0.2 0.5 1.0; do
  TAG=$(echo "${SW}" | tr '.' 'p')
  echo "############ sparsity_weight=${SW} ############"
  python -m experiments.train_sparse_model \
    --train "${TR}" --val "${VA}" \
    --run-name "ablation_sparsity_sw${TAG}" \
    --epochs 15 --sparsity-weight "${SW}" --auto-balance-bce --seed 0
done
echo "############ SPARSITY ABLATION DONE ############"
