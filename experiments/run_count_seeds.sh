#!/usr/bin/env bash
# Run seeds 0,1,2 for one object count. Usage: run_count_seeds.sh N [OBJECT_BOUND] [MIN_SEP]
set -euo pipefail
N="$1"; shift
for SEED in 0 1 2; do
  echo "==================== ${N}obj seed ${SEED} ===================="
  bash experiments/scale_pipeline.sh "${N}" "${SEED}" "$@"
done
echo "==================== ${N}obj ALL SEEDS DONE ===================="
