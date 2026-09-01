#!/usr/bin/env bash
# Train on the interaction benchmark, then audit it with the full trivial-rule battery.
#
# The interaction benchmark keeps only transitions where a stationary object starts moving
# while some OTHER object is closer to the pusher -- change that reached the object through
# another object rather than from the pusher directly. It exists because both previous
# benchmarks were shown to be solvable without learning:
#
#   motion benchmark  -> "already moving"       F1 0.866
#   onset  benchmark  -> "nearest to pusher"    F1 0.925, ZERO parameters
#                        logistic on 1 feature  F1 0.916
#
# THE ONLY RESULT THAT MATTERS HERE IS THE AUDIT, AND IT MAY WELL BE NEGATIVE.
# The benchmark was designed using distance-to-pusher, which is the same quantity the winning
# trivial rule uses, so it would be circular to declare victory just because that one rule
# now fails. The battery therefore includes five rules invented specifically against this
# benchmark -- above all `near_a_mover`, since in a chain the object that starts moving is by
# definition adjacent to something already in motion, which is the obvious next one-liner.
#
# A benchmark that defeats the rule it was designed against and is then won by the next
# one-liner is no improvement. If that is what happens, it gets reported as such.
#
# Usage: bash experiments/interaction_pipeline.sh [ENV] [COUNTS] [SEEDS]
set -uo pipefail

ENVNAME="${1:-billiards}"
COUNTS="${2:-5 8}"
SEEDS="${3:-0 1 2}"
# Per-domain motion threshold from experiments/domain_characterization.py.
case "$ENVNAME" in
  billiards) TH=0.031 ;;
  clutter)   TH=0.029 ;;
  planar)    TH=0.010 ;;
  tabletop)  TH=0.020 ;;
  *) echo "unknown env $ENVNAME" >&2; exit 1 ;;
esac

# N=3 is deliberately excluded by default: with three objects an indirect onset is rare
# enough (~0.00048 of steps) that a cell yields ~250 train / ~35 test rows, which is too few
# to train on and too few to measure with. Reported rather than quietly included.
echo "############ TRAIN on the interaction benchmark (${ENVNAME}) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    D="data/transitions/splits_interaction_${ENVNAME}_${N}obj_s${S}"
    ST="interaction_${ENVNAME}_${N}obj_s${S}_hard"
    if [[ ! -f "$D/${ST}_train.npz" ]]; then
      echo "  MISSING $D/${ST}_train.npz -- skipping"
      continue
    fi
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_interaction_${ENVNAME}_${N}obj_s${S}" \
      --epochs 25 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    python -m experiments.train_sparse_model \
      --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
      --run-name "sparse_interaction_contact_${ENVNAME}_${N}obj_s${S}" --feature-mode contact \
      --epochs 25 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ AUDIT (the whole point) ############"
python -m experiments.onset_shortcut_audit \
  --benchmarks interaction --counts $COUNTS --seeds $SEEDS \
  --split-templates "interaction=data/transitions/splits_interaction_${ENVNAME}_{n}obj_s{seed}/interaction_${ENVNAME}_{n}obj_s{seed}_hard_{split}.npz" \
  --gate-templates "interaction=models/checkpoints/sparse_interaction_${ENVNAME}_{n}obj_s{seed}.pt" \
  --contact-gate-templates "interaction=models/checkpoints/sparse_interaction_contact_${ENVNAME}_{n}obj_s{seed}.pt" \
  --run-name "audit_interaction_${ENVNAME}"

echo "############ INTERACTION_PIPELINE_DONE ############"
