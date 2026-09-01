#!/usr/bin/env bash
# Rebuild the clutter onset benchmark at an episode count that can actually be trained on.
#
# WHY THIS EXISTS. `iclr_closeout.sh` stage 1 built the dense-packed clutter onset benchmark
# at 1500 episodes per cell, the count the tabletop onset benchmark uses. On clutter that
# yields 38-174 training rows and 6-24 TEST rows per cell:
#
#   N=3  38 / 47 / 43 train,   6 /  6 / 10 test
#   N=5  80 / 83 / 102 train, 17 / 11 / 19 test
#   N=8 133 / 174 / 133 train, 24 / 20 /  9 test
#
# That is below this project's own stated minimum. RESULTS.md excludes N=3 from the
# interaction benchmark on exactly this ground -- "at ~250 train / ~35 test rows per cell it
# is too thin to train on or measure with" -- and these cells are thinner still. The learned
# gate's 0.32 F1 in `runs/audit_onset_clutter/` is therefore uninterpretable: it is at least
# as likely to be starvation as a finding, and reporting it would repeat the mistake this
# whole line of work exists to expose, in the opposite direction.
#
# Onset events are ~0.065% of clutter steps against ~3% of tabletop steps -- a 46x
# difference -- because in dense high-friction clutter objects are mostly already in contact
# and jostling, so a clean stationary-to-moving transition is rare. The fix is the same one
# `onset_benchmark_pipeline.sh` applies to the tabletop: generate enough episodes that the
# training split lands near the ~750-950 rows the other benchmarks have, so the comparison
# is on task rather than on sample size.
#
# N=3 is excluded and stays excluded: it would need ~27000 episodes per cell to reach the
# same row count, and the exclusion is reported rather than quietly applied.
#
# Usage: bash experiments/clutter_onset_rebuild.sh          (run AFTER iclr_closeout.sh)

set -uo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TH=0.029          # derived by domain_characterization.py
BOUND=0.26
SEP=0.07          # the clutter env's designed separation: the dense packing under test
EPISODES=12000    # ~8x the tabletop count, from the measured 46x lower onset rate
COUNTS="5 8"
SEEDS="0 1 2"

FAILURES=()
run() {
  local label="$1"; shift
  echo ""; echo "---- ${label}"
  if ! "$@"; then echo "!!!! FAILED: ${label}"; FAILURES+=("${label}"); fi
}

echo "############ GENERATE ${EPISODES} episodes/cell (onset is ~0.065% of clutter steps) ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    OUT="data/transitions/onsetclutter12k_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    run "generate clutter12k N=${N} s=${S}" \
      python -m experiments.generate_transitions --policy scripted --env clutter \
        --episodes "$EPISODES" --max-steps 100 --num-objects "$N" --seed "$S" \
        --object-bound "$BOUND" --min-object-separation "$SEP" \
        --output "$OUT" --run-name "onsetclutter12k_${N}obj_s${S}"
  done
done

echo "############ ONSET SPLITS ############"
run "clutter12k onset splits" \
  python -m experiments.build_clean_splits --counts $COUNTS --seeds $SEEDS \
    --filter-mode onset --min-max-xy-delta "$TH" \
    --input-template "data/transitions/onsetclutter12k_{n}obj_s{seed}.npz" \
    --output-template "data/transitions/splits_onsetclutter12k_{n}obj_s{seed}"

echo "############ TRAIN both gates ############"
for N in $COUNTS; do
  for S in $SEEDS; do
    D="data/transitions/splits_onsetclutter12k_${N}obj_s${S}"
    ST="onsetclutter12k_${N}obj_s${S}_hard"
    if [[ ! -f "$D/${ST}_train.npz" ]]; then echo "  MISSING $D/${ST}_train.npz"; continue; fi
    run "sparse global clutter12k ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_onsetclutter12k_${N}obj_s${S}" \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    run "sparse contact clutter12k ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_onsetclutter12k_contact_${N}obj_s${S}" --feature-mode contact \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ AUDIT ############"
run "audit clutter12k onset" \
  python -m experiments.onset_shortcut_audit \
    --benchmarks onsetclutter12k --counts $COUNTS --seeds $SEEDS \
    --split-templates "onsetclutter12k=data/transitions/splits_onsetclutter12k_{n}obj_s{seed}/onsetclutter12k_{n}obj_s{seed}_hard_{split}.npz" \
    --gate-templates "onsetclutter12k=models/checkpoints/sparse_onsetclutter12k_{n}obj_s{seed}.pt" \
    --contact-gate-templates "onsetclutter12k=models/checkpoints/sparse_onsetclutter12k_contact_{n}obj_s{seed}.pt" \
    --run-name "audit_onset_clutter12k"

echo ""
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "############ CLUTTER_ONSET_REBUILD_DONE -- no failures ############"
else
  echo "############ CLUTTER_ONSET_REBUILD_DONE -- ${#FAILURES[@]} FAILED ############"
  for f in "${FAILURES[@]}"; do echo "  FAILED: $f"; done
fi
