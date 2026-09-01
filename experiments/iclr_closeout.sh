#!/usr/bin/env bash
# Everything RESULTS.md still lists as outstanding, serialised into one unattended run.
#
# Four gaps, in descending order of how much they can change the paper. Stages run in that
# order deliberately: if the machine is stopped halfway, what completed is the part worth
# having.
#
#   STAGE 1  The clutter domain, audited.
#            RESULTS.md ("Why it happens, and what a non-degenerate benchmark would need")
#            names dense packing + the onset filter as the one combination in the project
#            that might resist both shortcuts, and says explicitly that no dataset combines
#            them. This builds it, at the clutter env's designed separation of 0.07 rather
#            than the 0.09 the cross-domain pipeline overrode it with -- the packing IS the
#            manipulated variable, so running it at the loose separation would test nothing.
#            Then the same for the interaction filter, so all three filters are covered on
#            the domain most likely to break the pattern.
#            This is the only stage whose outcome could change what the paper claims.
#
#   STAGE 2  Five seeds on the headline audit tables.
#            Every audit comparison currently reports p = 0.250, which is the SMALLEST
#            attainable p-value with three seeds (2^3 = 8 sign-flip outcomes). Reviewers read
#            a floored p as a null result. Five seeds drop the floor to 0.0625.
#
#   STAGE 3  Three training seeds for the published baselines in the planning comparison.
#            The PETS-over-sparse ordering (0.350 vs 0.250) rests on ONE training seed and 20
#            episodes, where a success rate carries about +/-0.10. RESULTS.md flags this as
#            "the obvious next step and is not yet done". Until it is, the ordering cannot be
#            stated -- only the qualitative claim that four published models plan.
#
#   STAGE 4  The pixel validation run that memory pressure kept killing.
#            The foreground-weighted loss was implemented and unit-tested but no caller could
#            reach it (fixed: `--foreground-weight`). Runs alone, last, because it is the
#            stage most likely to be cut for time and the one whose failure costs least.
#
# Serialised on purpose. Running these concurrently is what killed the cross-domain split
# stage and every previous pixel attempt with a bare MemoryError on this 16 GB machine.
#
# Every stage skips work whose output already exists, so re-running resumes rather than
# repeats. Failures are logged and do NOT abort the run: an overnight job that dies in stage 1
# and leaves stages 2-4 undone is worse than one that reports a broken stage and continues.
#
# Usage: bash experiments/iclr_closeout.sh [STAGES]      e.g. "1 3"   (default: 1 2 3 4)

set -uo pipefail

STAGES="${1:-1 2 3 4}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

FAILURES=()
run() {
  local label="$1"; shift
  echo ""
  echo "---- ${label}"
  if ! "$@"; then
    echo "!!!! FAILED: ${label}"
    FAILURES+=("${label}")
  fi
}

wants() { [[ " ${STAGES} " == *" $1 "* ]]; }

# --------------------------------------------------------------------------------------
if wants 1; then
echo "################################################################"
echo "############ STAGE 1: the clutter domain, all three filters ####"
echo "################################################################"

CLUTTER_TH=0.029          # derived by domain_characterization.py, not hand-tuned
CLUTTER_BOUND=0.26
CLUTTER_SEP=0.07          # the env's designed value: THIS is the dense packing under test
CLUTTER_EPISODES=1500     # onset events are ~3% of steps; matches onset_benchmark_pipeline.sh

echo "############ 1a: generate clutter episodes (dense packing) ############"
for N in 3 5 8; do
  for S in 0 1 2; do
    OUT="data/transitions/onsetclutter_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    run "generate clutter N=${N} s=${S}" \
      python -m experiments.generate_transitions --policy scripted --env clutter \
        --episodes "$CLUTTER_EPISODES" --max-steps 100 --num-objects "$N" --seed "$S" \
        --object-bound "$CLUTTER_BOUND" --min-object-separation "$CLUTTER_SEP" \
        --output "$OUT" --run-name "onsetclutter_${N}obj_s${S}"
  done
done

echo "############ 1b: onset splits (episode-disjoint) ############"
run "clutter onset splits" \
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2 \
    --filter-mode onset --min-max-xy-delta "$CLUTTER_TH" \
    --input-template "data/transitions/onsetclutter_{n}obj_s{seed}.npz" \
    --output-template "data/transitions/splits_onsetclutter_{n}obj_s{seed}"

echo "############ 1c: train both gates on the clutter onset benchmark ############"
for N in 3 5 8; do
  for S in 0 1 2; do
    D="data/transitions/splits_onsetclutter_${N}obj_s${S}"
    ST="onsetclutter_${N}obj_s${S}_hard"
    if [[ ! -f "$D/${ST}_train.npz" ]]; then echo "  MISSING $D/${ST}_train.npz -- skipping"; continue; fi
    run "sparse global clutter-onset ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_onsetclutter_${N}obj_s${S}" \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    # The velocity-free featurisation is the control that matters: a proximity rule beating a
    # velocity-featurised gate says little, but beating a gate that can already SEE signed
    # contact distance says the task itself is degenerate.
    run "sparse contact clutter-onset ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_onsetclutter_contact_${N}obj_s${S}" --feature-mode contact \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ 1d: AUDIT the clutter onset benchmark (the whole point) ############"
run "audit clutter onset" \
  python -m experiments.onset_shortcut_audit \
    --benchmarks onsetclutter --counts 3 5 8 --seeds 0 1 2 \
    --split-templates "onsetclutter=data/transitions/splits_onsetclutter_{n}obj_s{seed}/onsetclutter_{n}obj_s{seed}_hard_{split}.npz" \
    --gate-templates "onsetclutter=models/checkpoints/sparse_onsetclutter_{n}obj_s{seed}.pt" \
    --contact-gate-templates "onsetclutter=models/checkpoints/sparse_onsetclutter_contact_{n}obj_s{seed}.pt" \
    --run-name "audit_onset_clutter"

echo "############ 1e: the interaction filter on clutter ############"
run "build clutter interaction benchmark" \
  python -m experiments.build_interaction_benchmark --env clutter --counts 5 8 --seeds 0 1 2 \
    --object-bound "$CLUTTER_BOUND" --min-object-separation "$CLUTTER_SEP" \
    --run-name "build_interaction_clutter"
run "train + audit clutter interaction" \
  bash experiments/interaction_pipeline.sh clutter "5 8" "0 1 2"

echo "############ STAGE_1_DONE ############"
fi

# --------------------------------------------------------------------------------------
if wants 2; then
echo "################################################################"
echo "############ STAGE 2: five seeds on the audit tables ###########"
echo "################################################################"

echo "############ 2a: motion benchmark, seeds 3 and 4 ############"
for N in 3 5 8; do
  # N=8 needs the wider bounds / tighter separation eight boxes were generated with at seeds
  # 0-2; passing the env default makes placement impossible and generation dies at reset.
  if [[ "$N" == "8" ]]; then GEOM=(--object-bound 0.22 --min-object-separation 0.09); else GEOM=(); fi
  for S in 3 4; do
    OUT="data/transitions/scale_${N}obj_s${S}.npz"
    if [[ -f "$OUT" ]]; then echo "  skip $OUT (exists)"; continue; fi
    run "generate motion N=${N} s=${S}" \
      python -m experiments.generate_transitions --policy scripted \
        --episodes 250 --max-steps 100 --num-objects "$N" --seed "$S" "${GEOM[@]}" \
        --output "$OUT" --run-name "scale_${N}obj_s${S}"
  done
done

run "clean motion splits seeds 3-4" \
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 3 4 \
    --input-template "data/transitions/scale_{n}obj_s{seed}.npz" \
    --output-template "data/transitions/splits_clean_{n}obj_s{seed}"

for N in 3 5 8; do
  for S in 3 4; do
    D="data/transitions/splits_clean_${N}obj_s${S}"; ST="scale_${N}obj_s${S}_hard"
    if [[ ! -f "$D/${ST}_train.npz" ]]; then echo "  MISSING $D/${ST}_train.npz -- skipping"; continue; fi
    run "sparse global motion ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_clean_${N}obj_s${S}" \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    run "sparse contact motion ${N}obj s${S}" \
      python -m experiments.train_sparse_model \
        --train "$D/${ST}_train.npz" --val "$D/${ST}_val.npz" \
        --run-name "sparse_contact_clean_${N}obj_s${S}" --feature-mode contact \
        --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done
done

echo "############ 2b: onset benchmark, seeds 3 and 4 ############"
run "onset benchmark seeds 3-4" bash experiments/onset_benchmark_pipeline.sh "3 5 8" "3 4"

echo "############ 2c: re-run the audit at five seeds ############"
# Same defaults as the published audit; only --seeds changes, so the 3-seed and 5-seed runs
# are directly comparable and any drift is the seeds rather than the configuration.
run "audit motion+onset, 5 seeds" \
  python -m experiments.onset_shortcut_audit \
    --benchmarks motion onset --counts 3 5 8 --seeds 0 1 2 3 4 \
    --run-name "onset_shortcut_audit_5seed"

echo "############ 2d: five seeds for the published baselines too ############"
run "literature baselines motion, seeds 3-4" \
  python -m experiments.literature_baselines --counts 3 5 8 --seeds 3 4 \
    --run-name "literature_baselines_motion_s34"
run "literature baselines onset, seeds 3-4" \
  python -m experiments.literature_baselines --counts 3 5 8 --seeds 3 4 \
    --split-template "data/transitions/splits_onset_{n}obj_s{seed}/onset_{n}obj_s{seed}_hard_{split}.npz" \
    --run-name "literature_baselines_onset_s34"

echo "############ STAGE_2_DONE ############"
fi

# --------------------------------------------------------------------------------------
if wants 3; then
echo "################################################################"
echo "############ STAGE 3: planning, three training seeds ###########"
echo "################################################################"

# Only the training seed varies -- same split, same 20 held-out episode configurations
# (base seed 5000), same planner, same cost. That is the protocol the sparse and dense
# conditions already follow, so this makes the published baselines comparable to them.
for S in 1 2; do
  run "literature baselines for planning, seed ${S}" \
    python -m experiments.literature_baselines --counts 3 --seeds "$S" \
      --split-template "data/transitions/splits_plan_mixed_3obj/plan_mixed_3obj_{split}.npz" \
      --feature-mode contact --epochs 25 \
      --checkpoint-dir models/checkpoints --checkpoint-tag litplan \
      --run-name "literature_baselines_planmixed_s${S}"
done

for S in 1 2; do
  run "plan through published baselines, seed ${S}" \
    python -m experiments.planning_mpc \
      --conditions gns cswm slotformer pets nps \
      --literature-checkpoint-dir models/checkpoints --literature-tag litplan \
      --literature-seed "$S" \
      --num-objects 3 --num-episodes 20 --base-seed 5000 --max-steps 60 \
      --run-name "planning_literature_3obj_s${S}"
done

echo "############ STAGE_3_DONE ############"
fi

# --------------------------------------------------------------------------------------
if wants 4; then
echo "################################################################"
echo "############ STAGE 4: pixels, uninterrupted ####################"
echo "################################################################"

# foreground-weight 10 is the value tests/test_statistics_and_perception.py pins as clearly
# foreground-dominated (object-pixel errors count >10x background ones). Weight 0 is run as
# the control: without it, an improvement cannot be attributed to the reweighting rather
# than to running alone on an unloaded machine, which is the other thing that changed.
for W in 0 10; do
  run "pixel keypoint, foreground-weight ${W}" \
    python -m experiments.pixel_benchmark --perception keypoint --counts 3 --seeds 0 \
      --foreground-weight "$W" --slot-epochs 40 \
      --run-name "pixel_keypoint_fg${W}"
done
run "pixel slot attention (the collapse, for the record)" \
  python -m experiments.pixel_benchmark --perception slot --counts 3 --seeds 0 \
    --slot-epochs 40 --run-name "pixel_slot_control"

echo "############ STAGE_4_DONE ############"
fi

# --------------------------------------------------------------------------------------
echo ""
echo "################################################################"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "############ CLOSEOUT_DONE -- no failures ############"
else
  echo "############ CLOSEOUT_DONE -- ${#FAILURES[@]} FAILED STEP(S) ############"
  for f in "${FAILURES[@]}"; do echo "  FAILED: $f"; done
fi
