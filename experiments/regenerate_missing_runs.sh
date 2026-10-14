#!/usr/bin/env bash
# Regenerate the run directories paper/main.tex quotes but which were lost from disk
# (experiments/runs/ is gitignored apart from summary.json/config.json).
#
# STRICTLY SERIAL. This machine (16 GB RAM, 4 GB GPU) cannot hold two pipelines at once;
# running them concurrently is what killed earlier attempts with a bare MemoryError. Nothing
# here is backgrounded, and every job is forced onto the CPU (as the originals were: every
# script used defaults to --device cpu).
#
# RESUMABLE. A stage is skipped when all of its final summary.json files exist; inside a stage
# every training / analysis step is skipped when its run summary.json (and checkpoint, for
# training) exists. A step that crashed half-way leaves a run dir without summary.json, which
# is cleared and redone on the next invocation.
#
# STOPS ON THE FIRST ERROR (set -e + pipefail). Each stage logs to
# experiments/runs/_log_regenerate_<stage>.log.
#
# Run names are the ORIGINAL ones wherever the original is known, so paper/claims.py,
# make_iclr_tables.py and RESULTS.md find them. Where no original name survived, the new name
# is marked NEW below.
#
# Stages, cheapest first (estimates from the 2026-09-03 run timestamps on this machine):
#   interaction_stats  group 7  ~10 s    interaction_construction_stats (NEW name)
#   dense_packed       group 2  ~3 min   audit_dense_packed
#   lit_onset_s34      group 8  ~5 min   literature_baselines_onset_s34 (rerun at full 3x2 coverage)
#   planar12k          group 1  ~8 min   audit_planar12k, momentum_shortcut_planar12k, momentum_shortcut_planar
#   domchar            group 3  ~10 min  domain_characterization
#   w3_validity        group 4  ~20 min  w3_validity_{3,5,8}obj_s{0,1,2}
#   w4_scale           group 5  ~20 min  dense_uni_*/sparse_uni_* + scale_analysis
#   planning           group 6  ~1.5-2 h planning_literature_3obj (+ sparse at seeds 1,2, NEW names)
#
# Usage (from anywhere):
#   bash experiments/regenerate_missing_runs.sh                 # all stages, in the order above
#   bash experiments/regenerate_missing_runs.sh planar12k domchar
#   bash experiments/regenerate_missing_runs.sh --list

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4      # matches iclr_closeout.sh / clutter_onset_rebuild.sh
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="" # CPU only, as every original run was
export PYTHONWARNINGS="ignore::FutureWarning"

ALL_STAGES=(interaction_stats dense_packed lit_onset_s34 planar12k domchar w3_validity w4_scale planning)
RUNS=experiments/runs
CK=models/checkpoints
T=data/transitions

if [[ "${1:-}" == "--list" ]]; then printf '%s\n' "${ALL_STAGES[@]}"; exit 0; fi
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=("${ALL_STAGES[@]}")
for s in "${STAGES[@]}"; do
  [[ " ${ALL_STAGES[*]} " == *" $s "* ]] || { echo "unknown stage: $s (try --list)" >&2; exit 2; }
done

# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
say() { echo "[$(date '+%F %T')] $*"; }

done_run() { [[ -f "$RUNS/$1/summary.json" ]]; }

# A run dir without summary.json is a crashed partial: clear it so ExperimentLogger does not
# append to a stale metrics.csv. A dir WITH summary.json whose checkpoint was lost only has its
# metrics.csv removed (summary/config are overwritten by the rerun).
prepare_run_dir() {
  local name="$1"
  if [[ -d "$RUNS/$name" && ! -f "$RUNS/$name/summary.json" ]]; then rm -rf "${RUNS:?}/$name"; fi
  rm -f "$RUNS/$name/metrics.csv"
}

require() { for f in "$@"; do [[ -e "$f" ]] || { echo "MISSING INPUT: $f" >&2; exit 1; }; done; }

# train_sparse NAME TRAIN VAL [extra args...]   (15 epochs / sparsity 0.2 / auto-balance unless overridden)
train_sparse() {
  local name="$1" tr="$2" va="$3"; shift 3
  if done_run "$name" && [[ -f "$CK/$name.pt" ]]; then say "skip train $name (exists)"; return; fi
  require "$tr" "$va"
  prepare_run_dir "$name"
  say "train sparse $name"
  python -m experiments.train_sparse_model --train "$tr" --val "$va" --run-name "$name" "$@"
}

train_dense() {
  local name="$1" tr="$2" va="$3"; shift 3
  if done_run "$name" && [[ -f "$CK/$name.pt" ]]; then say "skip train $name (exists)"; return; fi
  require "$tr" "$va"
  prepare_run_dir "$name"
  say "train dense $name"
  python -m experiments.train_dense_baseline --train "$tr" --val "$va" --run-name "$name" \
    --epochs 25 --hidden-dim 256 --num-layers 3 "$@"
}

# analysis NAME -- command...   runs the command unless NAME/summary.json exists
analysis() {
  local name="$1"; shift; [[ "$1" == "--" ]] && shift
  if done_run "$name"; then say "skip $name (summary.json exists)"; return; fi
  prepare_run_dir "$name"
  say "run $name"
  "$@"
}

# run_stage NAME DONE_FN BODY_FN : skip if DONE_FN succeeds, else run BODY_FN with its output
# tee'd to the stage log. pipefail + set -e make a failing body abort the whole script.
run_stage() {
  local name="$1" done_fn="$2" body_fn="$3"
  local log="$RUNS/_log_regenerate_${name}.log"
  if "$done_fn"; then say "STAGE $name: already complete -- skipping"; return; fi
  mkdir -p "$RUNS"
  say "STAGE $name: starting (log: $log)"
  local start=$SECONDS
  ( set -euo pipefail; "$body_fn" ) 2>&1 | tee -a "$log"
  say "STAGE $name: done in $(( (SECONDS - start) / 60 )) min" | tee -a "$log"
}

# ---------------------------------------------------------------------------------------
# input-data guards: every input below is present on disk as of 2026-10-12. If one goes
# missing, it is regenerated with the ORIGINAL seeded command before the stage uses it.
# (build_clean_splits rebuilds every cell it is given; the outputs are deterministic, so
# rebuilding a cell that already existed reproduces it.)
# ---------------------------------------------------------------------------------------
ensure_onsetplanar12k_splits() {
  local missing=0
  for N in 3 5 8; do for S in 0 1 2 3 4; do
    [[ -f "$T/splits_onsetplanar12k_${N}obj_s${S}/onsetplanar12k_${N}obj_s${S}_hard_test.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "planar onset 12k splits missing -- regenerating via onset_planar_pipeline.sh's commands"
  for N in 3 5 8; do for S in 0 1 2 3 4; do
    local out="$T/onsetplanar12k_${N}obj_s${S}.npz"
    [[ -f "$out" ]] && continue
    python -m experiments.generate_transitions --policy scripted --episodes 12000 --max-steps 100 \
      --num-objects "$N" --seed "$S" --object-bound 0.26 --min-object-separation 0.09 --env planar \
      --output "$out" --run-name "onsetplanar12k_${N}obj_s${S}"
  done; done
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2 3 4 \
    --filter-mode onset --min-max-xy-delta 0.010 \
    --input-template "$T/onsetplanar12k_{n}obj_s{seed}.npz" \
    --output-template "$T/splits_onsetplanar12k_{n}obj_s{seed}"
}

ensure_planar_motion_splits() {   # scale_series_pipeline.sh "3 5 8 12 20" "0 1 2 3 4" planar planar
  local missing=0
  for N in 3 5 8; do for S in 0 1 2; do
    [[ -f "$T/splits_clean_${N}obj_planar_s${S}/scale_${N}obj_planar_s${S}_hard_test.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "planar motion splits missing -- regenerating via scale_series_pipeline.sh's commands"
  for N in 3 5 8; do for S in 0 1 2; do
    local out="$T/scale_${N}obj_planar_s${S}.npz"
    [[ -f "$out" ]] && continue
    python -m experiments.generate_transitions --policy scripted --env planar --episodes 250 \
      --max-steps 100 --num-objects "$N" --seed "$S" --object-bound 0.26 --min-object-separation 0.09 \
      --output "$out" --run-name "scale_${N}obj_planar_s${S}"
  done; done
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2 --min-max-xy-delta 0.010 \
    --input-template "$T/scale_{n}obj_planar_s{seed}.npz" \
    --output-template "$T/splits_clean_{n}obj_planar_s{seed}"
}

ensure_dense_variant_splits() {   # RESULTS.md "Reproduce": VARIANT=dense run_count_seeds.sh, then clean splits
  local missing=0
  for N in 3 5 8; do for S in 0 1 2; do
    [[ -f "$T/splits_clean_${N}obj_dense_s${S}/scale_${N}obj_dense_s${S}_hard_test.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "dense-variant clean splits missing -- regenerating"
  for N in 3 5 8; do
    case $N in 3) G=(--object-bound 0.10 --min-object-separation 0.08);;
               5) G=(--object-bound 0.13 --min-object-separation 0.085);;
               8) G=(--object-bound 0.16 --min-object-separation 0.08);; esac
    for S in 0 1 2; do
      local out="$T/scale_${N}obj_dense_s${S}.npz"
      [[ -f "$out" ]] && continue
      python -m experiments.generate_transitions --policy scripted --episodes 250 --max-steps 100 \
        --num-objects "$N" --seed "$S" "${G[@]}" --output "$out" --run-name "scale_${N}obj_dense_s${S}"
    done
  done
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2 \
    --input-template "$T/scale_{n}obj_dense_s{seed}.npz" \
    --output-template "$T/splits_clean_{n}obj_dense_s{seed}"
}

ensure_onset_s34_splits() {       # iclr_closeout.sh stage 2b -> onset_benchmark_pipeline.sh "3 5 8" "3 4"
  local missing=0
  for N in 3 5 8; do for S in 3 4; do
    [[ -f "$T/splits_onset_${N}obj_s${S}/onset_${N}obj_s${S}_hard_train.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "onset splits for seeds 3-4 missing -- regenerating via onset_benchmark_pipeline.sh"
  bash experiments/onset_benchmark_pipeline.sh "3 5 8" "3 4"
}

ensure_clean_motion_splits() {    # build_clean_splits defaults = the published clean motion splits
  local missing=0
  for N in 3 5 8; do for S in 0 1 2; do
    [[ -f "$T/splits_clean_${N}obj_s${S}/scale_${N}obj_s${S}_hard_train.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "clean motion splits missing -- rebuilding (raw scale_{N}obj_s{S}.npz required)"
  for N in 3 5 8; do for S in 0 1 2; do require "$T/scale_${N}obj_s${S}.npz"; done; done
  python -m experiments.build_clean_splits --counts 3 5 8 --seeds 0 1 2
}

ensure_uni_splits() {             # scale_series_pipeline.sh "3 5 8 12 20" "0 1 2 3 4" (tabletop, tag uni)
  local missing=0
  for N in 3 5 8 12 20; do for S in 0 1 2 3 4; do
    [[ -f "$T/splits_clean_${N}obj_uni_s${S}/scale_${N}obj_uni_s${S}_hard_test.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "W4 uni splits missing -- regenerating via scale_series_pipeline.sh's commands"
  for N in 3 5 8 12 20; do for S in 0 1 2 3 4; do
    local out="$T/scale_${N}obj_uni_s${S}.npz"
    [[ -f "$out" ]] && continue
    python -m experiments.generate_transitions --policy scripted --env tabletop --episodes 250 \
      --max-steps 100 --num-objects "$N" --seed "$S" --object-bound 0.26 --min-object-separation 0.09 \
      --output "$out" --run-name "scale_${N}obj_uni_s${S}"
  done; done
  python -m experiments.build_clean_splits --counts 3 5 8 12 20 --seeds 0 1 2 3 4 \
    --min-max-xy-delta 0.020 \
    --input-template "$T/scale_{n}obj_uni_s{seed}.npz" \
    --output-template "$T/splits_clean_{n}obj_uni_s{seed}"
}

ensure_plan_mixed_splits() {      # RESULTS.md "Round 2" reproduce block
  [[ -f "$T/splits_plan_mixed_3obj/plan_mixed_3obj_train.npz" ]] && return 0
  say "plan_mixed splits missing -- regenerating via the Round 2 commands"
  [[ -f "$T/plan_scripted_3obj.npz" ]] || python -m experiments.generate_transitions --policy scripted \
    --episodes 200 --max-steps 80 --num-objects 3 --seed 0 --output "$T/plan_scripted_3obj.npz"
  [[ -f "$T/plan_random_3obj.npz" ]] || python -m experiments.generate_transitions --policy random \
    --episodes 350 --max-steps 60 --num-objects 3 --seed 1 --output "$T/plan_random_3obj.npz"
  [[ -f "$T/plan_mixed_3obj.npz" ]] || python -m experiments.concat_transitions \
    --inputs "$T/plan_scripted_3obj.npz" "$T/plan_random_3obj.npz" --output "$T/plan_mixed_3obj.npz"
  python -m experiments.split_dataset --input "$T/plan_mixed_3obj.npz" \
    --output-dir "$T/splits_plan_mixed_3obj" --seed 0
}

ensure_interaction_billiards_splits() {   # build_interaction_benchmark.py, 9 cells (log: _log_interaction_build.log)
  local missing=0
  for N in 3 5 8; do for S in 0 1 2; do
    [[ -f "$T/splits_interaction_billiards_${N}obj_s${S}/interaction_billiards_${N}obj_s${S}_hard_train.npz" ]] || missing=1
  done; done
  (( missing )) || return 0
  say "billiards interaction splits missing -- rebuilding (expensive: ~4.5M simulated steps per cell)"
  python -m experiments.build_interaction_benchmark --env billiards --counts 3 5 8 --seeds 0 1 2 \
    --run-name build_interaction_billiards
}

# ---------------------------------------------------------------------------------------
# STAGE interaction_stats (group 7) -- RESULTS.md "The third benchmark fails too"
# No script ever computed these; the numbers were an ad-hoc check. This recomputes them with
# the exact definitions compute_interaction_keep_mask uses (post-action pusher position,
# REST_SPEED, billiards threshold 0.031) and records several variants, because the original
# definition is not recorded. See the report: 0.061 reproduces, 0.151 / 94.4% do not exactly.
# ---------------------------------------------------------------------------------------
done_interaction_stats() { done_run interaction_construction_stats; }
stage_interaction_stats() {
  ensure_interaction_billiards_splits
  prepare_run_dir interaction_construction_stats
  python - <<'PY'
import json
from pathlib import Path
import numpy as np
from models.layout import StateLayout
from experiments.create_hard_subset import REST_SPEED, PUSHER_ACTION_SCALE, PUSHER_BOUND

TH = 0.031  # billiards motion threshold (domain_characterization.py)
out = Path("experiments/runs/interaction_construction_stats"); out.mkdir(parents=True, exist_ok=True)

def cell(n, s):
    acc = {k: [] for k in ("has_indirect", "indirect_d", "nearest_d", "nearest_changed",
                           "nearest_moving_at_t", "indirect_d_pre", "nearest_d_pre")}
    for split in ("train", "val", "test"):
        d = np.load(f"data/transitions/splits_interaction_billiards_{n}obj_s{s}/"
                    f"interaction_billiards_{n}obj_s{s}_hard_{split}.npz")
        L = StateLayout(num_objects=n); st = d["s_t"]
        vel = st[:, L.object_velocity_slice].reshape(-1, n, 6)
        rest = np.linalg.norm(vel[:, :, 3:5], axis=2) <= REST_SPEED
        changed = d["object_change_mask"] > 0.5
        xy = np.linalg.norm(d["object_delta"][:, :, :2], axis=2)
        onset = rest & changed & (xy >= TH)
        p = st[:, 0:2]; a = np.clip(d["a_t"], -1, 1)
        p_next = np.clip(p + a * PUSHER_ACTION_SCALE, -PUSHER_BOUND, PUSHER_BOUND)
        obj = st[:, L.object_pose_slice].reshape(-1, n, 3)[:, :, :2]
        dist = np.linalg.norm(obj - p_next[:, None, :], axis=2)
        dist_pre = np.linalg.norm(obj - p[:, None, :], axis=2)
        nearest = dist.argmin(1); r = np.arange(len(nearest))
        not_nearest = np.ones_like(onset); not_nearest[r, nearest] = False
        indirect = onset & not_nearest
        acc["has_indirect"].append(indirect.any(1)); acc["indirect_d"].append(dist[indirect])
        acc["indirect_d_pre"].append(dist_pre[indirect])
        acc["nearest_d"].append(dist[r, nearest]); acc["nearest_d_pre"].append(dist_pre[r, nearest])
        acc["nearest_changed"].append(changed[r, nearest]); acc["nearest_moving_at_t"].append(~rest[r, nearest])
    return {k: np.concatenate(v) for k, v in acc.items()}

def stats(cells):
    parts = [cell(n, s) for n, s in cells]
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return {
        "rows": int(c["has_indirect"].size),
        "fraction_rows_with_indirect_onset": float(c["has_indirect"].mean()),
        "median_indirect_mover_distance_to_actuator": float(np.median(c["indirect_d"])),
        "median_nearest_object_distance_to_actuator": float(np.median(c["nearest_d"])),
        "fraction_rows_nearest_object_also_moves": float(c["nearest_changed"].mean()),
        "variant_pre_action_pusher": {
            "median_indirect_mover_distance": float(np.median(c["indirect_d_pre"])),
            "median_nearest_object_distance": float(np.median(c["nearest_d_pre"])),
        },
        "variant_nearest_object_moving_at_t": float(c["nearest_moving_at_t"].mean()),
    }

summary = {
    "published": {"median_indirect": 0.151, "median_nearest": 0.061, "nearest_also_moves": 0.944},
    "all_9_cells_N3_5_8": stats([(n, s) for n in (3, 5, 8) for s in (0, 1, 2)]),
    "audited_cells_N5_8": stats([(n, s) for n in (5, 8) for s in (0, 1, 2)]),
    "per_cell": {f"{n}obj_s{s}": stats([(n, s)]) for n in (3, 5, 8) for s in (0, 1, 2)},
}
config = {"task": "interaction_construction_stats", "env": "billiards", "threshold": TH,
          "rest_speed": REST_SPEED, "splits": "data/transitions/splits_interaction_billiards_{n}obj_s{seed}",
          "definition": "indirect = at-rest & changed & |dxy|>=0.031 & not nearest to post-action pusher; "
                        "'also moves' = nearest object's object_change_mask"}
(out / "config.json").write_text(json.dumps(config, indent=2))
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("published", "all_9_cells_N3_5_8", "audited_cells_N5_8")}, indent=2))
PY
}

# ---------------------------------------------------------------------------------------
# STAGE dense_packed (group 2) -- RESULTS.md "Why it happens" -> audit_dense_packed
# Inputs: splits_clean_{N}obj_dense_s{0,1,2} (present). Gates sparse_clean_{N}obj_dense_s{S}.pt
# are the clean_pipeline.sh VARIANT=dense names (lost; retrained here, sparse half only -- the
# dense half of clean_pipeline.sh is not used by the audit). The contact gate on this variant
# never had a recorded name; it is trained under a NEW name so the audit can report both.
# Benchmark key "dense" is a reconstruction (the original key is not recorded).
# ---------------------------------------------------------------------------------------
done_dense_packed() { done_run audit_dense_packed; }
stage_dense_packed() {
  ensure_dense_variant_splits
  for N in 3 5 8; do for S in 0 1 2; do
    D="$T/splits_clean_${N}obj_dense_s${S}"; ST="scale_${N}obj_dense_s${S}_hard"
    train_sparse "sparse_clean_${N}obj_dense_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    train_sparse "sparse_contact_clean_${N}obj_dense_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --feature-mode contact --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done; done
  analysis audit_dense_packed -- python -m experiments.onset_shortcut_audit \
    --benchmarks dense --counts 3 5 8 --seeds 0 1 2 \
    --split-templates "dense=$T/splits_clean_{n}obj_dense_s{seed}/scale_{n}obj_dense_s{seed}_hard_{split}.npz" \
    --gate-templates "dense=$CK/sparse_clean_{n}obj_dense_s{seed}.pt" \
    --contact-gate-templates "dense=$CK/sparse_contact_clean_{n}obj_dense_s{seed}.pt" \
    --run-name audit_dense_packed
}

# ---------------------------------------------------------------------------------------
# STAGE lit_onset_s34 (group 8) -- iclr_closeout.sh stage 2d, verbatim.
# The 2026-08-18 run skipped onset_5obj_s4, onset_8obj_s3 and onset_8obj_s4 because those
# splits did not exist yet; they do now. Rerunning the ORIGINAL command under the ORIGINAL name
# fills the gap (3 counts x seeds 3,4). The previous dir is kept as *.pre_n8 for comparison.
# NOTE: make_iclr_tables.py / claims.py pool this run's CSV, so the published-model onset-bench
# means (Table 2, claims 0.166 / 0.388 / 0.519 / 0.884 / 0.919) can shift. Re-check them.
# ---------------------------------------------------------------------------------------
lit_s34_complete() {
  python - <<'PY'
import csv, sys
from pathlib import Path
p = Path("experiments/runs/literature_baselines_onset_s34/literature_baselines.csv")
if not p.exists(): sys.exit(1)
cells = {(r["object_count"], r["seed"]) for r in csv.DictReader(p.open())}
need = {(n, s) for n in ("3", "5", "8") for s in ("3", "4")}
sys.exit(0 if need <= cells else 1)
PY
}
done_lit_onset_s34() { lit_s34_complete; }
stage_lit_onset_s34() {
  ensure_onset_s34_splits
  local dir="$RUNS/literature_baselines_onset_s34"
  if [[ -d "$dir" && ! -d "${dir}.pre_n8" ]]; then cp -r "$dir" "${dir}.pre_n8"; say "backed up $dir -> ${dir}.pre_n8"; fi
  rm -f "$dir/summary.json" "$dir/metrics.csv"
  say "run literature_baselines_onset_s34 (counts 3 5 8, seeds 3 4)"
  python -m experiments.literature_baselines --counts 3 5 8 --seeds 3 4 \
    --split-template "$T/splits_onset_{n}obj_s{seed}/onset_{n}obj_s{seed}_hard_{split}.npz" \
    --run-name "literature_baselines_onset_s34"
  lit_s34_complete || { echo "literature_baselines_onset_s34 still lacks a (count, seed) cell" >&2; exit 1; }
}

# ---------------------------------------------------------------------------------------
# STAGE planar12k (group 1)
#  a) gates: onset_planar_pipeline.sh's sparse/contact commands (dense half not needed here)
#  b) audit_planar12k: the battery over 3 counts x 5 seeds (benchmark key "onsetplanar12k" is a
#     reconstruction following audit_onset_clutter12k's pattern)
#  c) momentum_shortcut_planar12k: the per-count velocity rule 0.085/0.143/0.243 table
#  d) momentum_shortcut_planar: the 0.990 "before the onset filter" number (motion-filtered
#     planar splits, 3 seeds). Its global gates sparse_planar_{N}obj_s{S} are the
#     scale_series_pipeline.sh planar-tag names; there was no contact gate, so the contact
#     template points at a name that does not exist (otherwise the script's DEFAULT would load
#     the TABLETOP contact gates onto planar data).
# ---------------------------------------------------------------------------------------
done_planar12k() { done_run audit_planar12k && done_run momentum_shortcut_planar12k && done_run momentum_shortcut_planar; }
stage_planar12k() {
  ensure_onsetplanar12k_splits
  ensure_planar_motion_splits
  for N in 3 5 8; do for S in 0 1 2 3 4; do
    D="$T/splits_onsetplanar12k_${N}obj_s${S}"; ST="onsetplanar12k_${N}obj_s${S}_hard"
    train_sparse "sparse_onsetplanar12k_${N}obj_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
    train_sparse "sparse_onsetplanar12k_contact_${N}obj_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --feature-mode contact --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done; done
  analysis audit_planar12k -- python -m experiments.onset_shortcut_audit \
    --benchmarks onsetplanar12k --counts 3 5 8 --seeds 0 1 2 3 4 \
    --split-templates "onsetplanar12k=$T/splits_onsetplanar12k_{n}obj_s{seed}/onsetplanar12k_{n}obj_s{seed}_hard_{split}.npz" \
    --gate-templates "onsetplanar12k=$CK/sparse_onsetplanar12k_{n}obj_s{seed}.pt" \
    --contact-gate-templates "onsetplanar12k=$CK/sparse_onsetplanar12k_contact_{n}obj_s{seed}.pt" \
    --run-name audit_planar12k
  analysis momentum_shortcut_planar12k -- python -m experiments.momentum_shortcut \
    --counts 3 5 8 --seeds 0 1 2 3 4 \
    --split-template "$T/splits_onsetplanar12k_{n}obj_s{seed}" --stem-template "onsetplanar12k_{n}obj_s{seed}" \
    --global-template "$CK/sparse_onsetplanar12k_{n}obj_s{seed}.pt" \
    --contact-template "$CK/sparse_onsetplanar12k_contact_{n}obj_s{seed}.pt" \
    --run-name momentum_shortcut_planar12k
  for N in 3 5 8; do for S in 0 1 2; do
    D="$T/splits_clean_${N}obj_planar_s${S}"; ST="scale_${N}obj_planar_s${S}_hard"
    train_sparse "sparse_planar_${N}obj_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done; done
  analysis momentum_shortcut_planar -- python -m experiments.momentum_shortcut \
    --counts 3 5 8 --seeds 0 1 2 \
    --split-template "$T/splits_clean_{n}obj_planar_s{seed}" --stem-template "scale_{n}obj_planar_s{seed}" \
    --global-template "$CK/sparse_planar_{n}obj_s{seed}.pt" \
    --contact-template "$CK/__no_planar_motion_contact_gate__{n}_{seed}.pt" \
    --run-name momentum_shortcut_planar
}

# ---------------------------------------------------------------------------------------
# STAGE domchar (group 3) -- domain_characterization.py with its defaults (4 domains x 3 counts
# x 3 seeds x 60 episodes, bound 0.26 / sep 0.09, scripted). Matches _log_domchar.log, which
# still holds the full original summary for comparison.
# ---------------------------------------------------------------------------------------
done_domchar() { done_run domain_characterization; }
stage_domchar() {
  analysis domain_characterization -- python -m experiments.domain_characterization \
    --domains tabletop planar billiards clutter --counts 3 5 8 --seeds 0 1 2 --episodes 60 \
    --max-steps 100 --object-bound 0.26 --min-object-separation 0.09 --policy scripted \
    --run-name domain_characterization
}

# ---------------------------------------------------------------------------------------
# STAGE w3_validity (group 4) -- counterfactual_augmentation.py --stages validity.
# Geometry per close_coverage_gaps.sh / the script's own --object-bound help: N=3,5 use the env
# default (the published scale_{3,5}obj data's geometry); N=8 needs 0.22 / 0.09 or placement
# fails at reset (that is exactly the failure in w3_batch.log; w3_8obj.log is the fixed rerun).
# Gate: models/checkpoints/sparse_clean_{N}obj_s{S}.pt (present).
# ---------------------------------------------------------------------------------------
done_w3_validity() { for N in 3 5 8; do for S in 0 1 2; do done_run "w3_validity_${N}obj_s${S}" || return 1; done; done; }
stage_w3_validity() {
  ensure_clean_motion_splits
  for N in 3 5 8; do
    if [[ "$N" == 8 ]]; then G=(--object-bound 0.22 --min-object-separation 0.09); else G=(); fi
    for S in 0 1 2; do
      require "$CK/sparse_clean_${N}obj_s${S}.pt"
      analysis "w3_validity_${N}obj_s${S}" -- python -m experiments.counterfactual_augmentation \
        --count "$N" --seed "$S" --stages validity "${G[@]}" --run-name "w3_validity_${N}obj_s${S}"
    done
  done
  python - <<'PY'
import json, numpy as np
print("W3 per-object validity (mean over seeds 0-2) -- published: gate .949/.895/.887, oracle .951/.896/.880, none .867/.756/.713")
for n in (3, 5, 8):
    v = [json.load(open(f"experiments/runs/w3_validity_{n}obj_s{s}/summary.json"))["validity"] for s in (0, 1, 2)]
    print(n, {c: round(float(np.mean([x[c]["per_object_stationary_fraction"] for x in v])), 3)
              for c in ("learned_gate", "oracle_mask", "no_mask")})
PY
}

# ---------------------------------------------------------------------------------------
# STAGE w4_scale (group 5) -- scale_series_pipeline.sh "3 5 8 12 20" "0 1 2 3 4" (tabletop, tag
# uni), TRAIN section only (generation + splits survived), then scale_analysis.py defaults.
# ---------------------------------------------------------------------------------------
done_w4_scale() { done_run scale_analysis; }
stage_w4_scale() {
  ensure_uni_splits
  for N in 3 5 8 12 20; do for S in 0 1 2 3 4; do
    D="$T/splits_clean_${N}obj_uni_s${S}"; ST="scale_${N}obj_uni_s${S}_hard"
    train_dense "dense_uni_${N}obj_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" --seed "$S"
    train_sparse "sparse_uni_${N}obj_s${S}" "$D/${ST}_train.npz" "$D/${ST}_val.npz" \
      --epochs 15 --sparsity-weight 0.2 --auto-balance-bce --seed "$S"
  done; done
  analysis scale_analysis -- python -m experiments.scale_analysis \
    --counts 3 5 8 12 20 --seeds 0 1 2 3 4 --tag uni --run-name scale_analysis
}

# ---------------------------------------------------------------------------------------
# STAGE planning (group 6)
#  a) published baselines, training seed 0: iclr_closeout.sh stage 3 with --seeds 0
#     (_log_lit_planmixed.log is the original seed-0 output; compare its F1 lines)
#  b) the three forward models of the seed-0 table, Round 2 recipe (RESULTS.md "Reproduce"):
#     sparse_contact_3obj_v1, dense_mixed_3obj_v1, latent_plan_3obj_s0 (train_latent_dynamics
#     docstring). The sparse/dense checkpoint choice is INFERRED (the seed-0 command was not
#     recorded): dense and latent reproduce W5's 0.331 / 0.327 final distance exactly, which
#     pins those; "sparse (contact) ... trained on the same planner-distribution data" pins
#     sparse_contact_3obj_v1. Seed-0 sparse = 0.250 / 0.156 confirms or refutes it.
#  c) planning_literature_3obj: all ten conditions, held-out base seed 5000, 20 eps, 60 steps
#  d) NEW: the sparse model at training seeds 1, 2 (sparse_contact_3obj_s{1,2}, RESULTS.md's
#     seed recipe) under the identical protocol, so sparse can be pooled over 3 seeds like PETS.
# ---------------------------------------------------------------------------------------
PLAN_ARGS=(--num-objects 3 --num-episodes 20 --base-seed 5000 --max-steps 60
           --num-samples 256 --cem-iters 3 --horizon 15)
PM="$T/splits_plan_mixed_3obj/plan_mixed_3obj"
done_planning() { done_run planning_literature_3obj && done_run planning_sparse_contact_3obj_heldout_s1 \
                  && done_run planning_sparse_contact_3obj_heldout_s2; }
stage_planning() {
  ensure_plan_mixed_splits
  local have_lit=1
  for m in gns cswm slotformer pets nps; do [[ -f "$CK/litplan_${m}_3obj_s0.pt" ]] || have_lit=0; done
  if (( have_lit )) && done_run literature_baselines_planmixed; then
    say "skip literature_baselines_planmixed (seed-0 litplan checkpoints exist)"
  else
    prepare_run_dir literature_baselines_planmixed
    say "train published baselines on plan_mixed, seed 0"
    python -m experiments.literature_baselines --counts 3 --seeds 0 \
      --split-template "$T/splits_plan_mixed_3obj/plan_mixed_3obj_{split}.npz" \
      --feature-mode contact --epochs 25 \
      --checkpoint-dir "$CK" --checkpoint-tag litplan \
      --run-name literature_baselines_planmixed
  fi

  train_sparse sparse_contact_3obj_v1 "${PM}_train.npz" "${PM}_val.npz" \
    --feature-mode contact --epochs 25 --sparsity-weight 0.05 --auto-balance-bce --seed 0
  train_dense dense_mixed_3obj_v1 "${PM}_train.npz" "${PM}_val.npz" --seed 0
  if done_run latent_plan_3obj_s0 && [[ -f "$CK/latent_plan_3obj_s0.pt" ]]; then
    say "skip train latent_plan_3obj_s0 (exists)"
  else
    prepare_run_dir latent_plan_3obj_s0
    say "train latent latent_plan_3obj_s0"
    python -m experiments.train_latent_dynamics --train "${PM}_train.npz" --val "${PM}_val.npz" \
      --run-name latent_plan_3obj_s0 --seed 0
  fi

  require "$CK/sparse_contact_3obj_v1.pt" "$CK/dense_mixed_3obj_v1.pt" "$CK/latent_plan_3obj_s0.pt"
  for m in gns cswm slotformer pets nps; do require "$CK/litplan_${m}_3obj_s0.pt"; done
  analysis planning_literature_3obj -- python -m experiments.planning_mpc \
    --conditions sparse dense latent gns cswm slotformer pets nps scripted random \
    --sparse-checkpoint "$CK/sparse_contact_3obj_v1.pt" \
    --dense-checkpoint "$CK/dense_mixed_3obj_v1.pt" \
    --latent-checkpoint "$CK/latent_plan_3obj_s0.pt" \
    --literature-checkpoint-dir "$CK" --literature-tag litplan --literature-seed 0 \
    "${PLAN_ARGS[@]}" --run-name planning_literature_3obj

  for S in 1 2; do
    train_sparse "sparse_contact_3obj_s${S}" "${PM}_train.npz" "${PM}_val.npz" \
      --feature-mode contact --epochs 25 --sparsity-weight 0.05 --auto-balance-bce --seed "$S"
    analysis "planning_sparse_contact_3obj_heldout_s${S}" -- python -m experiments.planning_mpc \
      --conditions sparse --sparse-checkpoint "$CK/sparse_contact_3obj_s${S}.pt" \
      "${PLAN_ARGS[@]}" --run-name "planning_sparse_contact_3obj_heldout_s${S}"
  done

  python - <<'PY'
import json
s0 = json.load(open("experiments/runs/planning_literature_3obj/summary.json"))
def rate(run, cond):
    return json.load(open(f"experiments/runs/{run}/summary.json"))["conditions"][cond]["success_rate"]
print("seed 0 (published: pets .35 sparse .25 slotformer .15 nps .15 gns .10 dense 0 cswm 0 latent 0):")
print({c: v["success_rate"] for c, v in s0["conditions"].items()})
sp = [s0["conditions"]["sparse"]["success_rate"]] + [rate(f"planning_sparse_contact_3obj_heldout_s{s}", "sparse") for s in (1, 2)]
print("sparse over training seeds 0/1/2:", sp, "mean", sum(sp) / 3, "(PETS pooled 0.300)")
PY
}

# ---------------------------------------------------------------------------------------
say "regenerate_missing_runs: stages = ${STAGES[*]}"
for s in "${STAGES[@]}"; do
  run_stage "$s" "done_${s}" "stage_${s}"
done
say "ALL REQUESTED STAGES COMPLETE"
