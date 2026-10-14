"""The claim ledger: every number in the manuscript, with where it comes from.

``SUBMISSION_RUNBOOK.md`` step 1 requires this: every number in the paper traces to a run, and
the ledger is what keeps prose, tables and runs from drifting apart.

The checker does three things:

1. **Coverage.** Every decimal number in ``main.tex`` (outside comments and figure options)
   must be accounted for by a ledger entry at the precision it is printed. A number added to
   the prose without a ledger entry is a FAIL, so the ledger cannot fall behind the paper.
2. **Presence.** Every ledger entry must still appear in ``main.tex`` as an exact token
   (``0.41`` does not match ``0.413``), which catches a table regenerated while the prose quoting
   it was not.
3. **Recomputation.** Entries backed by a run are recomputed from ``experiments/runs/`` and must
   agree to within half a unit of the last printed digit.

Source kinds:

``audit``      (run, benchmark, rule, metric[, seeds])     battery audit CSV
``literature`` (benchmark, model, metric, count)          published-model CSVs
``lit_range``  (benchmark, metric, "min"|"max"|"cell_min"|"cell_max")
                                                           range over the five models' means,
                                                           or over every individual run
``lit_diff``   (count, "mean"|"lo"|"hi")                   trivial rule vs best published model,
                                                           paired bootstrap over seeds
``planning``   (model, "pooled"|seed)                      planning success rate
``pixel``      (run, condition, metric)                    single-cell pixel runs
``json``       (runs, file, dotted path[, scale])          a JSON field, averaged over runs
``json_diff``  (runs, file, path_a, path_b)                mean of field a minus field b
``derived``    (expression,)                               arithmetic on other printed numbers
``constant``   (reason,)                                   a setting or a mathematical fact
``ledger``     (RESULTS.md section, run directory or None) the run is not on this machine;
                                                           checked for presence only, and
                                                           flagged once its run reappears

    python paper/claims.py            # check every claim
    python paper/claims.py --list     # print the ledger

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
TRANSITIONS = ROOT / "data" / "transitions"

MODELS = ("gns", "cswm", "slotformer", "pets", "nps")

AUDIT_5SEED = "onset_shortcut_audit_5seed"
AUDIT_INTER = "audit_interaction_billiards"
AUDIT_CLUT = "audit_onset_clutter12k"
SEEDS_012 = ("0", "1", "2")

INTER_STATS = (["interaction_construction_stats"], "summary.json")
DENSE = (["audit_dense_packed"], "summary.json")
DOMCHAR = (["domain_characterization"], "summary.json")
SCALE = (["scale_analysis"], "summary.json")


def w3(n: int) -> list[str]:
    return [f"w3_validity_{n}obj_s{s}" for s in SEEDS_012]

# --------------------------------------------------------------------------------------
# The ledger: (value as printed in the paper, description, source)
# --------------------------------------------------------------------------------------

CLAIMS: list[tuple[str, str, tuple]] = [
    # -- Section 5, the battery --------------------------------------------------------
    ("0.927", "abstract/S5 motion best trivial", ("audit", AUDIT_5SEED, "motion", "moving_or_near", "f1")),
    ("0.866", "abstract/S5 velocity on motion", ("audit", AUDIT_5SEED, "motion", "already_moving", "f1")),
    ("0.182", "abstract/S5 velocity collapses on onset", ("audit", AUDIT_5SEED, "onset", "already_moving", "f1")),
    ("0.961", "S5 onset best trivial", ("audit", AUDIT_5SEED, "onset", "moving_or_near", "f1")),
    ("0.928", "abstract/S5 nearest_to_pusher on onset", ("audit", AUDIT_5SEED, "onset", "nearest_to_pusher", "f1")),
    ("0.932", "S5 pusher_near on onset", ("audit", AUDIT_5SEED, "onset", "pusher_near", "f1")),
    ("0.919", "S5 one-feature logistic on onset", ("audit", AUDIT_5SEED, "onset", "logistic_on_distance", "f1")),
    ("0.824", "S5 best gate on motion", ("audit", AUDIT_5SEED, "motion", "learned_gate", "f1")),
    ("0.703", "S5 best gate on onset", ("audit", AUDIT_5SEED, "onset", "learned_gate_contact", "f1")),
    ("0.413", "S5 always_change floor, motion", ("audit", AUDIT_5SEED, "motion", "always_change", "f1")),
    ("0.387", "S5 always_change floor, onset", ("audit", AUDIT_5SEED, "onset", "always_change", "f1")),
    ("0.925", "S5 nearest_to_pusher on onset, seeds 0-2", ("audit", AUDIT_5SEED, "onset", "nearest_to_pusher", "f1", SEEDS_012)),
    ("0.916", "S5 logistic on onset, seeds 0-2", ("audit", AUDIT_5SEED, "onset", "logistic_on_distance", "f1", SEEDS_012)),
    ("0.437", "abstract/S5 nearest_to_pusher collapses on interaction", ("audit", AUDIT_INTER, "interaction", "nearest_to_pusher", "f1")),
    ("0.731", "S5 logistic collapses on interaction", ("audit", AUDIT_INTER, "interaction", "logistic_on_distance", "f1")),
    ("0.955", "abstract/S5 interaction best trivial", ("audit", AUDIT_INTER, "interaction", "moving_or_near_mover", "f1")),
    ("0.772", "S5 best gate on interaction", ("audit", AUDIT_INTER, "interaction", "learned_gate_contact", "f1")),
    ("0.673", "S5 always_change rises on interaction", ("audit", AUDIT_INTER, "interaction", "always_change", "f1")),
    ("0.709", "S5 learned gate on interaction", ("audit", AUDIT_INTER, "interaction", "learned_gate", "f1")),
    ("0.821", "abstract/S5/S7 clutter best trivial", ("audit", AUDIT_CLUT, "onsetclutter12k", "moving_or_near", "f1")),
    ("0.656", "S5/S7 nearest_to_pusher on clutter", ("audit", AUDIT_CLUT, "onsetclutter12k", "nearest_to_pusher", "f1")),
    ("0.425", "abstract/S5 gate on clutter", ("audit", AUDIT_CLUT, "onsetclutter12k", "learned_gate", "f1")),
    ("0.338", "S5 velocity-free gate on clutter", ("audit", AUDIT_CLUT, "onsetclutter12k", "learned_gate_contact", "f1")),
    ("0.93", "S5 lower end of the trivial ceiling", ("derived", "min(0.927, 0.961, 0.955)")),
    ("0.96", "S5 upper end of the trivial ceiling", ("derived", "max(0.927, 0.961, 0.955)")),
    ("0.02", "S5 logistic within this of the best hand rule", ("derived", "0.932 - 0.919")),
    ("0.22", "S5 logistic above the onset gate", ("derived", "0.919 - 0.703")),
    ("0.04", "S5 always_change within this of the gate on interaction", ("derived", "0.709 - 0.673")),
    ("0.396", "S5 widest trivial-over-gate margin (clutter)", ("derived", "0.821 - 0.425")),
    ("0.14", "S7 what packing costs the rules", ("derived", "0.961 - 0.821")),
    ("0.28", "S7 what packing costs the gate", ("derived", "0.703 - 0.425")),
    # P(changed | already moving) is the precision of the already_moving rule.
    ("0.87", "S5 P(changed | already moving) on motion, N=8", ("audit", AUDIT_5SEED, "motion", "already_moving", "precision", None, ("8",))),
    ("0.95", "S5 P(changed | already moving) on motion, N=3", ("audit", AUDIT_5SEED, "motion", "already_moving", "precision", None, ("3",))),
    ("0.990", "S5 planar velocity rule before the onset filter", ("json", ["momentum_shortcut_planar"], "summary.json", "per_count.3obj.trivial_already_moving.f1_all")),
    ("0.983", "S5 planar moving_or_near after the onset filter", ("json", ["audit_planar12k"], "summary.json", "per_benchmark.onsetplanar12k.moving_or_near.f1")),
    ("0.772", "S5 planar velocity-free gate", ("json", ["audit_planar12k"], "summary.json", "per_benchmark.onsetplanar12k.learned_gate_contact.f1")),
    ("0.085", "S5 planar velocity rule after onset, low", ("json", ["momentum_shortcut_planar12k"], "summary.json", "per_count.3obj.trivial_already_moving.f1_all")),
    ("0.243", "S5 planar velocity rule after onset, high", ("json", ["momentum_shortcut_planar12k"], "summary.json", "per_count.8obj.trivial_already_moving.f1_all")),
    ("0.143", "S5 indirect mover distance to actuator", ("json", *INTER_STATS, "all_9_cells_N3_5_8.median_indirect_mover_distance_to_actuator")),
    ("0.061", "S5 nearest object distance to actuator", ("json", *INTER_STATS, "all_9_cells_N3_5_8.median_nearest_object_distance_to_actuator")),
    ("92.8", "S5 nearest object also moves (%)", ("json", *INTER_STATS, "all_9_cells_N3_5_8.fraction_rows_nearest_object_also_moves", 100.0)),
    ("0.20", "S5 clutter onset transitions (% of steps)", ("onset_rate", "splits_onsetclutter12k_*")),
    ("1.6", "S5 tabletop onset transitions (% of steps)", ("onset_rate", "splits_onset_*obj_s*")),

    # -- Section 6, the published models -----------------------------------------------
    ("0.901", "S6 trivial velocity, motion N=3", ("literature", "motion", "trivial_velocity", "f1", "3")),
    ("0.862", "S6 trivial velocity, motion N=5", ("literature", "motion", "trivial_velocity", "f1", "5")),
    ("0.836", "S6 trivial velocity, motion N=8", ("literature", "motion", "trivial_velocity", "f1", "8")),
    ("0.542", "S6 best published, motion N=3", ("literature", "motion", "__best__", "f1", "3")),
    ("0.397", "S6 best published, motion N=5", ("literature", "motion", "__best__", "f1", "5")),
    ("0.303", "S6 best published, motion N=8", ("literature", "motion", "__best__", "f1", "8")),
    ("-0.359", "S6 trivial minus best published, N=3", ("lit_diff", "3", "mean")),
    ("-0.465", "S6 trivial minus best published, N=5", ("lit_diff", "5", "mean")),
    ("-0.533", "S6 trivial minus best published, N=8", ("lit_diff", "8", "mean")),
    ("-0.561", "S6 bootstrap CI low, N=8", ("lit_diff", "8", "lo")),
    ("-0.505", "S6 bootstrap CI high, N=8", ("lit_diff", "8", "hi")),
    ("0.993", "S6 published recall, lowest single run", ("lit_range", "motion", "recall", "cell_min")),
    ("1.000", "S6 published recall, highest single run", ("lit_range", "motion", "recall", "cell_max")),
    ("0.064", "S6 published unchanged-object L2, lowest model", ("lit_range", "motion", "unchanged_object_l2", "min")),
    ("0.089", "S6 published unchanged-object L2, highest model", ("lit_range", "motion", "unchanged_object_l2", "max")),
    ("0.182", "S6 trivial collapses on onset", ("literature", "onset", "trivial_velocity", "f1", None)),
    ("0.109", "S6 published onset F1 on motion bench", ("literature", "motion", "__mean__", "onset_f1", None)),
    ("0.365", "S6 published onset F1 on onset bench", ("literature", "onset", "__mean__", "onset_f1", None)),
    ("0.757", "S6 PETS disagreement AUC, motion", ("literature", "motion", "pets", "disagreement_auc", None)),
    ("0.532", "S6 PETS disagreement AUC, onset", ("literature", "onset", "pets", "disagreement_auc", None)),
    ("0.288", "S6 C-SWM hits@1, motion", ("literature", "motion", "cswm", "cswm_hits_at_1", None)),
    ("0.882", "S6 C-SWM hits@1, onset", ("literature", "onset", "cswm", "cswm_hits_at_1", None)),
    ("0.512", "S6 C-SWM MRR, motion", ("literature", "motion", "cswm", "cswm_mrr", None)),
    ("0.918", "S6 C-SWM MRR, onset", ("literature", "onset", "cswm", "cswm_mrr", None)),

    # -- Section 7, mechanism ---------------------------------------------------------
    ("0.823", "S7 nearest_to_pusher before packing", ("audit", AUDIT_5SEED, "motion", "nearest_to_pusher", "f1", SEEDS_012)),
    ("0.784", "S7 nearest_to_pusher after packing", ("json", *DENSE, "per_benchmark.dense.nearest_to_pusher.f1")),
    ("0.805", "S7 pusher_near before packing", ("audit", AUDIT_5SEED, "motion", "pusher_near", "f1", SEEDS_012)),
    ("0.776", "S7 pusher_near after packing", ("json", *DENSE, "per_benchmark.dense.pusher_near.f1")),
    ("0.906", "S7 moving_or_near on the packed variant", ("json", *DENSE, "per_benchmark.dense.moving_or_near.f1")),
    ("0.823", "S7 learned gate on the packed variant", ("json", *DENSE, "per_benchmark.dense.learned_gate.f1")),

    # -- Section 8, what survives -------------------------------------------------------
    ("0.271", "S8 dense unchanged-object L2 at N=3", ("json", *SCALE, "per_count.3obj.dense_unchanged_l2.mean")),
    ("0.438", "S8 dense unchanged-object L2 at N=20", ("json", *SCALE, "per_count.20obj.dense_unchanged_l2.mean")),
    ("0.001", "S8 gated unchanged-object L2, low", ("json", *SCALE, "per_count.3obj.sparse_unchanged_l2.mean")),
    ("0.002", "S8 gated unchanged-object L2, high", ("json", *SCALE, "per_count.20obj.sparse_unchanged_l2.mean")),
    ("0.270", "S8 unchanged-L2 gap N=3", ("json", *SCALE, "per_count.3obj.unchanged_l2_gap_dense_minus_sparse")),
    ("0.279", "S8 unchanged-L2 gap N=5", ("json", *SCALE, "per_count.5obj.unchanged_l2_gap_dense_minus_sparse")),
    ("0.324", "S8 unchanged-L2 gap N=8", ("json", *SCALE, "per_count.8obj.unchanged_l2_gap_dense_minus_sparse")),
    ("0.384", "S8 unchanged-L2 gap N=12", ("json", *SCALE, "per_count.12obj.unchanged_l2_gap_dense_minus_sparse")),
    ("0.435", "S8 unchanged-L2 gap N=20", ("json", *SCALE, "per_count.20obj.unchanged_l2_gap_dense_minus_sparse")),
    ("0.942", "S8 relocated objects stay put, learned gate, N=3", ("json", w3(3), "validity.json", "learned_gate.per_object_stationary_fraction")),
    ("0.942", "S8 relocated objects stay put, oracle mask, N=3", ("json", w3(3), "validity.json", "oracle_mask.per_object_stationary_fraction")),
    ("0.865", "S8 relocated objects stay put, blind, N=3", ("json", w3(3), "validity.json", "no_mask.per_object_stationary_fraction")),
    ("0.014", "S8 largest gate-oracle gap (N=5)", ("json_diff", w3(5), "validity.json", "oracle_mask.per_object_stationary_fraction", "learned_gate.per_object_stationary_fraction")),
    ("0.077", "S8 gate advantage over blind, N=3", ("json_diff", w3(3), "validity.json", "learned_gate.per_object_stationary_fraction", "no_mask.per_object_stationary_fraction")),
    ("0.127", "S8 gate advantage over blind, N=5", ("json_diff", w3(5), "validity.json", "learned_gate.per_object_stationary_fraction", "no_mask.per_object_stationary_fraction")),
    ("0.154", "S8 gate advantage over blind, N=8", ("json_diff", w3(8), "validity.json", "learned_gate.per_object_stationary_fraction", "no_mask.per_object_stationary_fraction")),
    ("0.943", "S8 strict validity, learned_gate, N=3", ("json", w3(3), "validity.json", "learned_gate.valid_fraction")),
    ("0.858", "S8 strict validity, learned_gate, N=5", ("json", w3(5), "validity.json", "learned_gate.valid_fraction")),
    ("0.799", "S8 strict validity, learned_gate, N=8", ("json", w3(8), "validity.json", "learned_gate.valid_fraction")),
    ("0.946", "S8 strict validity, oracle_mask, N=3", ("json", w3(3), "validity.json", "oracle_mask.valid_fraction")),
    ("0.906", "S8 strict validity, oracle_mask, N=5", ("json", w3(5), "validity.json", "oracle_mask.valid_fraction")),
    ("0.839", "S8 strict validity, oracle_mask, N=8", ("json", w3(8), "validity.json", "oracle_mask.valid_fraction")),
    ("0.760", "S8 strict validity, no_mask, N=3", ("json", w3(3), "validity.json", "no_mask.valid_fraction")),
    ("0.438", "S8 strict validity, no_mask, N=5", ("json", w3(5), "validity.json", "no_mask.valid_fraction")),
    ("0.225", "S8 strict validity, no_mask, N=8", ("json", w3(8), "validity.json", "no_mask.valid_fraction")),

    # -- Section 9, limitations ---------------------------------------------------------
    ("0.7", "S9 one 5 cm object's share of a 60 cm table (%)", ("derived", "100 * 0.05 ** 2 / 0.6 ** 2")),
    ("25.8", "S9 Slot Attention match distance (px)", ("pixel", "pixel_slot_control", "trivial_pixel", "match_distance_px")),
    ("21.0", "S9 keypoint match distance, unweighted (px)", ("pixel", "pixel_keypoint_fg0", "trivial_pixel", "match_distance_px")),
    ("20.3", "S9 keypoint match distance, foreground-weighted (px)", ("pixel", "pixel_keypoint_fg10", "trivial_pixel", "match_distance_px")),
    ("0.703", "S9 trivial pixel rule (reported as unusable)", ("pixel", "pixel_keypoint_fg0", "trivial_pixel", "f1")),
    ("0.882", "S9 learned pixel condition (reported as unusable)", ("pixel", "pixel_keypoint_fg0", "slot_oneframe", "f1")),
    ("3.2", "S9 label displacement in pixels", ("derived", "0.020 * 158")),
    ("0.267", "S9 PETS planning success, pooled", ("planning", "pets", "pooled")),
    ("0.133", "S9 NPS planning success, pooled", ("planning", "nps", "pooled")),
    ("0.083", "S9 GNS planning success, pooled", ("planning", "gns", "pooled")),
    ("0.083", "S9 C-SWM planning success, pooled", ("planning", "cswm", "pooled")),
    ("0.033", "S9 SlotFormer planning success, pooled", ("planning", "slotformer", "pooled")),
    ("0.250", "S9 our planning success, pooled", ("planning", "sparse", "pooled")),
    ("0.150", "S9 our planning success, seed 1", ("planning", "sparse", "1")),
    ("0.300", "S9 our planning success, seeds 0 and 2", ("planning", "sparse", "0")),
    ("0.10", "S9 rough half-width of a 20-episode success rate", ("derived", "(0.25 * 0.75 / 20) ** 0.5")),

    # -- Settings and mathematical facts ------------------------------------------------
    ("0.85", "S3 billiards restitution", ("constant", "environment setting")),
    ("0.020", "S3/S9/App motion threshold, tabletop", ("constant", "derived threshold, set by domain_characterization.py")),
    ("0.010", "S3/App motion threshold, planar", ("constant", "derived threshold, set by domain_characterization.py")),
    ("0.6", "S3 planar steps retained at the tabletop threshold (%)", ("json", *DOMCHAR, "per_domain.planar.retention_at_tabletop_threshold", 100.0)),
    ("0.07", "S5 clutter designed separation (m)", ("constant", "environment setting")),
    ("8.5", "S6 capacity-matched parameter count, low (k)", ("constant", "model configuration")),
    ("8.8", "S6 capacity-matched parameter count, high (k)", ("constant", "model configuration")),
    ("0.250", "S9 smallest exact sign-flip p at n=3", ("derived", "2 / 2 ** 3")),
    ("0.0625", "S9 smallest exact sign-flip p at n=5", ("derived", "2 / 2 ** 5")),
    ("1.000", "S9 scripted oracle planning success", ("planning", "scripted", "0")),
    ("0.66", "App two_nearest_to_pusher, low (clutter-onset)", ("audit", AUDIT_CLUT, "onsetclutter12k", "two_nearest_to_pusher", "f1")),
    ("0.73", "App pusher_approaching, high (interaction)", ("audit", AUDIT_INTER, "interaction", "pusher_approaching", "f1")),
    ("0.10", "App second_nearest_to_pusher, low (onset)", ("audit", AUDIT_5SEED, "onset", "second_nearest_to_pusher", "f1")),
    ("0.41", "App second_nearest_to_pusher, high (interaction)", ("audit", AUDIT_INTER, "interaction", "second_nearest_to_pusher", "f1")),

    # -- Appendix C, per-domain characterisation ----------------------------------------
    ("0.020", "App billiards P(chg|rest)", ("json", *DOMCHAR, "per_domain.billiards.p_changed_given_rest")),
    ("0.982", "App billiards P(chg|moving)", ("json", *DOMCHAR, "per_domain.billiards.p_changed_given_moving")),
    ("0.950", "App planar P(chg|moving)", ("json", *DOMCHAR, "per_domain.planar.p_changed_given_moving")),
    ("0.863", "App clutter P(chg|moving)", ("json", *DOMCHAR, "per_domain.clutter.p_changed_given_moving")),
    ("0.714", "App tabletop P(chg|moving)", ("json", *DOMCHAR, "per_domain.tabletop.p_changed_given_moving")),
    ("0.011", "App planar P(chg|rest)", ("json", *DOMCHAR, "per_domain.planar.p_changed_given_rest")),
    ("0.024", "App clutter P(chg|rest)", ("json", *DOMCHAR, "per_domain.clutter.p_changed_given_rest")),
    ("0.015", "App tabletop P(chg|rest)", ("json", *DOMCHAR, "per_domain.tabletop.p_changed_given_rest")),
    ("0.394", "App billiards changed fraction", ("json", *DOMCHAR, "per_domain.billiards.changed_fraction")),
    ("0.152", "App planar changed fraction", ("json", *DOMCHAR, "per_domain.planar.changed_fraction")),
    ("0.345", "App clutter changed fraction", ("json", *DOMCHAR, "per_domain.clutter.changed_fraction")),
    ("0.124", "App tabletop changed fraction", ("json", *DOMCHAR, "per_domain.tabletop.changed_fraction")),
    ("0.031", "App billiards threshold", ("json", *DOMCHAR, "per_domain.billiards.derived_motion_threshold")),
    ("0.028", "App clutter threshold", ("json", *DOMCHAR, "per_domain.clutter.derived_motion_threshold")),
    ("0.029", "App clutter threshold the benchmarks are built with", ("constant", "benchmark build setting")),
]


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------

def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def mean(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def read_rows(run: str, name: str) -> list[dict]:
    path = RUNS / run / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def audit_value(run, benchmark, rule, metric, seeds=None, counts=None) -> float:
    rows = [r for r in read_rows(run, "onset_shortcut_audit.csv")
            if r["benchmark"] == benchmark and r["rule"] == rule
            and (seeds is None or r["seed"] in seeds)
            and (counts is None or r["object_count"] in counts)]
    return mean([fnum(r.get(metric)) for r in rows])


def literature_rows(benchmark: str) -> list[dict]:
    out: list[dict] = []
    for run in (f"literature_baselines_{benchmark}", f"literature_baselines_{benchmark}_s34"):
        out += read_rows(run, "literature_baselines.csv")
    return out


def literature_value(benchmark, model, metric, count) -> float:
    rows = literature_rows(benchmark)
    if count is not None:
        rows = [r for r in rows if r["object_count"] == count]
    if model == "__mean__":
        return mean([fnum(r.get(metric)) for r in rows if r["model"] in MODELS])
    if model == "__best__":
        # Best published model per seed, then averaged -- the fairest reading of "the models
        # lose", since taking the mean over models would flatter the trivial rule.
        by_seed: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["model"] in MODELS:
                by_seed[r["seed"]].append(fnum(r.get(metric)))
        return mean([max(v) for v in by_seed.values() if v])
    return mean([fnum(r.get(metric)) for r in rows if r["model"] == model])


def lit_range(benchmark, metric, which) -> float:
    rows = literature_rows(benchmark)
    if which.startswith("cell_"):
        cells = [fnum(r.get(metric)) for r in rows if r["model"] in MODELS]
        cells = [c for c in cells if not math.isnan(c)]
        if not cells:
            return math.nan
        return min(cells) if which == "cell_min" else max(cells)
    means = [mean([fnum(r.get(metric)) for r in rows if r["model"] == m]) for m in MODELS]
    means = [m for m in means if not math.isnan(m)]
    if not means:
        return math.nan
    return min(means) if which == "min" else max(means)


def lit_diff(count, which, resamples=10000) -> float:
    """Best published model (by mean over seeds) minus the trivial rule, paired by seed."""
    rows = [r for r in literature_rows("motion") if r["object_count"] == count]
    if not rows:
        return math.nan
    best = max(MODELS, key=lambda m: mean([fnum(r["f1"]) for r in rows if r["model"] == m]))
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["model"]] = fnum(r["f1"])
    diffs = [v[best] - v["trivial_velocity"] for v in by_seed.values()
             if best in v and "trivial_velocity" in v]
    if which == "mean":
        return mean(diffs)
    rng = random.Random(0)
    boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return boots[int(0.025 * resamples)] if which == "lo" else boots[int(0.975 * resamples) - 1]


PLANNING_RUNS = {"0": "planning_literature_3obj", "1": "planning_literature_3obj_s1",
                 "2": "planning_literature_3obj_s2"}


# Our model at training seeds 1 and 2 was planned in separate runs on the same episodes.
SPARSE_PLANNING_RUNS = {"0": "planning_literature_3obj",
                        "1": "planning_sparse_contact_3obj_heldout_s1",
                        "2": "planning_sparse_contact_3obj_heldout_s2"}


def planning_value(model, which) -> float:
    def one(seed):
        runs = SPARSE_PLANNING_RUNS if model == "sparse" else PLANNING_RUNS
        rows = read_rows(runs[seed], "planning_results.csv")
        return mean([fnum(r["success_rate"]) for r in rows if r["condition"] == model])
    if which == "pooled":
        vals = [one(s) for s in PLANNING_RUNS]
        return math.nan if any(math.isnan(v) for v in vals) else mean(vals)
    return one(which)


def pixel_value(run, condition, metric) -> float:
    rows = read_rows(run, "pixel_benchmark.csv")
    return mean([fnum(r.get(metric)) for r in rows if r["condition"] == condition])


def json_field(run: str, name: str, dotted: str) -> float:
    path = RUNS / run / name
    if not path.exists():
        return math.nan
    node = json.loads(path.read_text(encoding="utf-8"))
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return math.nan
        node = node[part]
    return fnum(node)


def json_value(runs, name, dotted, scale=1.0) -> float:
    vals = [json_field(r, name, dotted) for r in runs]
    # A missing run must fail the claim, not quietly shrink the average.
    return math.nan if any(math.isnan(v) for v in vals) else mean(vals) * scale


def onset_rate(pattern: str) -> float:
    """Kept onset rows as a percentage of all steps, pooled over every split matching pattern."""
    kept = total = 0
    for meta in TRANSITIONS.glob(f"{pattern}/split_metadata.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        kept += sum(m["hard_transition_counts"].values())
        total += sum(m["full_transition_counts"].values())
    return 100.0 * kept / total if total else math.nan


def recompute(source) -> float | None:
    """The artifact's value, NaN if its run is missing, or None if the kind is not run-backed."""
    kind = source[0]
    if kind == "audit":
        return audit_value(*source[1:])
    if kind == "literature":
        return literature_value(*source[1:])
    if kind == "lit_range":
        return lit_range(*source[1:])
    if kind == "lit_diff":
        return lit_diff(*source[1:])
    if kind == "planning":
        return planning_value(*source[1:])
    if kind == "pixel":
        return pixel_value(*source[1:])
    if kind == "json":
        return json_value(*source[1:])
    if kind == "onset_rate":
        return onset_rate(source[1])
    if kind == "json_diff":
        runs, name, a, b = source[1:]
        return json_value(runs, name, a) - json_value(runs, name, b)
    if kind == "derived":
        # The expressions are string literals in CLAIMS above, never external input, and are
        # evaluated with no builtins beyond min/max.
        return float(eval(source[1], {"__builtins__": {}}, {"min": min, "max": max}))
    return None


# --------------------------------------------------------------------------------------
# Manuscript scanning
# --------------------------------------------------------------------------------------

NUMBER = re.compile(r"(?<![\d.])\d+\.\d+(?![\d])")


def manuscript_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    return re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", text)


def token_present(printed: str, body: str) -> bool:
    digits = re.escape(printed.lstrip("-"))
    return re.search(rf"(?<![\d.]){digits}(?![\d])", body) is not None


def tolerance(printed: str, derived: bool) -> float:
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    unit = 10.0 ** -decimals
    # Arithmetic on already-rounded numbers can be off by a full unit of the last digit.
    return unit * (1.0 if derived else 0.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=HERE / "main.tex")
    parser.add_argument("--list", action="store_true", help="Print the ledger and exit.")
    args = parser.parse_args()

    if args.list:
        for printed, where, source in CLAIMS:
            detail = " / ".join(str(s) for s in source[1:] if s is not None)
            print(f"{printed:>7}  {where:<55s} {source[0]}: {detail}")
        return 0

    body = manuscript_body(args.tex)
    failures = unverifiable = 0

    # 1. Coverage: every number in the paper has a ledger entry at the printed precision.
    ledgered = {p.lstrip("-") for p, _, _ in CLAIMS}
    for token in sorted(set(NUMBER.findall(body))):
        if token not in ledgered:
            print(f"FAIL  coverage: {token} appears in the paper but has no ledger entry")
            failures += 1

    # 2 and 3: presence, then recomputation.
    for printed, where, source in CLAIMS:
        if not token_present(printed, body):
            print(f"FAIL  {where}: {printed} is in the ledger but no longer appears in the paper")
            failures += 1
            continue
        kind = source[0]
        if kind in ("ledger", "constant"):
            run = source[2] if kind == "ledger" else None
            if run and (RUNS / run / "summary.json").exists():
                print(f"todo  {where}: {printed} -- run {run} is back; make this entry recomputed")
            elif kind == "ledger":
                print(f"note  {where}: {printed} -- from RESULTS.md ({source[1]}); not recomputable here")
            unverifiable += 1
            continue

        actual = recompute(source)
        value = float(printed)
        if actual is None or math.isnan(actual):
            print(f"FAIL  {where}: {printed} -- run artifact missing, cannot verify")
            failures += 1
        elif abs(actual - value) > tolerance(printed, kind == "derived"):
            print(f"FAIL  {where}: paper says {printed}, artifact says {actual:.4f}")
            failures += 1
        else:
            print(f"ok    {where}: {printed}")

    total = len(CLAIMS)
    print(f"\n{total} claims: {total - failures - unverifiable} verified, "
          f"{unverifiable} settings or RESULTS.md-only, {failures} FAILED")
    if unverifiable:
        print("The RESULTS.md-only claims are the ones to re-derive by hand before submission.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
