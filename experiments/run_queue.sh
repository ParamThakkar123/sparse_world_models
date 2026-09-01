#!/usr/bin/env bash
# Serialise the remaining heavy pipelines.
#
# Running the cross-domain, planar-onset and coverage-gap pipelines concurrently exhausted
# system memory on a 16 GB machine and killed the cross-domain split stage with a bare
# MemoryError -- after its generation stage had already completed, so the failure cost the
# cheap half of the work and left the expensive half intact. This waits on the completion
# markers the other pipelines print, then runs the remaining stages alone.
#
# Every stage is idempotent: generation skips datasets that already exist, so a re-run
# resumes rather than repeats.
set -uo pipefail

wait_for() {
  local marker="$1" logfile="$2" label="$3"
  if [[ ! -f "$logfile" ]]; then
    echo "  no log for ${label}; assuming it is not running"
    return
  fi
  echo "  waiting for ${label} (${marker})"
  until grep -q "$marker" "$logfile" 2>/dev/null; do
    sleep 30
  done
  echo "  ${label} done"
}

echo "############ QUEUE: waiting for in-flight pipelines ############"
wait_for "ONSET_PLANAR_DONE"  experiments/runs/_log_onset_planar12k.log "planar onset 12k"
wait_for "COVERAGE_GAPS_DONE" experiments/runs/_log_coverage_gaps.log   "coverage gaps"

echo "############ QUEUE: cross-domain splits + training ############"
bash experiments/cross_domain_pipeline.sh

echo "############ QUEUE: cross-domain analysis ############"
python -m experiments.cross_domain_analysis \
  --domains tabletop planar billiards clutter \
  --counts 3 5 8 --seeds 0 1 2 3 4 \
  --run-name cross_domain_analysis

echo "############ QUEUE_DONE ############"
