#!/usr/bin/env bash
# PIT-honest whole-kept-pool combiner race for L4WF_terra_s0 (lagias).
#
# L4WF actually evolved across the reveals, so its 887-factor kept pool cannot
# be raced with full availability (look-ahead).  --availability pool_pit
# reveals each member one block after the generation that created it
# (block 11 -> 480 factors ... block 20 -> 846), which is the honest analogue
# of the pool races run for the selection-only arms (L1H, LDG, L1HB, ...).
#
# The race needs ~13 GB, the KG-campaign IC worker peaks near 18 GB, and the
# box has 31 GB — so wait for the KG worker to drain its queue (run 20 cum) or
# exit before starting.
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=data/l4wf_pool_race.log
CAMP=data/kg_campaign
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

echo "$(date -Is) [chain] waiting for the KG IC worker to drain" >>"$LOG"
while true; do
  grep -q "^20,cum,ridge," "$CAMP/results.csv" 2>/dev/null && break
  pgrep -f "kg_ic_worker.py" >/dev/null || break
  sleep 120
done
echo "$(date -Is) [chain] KG worker clear — starting L4WF pool race" >>"$LOG"

# resume-safe per (method, block): relaunch on crash until every cell is filled
for attempt in 1 2 3; do
  ./venv/bin/python scripts/wf_pit_combiner_study.py \
      --arm L4WF_terra_s0 \
      --availability pool_pit \
      --label L4WFPOOL_terra_s0 \
      --methods equal,ic,ridge,lasso,lightgbm >>"$LOG" 2>&1
  rc=$?
  echo "$(date -Is) [chain] attempt $attempt exited rc=$rc" >>"$LOG"
  [ $rc -eq 0 ] && break
  sleep 60
done
echo "$(date -Is) [chain] done" >>"$LOG"
