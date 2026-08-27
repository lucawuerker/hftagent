#!/bin/bash
# Autonomous post-analysis for the multi-seed replication runs (2026-08-21/22).
# Reads the shared job queue on lagias (data/rerun_queue/<job>/{lane,COMPLETE,rc},
# written by scripts/rerun_queue_lane.sh on both boxes) and, as each job
# completes, runs the standard ladder chain on the M2:
#   1. archive fids (curated book = union of the group Pareto archives)
#   2. wf_arm_factor_analysis.py  (per-factor block ICs, diversity, prequential)
#   3. PIT races — seeding-only arms (LDU8/L1H, children 0): curated book via
#      --keep-fids + availability full (<ARM>CUR) and the kept pool (full);
#      evolved arm (L4WF): book via lineage-replay snapshots (label = arm) and
#      the kept pool via pool_pit (<ARM>POOL)
#   4. scripts/rerun_seeds_table.py — one row per run + per-arm mean/sd
# Server-run arms are rsynced (no transcript). L4WF analyses wait until no
# L4WF evolution run is active on the M2 (8 GB box).
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
LOG=data/rerun_seeds_postanalysis.log
OUT=data/comparisons/wf_arm_analysis_local
PRERUNS=data/workspaces/fmp_archive_equity_nasdaq100pit/preruns
QDIR=/root/QuantFundAgent/data/rerun_queue
JOBS="L4WF_terra_s1 L4WF_terra_s2 LDU8_terra_s3 L1H_terra_s2 L4WF_terra_s3 LDU8_terra_s4 L1H_terra_s3 L4WF_terra_s4"
# runs finished before the queue existed (lane known, already COMPLETE)
EXTRA="LDU8_terra_s2:remote"
PY=./venv/bin/python
echo "$(date -Is) post-analysis watcher start (pid $$)" >> "$LOG"

qstate () {  # job -> "lane rc" (rc empty while running) or "" if unclaimed
  ssh -n -o ConnectTimeout=20 lagias "cd $QDIR/$1 2>/dev/null && echo \$(cat lane) \$(cat rc 2>/dev/null)" 2>/dev/null
}

analyse () {  # arm kind
  local arm="$1" kind="$2"
  echo "$(date -Is) === $arm ($kind): analysis start ===" >> "$LOG"
  $PY - "$arm" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
arm = sys.argv[1]
s = json.loads(Path(f"data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/{arm}/evolution/state.json").read_text())
fids = sorted({p["factor_id"] for grp in s.get("group_archives", []) for entry in grp
               for p in entry["genome"]["programs"]})
Path(f"data/comparisons/{arm}_archive_fids.json").write_text(json.dumps(fids))
print(f"{arm}: {len(fids)} archive factor ids")
PY
  nice -n 5 $PY scripts/wf_arm_factor_analysis.py --arm "$arm" --out-root "$OUT" >> "$LOG" 2>&1
  echo "$(date -Is) $arm factor analysis rc=$?" >> "$LOG"
  if [ "$kind" = seed ]; then
    nice -n 5 $PY scripts/wf_pit_combiner_study.py --out-root "$OUT" \
      --methods equal,ic,lasso,ridge,lightgbm --availability full \
      --arm "$arm" --label "${arm}CUR" \
      --keep-fids "data/comparisons/${arm}_archive_fids.json" >> "$LOG" 2>&1
    echo "$(date -Is) $arm book race rc=$?" >> "$LOG"
    nice -n 5 $PY scripts/wf_pit_combiner_study.py --out-root "$OUT" \
      --methods equal,ic,lasso,ridge,lightgbm --availability full \
      --arm "$arm" --label "$arm" >> "$LOG" 2>&1
    echo "$(date -Is) $arm pool race rc=$?" >> "$LOG"
  else
    nice -n 5 $PY scripts/wf_pit_combiner_study.py --out-root "$OUT" \
      --methods equal,ic,lasso,ridge,lightgbm --availability snapshots \
      --arm "$arm" --label "$arm" >> "$LOG" 2>&1
    echo "$(date -Is) $arm book race (snapshots) rc=$?" >> "$LOG"
    nice -n 5 $PY scripts/wf_pit_combiner_study.py --out-root "$OUT" \
      --methods equal,ic,lasso,ridge --availability pool_pit \
      --arm "$arm" --label "${arm}POOL" >> "$LOG" 2>&1
    echo "$(date -Is) $arm pool race (pool_pit) rc=$?" >> "$LOG"
  fi
  $PY scripts/rerun_seeds_table.py >> "$LOG" 2>&1
  touch "$OUT/$arm/.postanalysis_done"
  echo "$(date -Is) === $arm: analysis DONE ===" >> "$LOG"
}

while true; do
  todo=0
  for spec in $EXTRA $(for j in $JOBS; do echo "$j:queue"; done); do
    arm=${spec%%:*}; src=${spec#*:}
    [ -f "$OUT/$arm/.postanalysis_done" ] && continue
    if [ "$src" = queue ]; then
      st=$(qstate "$arm"); lane=${st%% *}; rc=${st#* }
      [ -z "$st" ] && { todo=1; continue; }                 # unclaimed yet
      [ "$rc" = "$st" ] && { todo=1; continue; }             # running (no rc)
      if [ "$rc" != 0 ]; then
        grep -q "$arm rc=$rc skipped" "$LOG" || echo "$(date -Is) $arm rc=$rc skipped (budget/zero-cand)" >> "$LOG"
        continue
      fi
      [ "$lane" = lagias ] && lane=remote
    else lane=$src; fi
    kind=seed; case "$arm" in L4WF*) kind=evo ;; esac
    if [ "$lane" = remote ]; then
      mkdir -p "$PRERUNS/$arm"
      rsync -az --exclude 'llm_transcript.jsonl' \
        "lagias:/root/QuantFundAgent/$PRERUNS/$arm/" "$PRERUNS/$arm/" >> "$LOG" 2>&1 \
        || { echo "$(date -Is) $arm rsync failed, retry later" >> "$LOG"; todo=1; continue; }
      rsync -az --ignore-existing \
        "lagias:/root/QuantFundAgent/quant_fund_agent/factors/researcher/" \
        quant_fund_agent/factors/researcher/ >> "$LOG" 2>&1
    fi
    if [ "$kind" = evo ] && pgrep -f "run_factor_evolution.py --name L4WF" > /dev/null; then
      todo=1; continue
    fi
    analyse "$arm" "$kind"
  done
  [ "$todo" = 0 ] && break
  sleep 600
done
echo "$(date -Is) post-analysis watcher DONE" >> "$LOG"
