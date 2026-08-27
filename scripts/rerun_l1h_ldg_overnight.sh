#!/bin/bash
# Overnight replication reruns of the thesis-ladder arms 3+4 (2026-08-18):
#   L1H_terra_s0b — graph+papers, seeding-only (same spec as L1H_terra_s0,
#                   scripts/l1h_local_supervisor.sh)
#   LDG_terra_s0b — graph briefs, NO papers (same spec as LDG_terra_s0,
#                   scripts/lagias_lane_decomp_a.sh lane-A second arm)
# Same seed 0, fresh prerun names -> true replication (LLM sampling noise).
# BOTH arms pin QF_GRAPH_PATH to the frozen ladder snapshot: the live
# graph.json got 859 factor link-backs on 2026-08-13 and would resolve
# different mechanism groups than the originals saw.
# Sequential (8 GB M2), nice'd, RSS watchdog at 5.4 GB (kill -> checkpoint
# resume), resource monitor every 5 min. Exit codes: 0 done, 3 zero-cand,
# 4 budget stop.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0
export QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/rerun_l1h_ldg.log
MONLOG=data/rerun_l1h_ldg_monitor.log
RSS_LIMIT_KB=5400000   # 5.4 GB

echo "$(date -Is) overnight rerun supervisor start (pid $$)" >> "$LOG"

# ---- resource monitor: load avg + free/active RAM + run RSS every 5 min ----
monitor () {
  while true; do
    local load mem rss
    load=$(sysctl -n vm.loadavg | tr -d '{}')
    mem=$(vm_stat | awk '/Pages free/{f=$3} /Pages active/{a=$3} END{gsub(/\./,"",f); gsub(/\./,"",a); printf "free=%.1fGB active=%.1fGB", f*16384/1e9, a*16384/1e9}')
    rss=$(pgrep -f "run_factor_evolution.py --name" | head -1 | xargs -I{} ps -o rss= -p {} 2>/dev/null | tr -d ' ')
    echo "$(date -Is) load=${load} ${mem} run_rss=${rss:-na}KB" >> "$MONLOG"
    sleep 300
  done
}
monitor &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

run_arm () {
  local name="$1"; shift
  echo "$(date -Is) === arm $name start ===" >> "$LOG"
  for i in $(seq 1 60); do
    nice -n 5 ./venv/bin/python run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --model gpt-5.6-terra --generations 20 \
      --children-per-deme 0 --population 16 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
      --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --max-cost-usd 40 --llm-provider openai "$@" >> "$LOG" 2>&1 &
    RUN_PID=$!
    # RSS watchdog: TERM the run if it outgrows the 8 GB box; supervisor resumes
    while kill -0 "$RUN_PID" 2>/dev/null; do
      sleep 60
      rss=$(ps -o rss= -p "$RUN_PID" 2>/dev/null | tr -d ' ')
      if [ -n "${rss:-}" ] && [ "$rss" -gt "$RSS_LIMIT_KB" ]; then
        echo "$(date -Is) WATCHDOG: rss ${rss}KB > ${RSS_LIMIT_KB}KB — TERM $RUN_PID" >> "$LOG"
        kill -TERM "$RUN_PID"; sleep 30; kill -KILL "$RUN_PID" 2>/dev/null
      fi
    done
    wait "$RUN_PID"; rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; return 0 ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc — stopping arm" >> "$LOG"; return $rc ;;
    esac
    sleep 20
  done
  echo "$(date -Is) $name gave up after 60 passes" >> "$LOG"
  return 1
}

# Arm 4 rerun: L1H = graphrag retrieval WITH papers, 8 groups max-mode
run_arm L1H_terra_s0b --retrieval graphrag \
  --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
  --seed-ideas-per-group 12
rc1=$?

# Arm 3 rerun: LDG = graph briefs, NO papers (--seed-paperless)
run_arm LDG_terra_s0b --retrieval graphrag --seed-paperless \
  --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
  --seed-ideas-per-group 12
rc2=$?

echo "$(date -Is) overnight rerun done: L1H_terra_s0b rc=$rc1, LDG_terra_s0b rc=$rc2" >> "$LOG"
