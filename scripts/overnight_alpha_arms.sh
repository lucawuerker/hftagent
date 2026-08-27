#!/bin/bash
# Overnight local arms (2026-08-19), sequential on the 8 GB M2:
#
#   LDG_4omini_s0b  — the LDG spec (graph briefs, NO papers) on gpt-4o-mini
#                     instead of Terra; SAME frozen ladder graph snapshot as
#                     LDG_terra_s0b, so model quality is the only variable.
#   L1HA_terra_s0b  — the L1H spec (graph + papers) on Terra, but the 8
#                     mechanism groups are the mechanisms the 101 formulaic
#                     alphas occupy (top 8 by the highest absolute fit-window
#                     IC among the alphas occupying them) instead of the
#                     graph's under-covered communities.  Retrieval still reads
#                     the frozen snapshot — only the group briefs change.
#
# Same watchdog/relaunch discipline as scripts/rerun_l1h_ldg_overnight.sh.
# Exit codes: 0 done, 3 zero-candidates, 4 budget stop.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0
export QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/overnight_alpha_arms.log
MONLOG=data/overnight_alpha_arms_monitor.log
RSS_LIMIT_KB=5400000   # 5.4 GB

echo "$(date -Is) overnight alpha-arms supervisor start (pid $$)" >> "$LOG"

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
  local name="$1"; local model="$2"; local cap="$3"; shift 3
  echo "$(date -Is) === arm $name ($model, cap \$$cap) start ===" >> "$LOG"
  for i in $(seq 1 60); do
    nice -n 5 ./venv/bin/python run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --model "$model" --llm-provider openai --generations 20 \
      --children-per-deme 0 --population 16 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
      --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --max-cost-usd "$cap" \
      --retrieval graphrag --mechanism-groups 8 --mechanism-groups-mode max \
      --demes-per-group 3 --seed-ideas-per-group 12 "$@" >> "$LOG" 2>&1 &
    RUN_PID=$!
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

# Arm A: LDG on gpt-4o-mini (graph briefs, no papers, frozen snapshot groups)
run_arm LDG_4omini_s0b gpt-4o-mini 15 --seed-paperless
rc1=$?

# Arm B: L1H on Terra, groups = the mechanisms the 101 formulaic alphas occupy
run_arm L1HA_terra_s0b gpt-5.6-terra 40 \
  --mechanism-groups-file data/knowledge/alpha_mechanism_groups.json
rc2=$?

echo "$(date -Is) overnight alpha arms done: LDG_4omini_s0b rc=$rc1, L1HA_terra_s0b rc=$rc2" >> "$LOG"
