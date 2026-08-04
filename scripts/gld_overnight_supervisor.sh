#!/bin/bash
# Overnight L4WF walk-forward run on GLD 10s bars (launched 2026-08-03).
#
# Mirrors the server ladder's L4WF arm (matrix/terra_wf_ladder.yaml defaults)
# on quant.config.gld_hf.yaml with the HF deltas decided for this run:
#   horizon 60 bars (=10 min), wf-block-bars 42000 (10 blocks ~= last 8.6
#   months prequentially walked forward), QF_EXECUTION_LAG_BARS=3 (signal
#   acted on 30 s after observation), no fixed/reference book (the daily
#   equity formulaic zoo does not apply to a single-ticker HF panel),
#   QF_USE_MCP=0 (single process — the 8 GB host cannot afford a second
#   panel copy in an MCP subprocess), QF_SIGNAL_CACHE_MAX=48.
#
# Behaviour: relaunches the (checkpoint-resuming) run after any crash/kill,
# watchdogs the python RSS, stops on rc=0 (done) / rc=3 (zero candidates) /
# rc=4 (LLM budget exhausted; archive persisted), then chains the post-run
# IC + per-underlying strategy comparison.
cd "$(dirname "$0")/.." || exit 1

SLOG=data/gld_l4wf_supervisor.log
RLOG=data/gld_l4wf_run.log
CLOG=data/gld_l4wf_comparison.log
MAX_RSS_MB=2600
MAX_RELAUNCH=60

export QF_EXECUTION_LAG_BARS=3
export QF_SIGNAL_CACHE_MAX=8
export QF_FIT_CACHE_MAX=16
export QF_USE_MCP=0
# Bound the fit libraries' parallel workspaces: 8-thread lightgbm/BLAS peaks
# were what tipped the 8 GB host into jetsam SIGKILLs mid-generation.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
# Planned process recycle every N completed generations: restarting at the
# checkpoint boundary is loss-free (per-generation checkpoints) and keeps the
# long-lived process's footprint below the jetsam kill threshold — an
# unplanned mid-generation SIGKILL re-spends that generation's LLM calls.
RECYCLE_GENS=1
export QF_LLM_TRANSCRIPT_PATH="data/workspaces/lobster_equity_gld_hf/preruns/L4WF_gld_s0/evolution/llm_transcript.jsonl"

note() { echo "$(date '+%F %T') $*" >> "$SLOG"; }

run_evo() {
  ./venv/bin/python run_factor_evolution.py \
    --config quant.config.gld_hf.yaml --config-name lobster_equity_gld_hf \
    --name L4WF_gld_s0 \
    --model gpt-5.6-terra --llm-provider openai \
    --generations 20 --population 16 \
    --mechanism-groups 8 --mechanism-groups-mode max \
    --demes-per-group 3 --children-per-deme 2 \
    --seed-ideas-per-group 12 --llm-workers 8 \
    --retrieval graphrag --graph-readonly \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 42000 \
    --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 \
    --marginal-model lightgbm --horizon 60 --seed 0 \
    --max-cost-usd 150 >> "$RLOG" 2>&1 &
  EVOPID=$!
  note "evolution pid $EVOPID"
  gens_at_start=$(grep -c "children admitted" "$RLOG" 2>/dev/null || echo 0)
  while kill -0 "$EVOPID" 2>/dev/null; do
    RSS=$(ps -o rss= -p "$EVOPID" 2>/dev/null | awk '{print int($1/1024)}')
    if [ -n "$RSS" ] && [ "$RSS" -gt "$MAX_RSS_MB" ]; then
      note "watchdog: RSS ${RSS}MB > ${MAX_RSS_MB}MB — restarting (checkpoint resume)"
      kill -TERM "$EVOPID" 2>/dev/null
      sleep 20
      kill -KILL "$EVOPID" 2>/dev/null
      break
    fi
    gens_now=$(grep -c "children admitted" "$RLOG" 2>/dev/null || echo 0)
    if [ $((gens_now - gens_at_start)) -ge "$RECYCLE_GENS" ]; then
      note "planned recycle: $RECYCLE_GENS generation(s) completed this launch (RSS ${RSS:-?}MB) — restarting at checkpoint boundary"
      kill -TERM "$EVOPID" 2>/dev/null
      sleep 20
      kill -KILL "$EVOPID" 2>/dev/null
      break
    fi
    sleep 30
  done
  wait "$EVOPID"
  return $?
}

note "supervisor started (pid $$)"
n=0
final_rc=""
while true; do
  n=$((n+1))
  note "launch #$n"
  run_evo
  rc=$?
  note "evolution exited rc=$rc"
  case $rc in
    0) final_rc=0; note "run COMPLETE"; break ;;
    3) note "zero candidates scored — aborting supervision"; exit 3 ;;
    4) final_rc=4; note "LLM budget exhausted — archive persisted, proceeding"; break ;;
  esac
  if [ "$n" -ge "$MAX_RELAUNCH" ]; then
    note "relaunch cap $MAX_RELAUNCH reached — giving up"
    exit 1
  fi
  sleep 20
done

note "starting post-run comparison (IC forwards + per-underlying strategy backtest)"
QF_CONFIG_FILE=quant.config.gld_hf.yaml ./venv/bin/python run_model_comparison.py \
  --preruns L4WF_gld_s0 \
  --out-dir data/comparisons/gld_hf_l4wf \
  --tickers GLD --horizon 60 --holding-period 60 \
  --no-downstream >> "$CLOG" 2>&1
note "comparison rc=$?"
note "supervisor DONE (evolution rc=$final_rc)"
