#!/bin/bash
# Local (M2) supervisor for the L1H selection-only ablation arm — moved off
# the throttere-prone server for speed (2026-08-09). Resumes from the pulled
# server checkpoints; relaunch-on-crash; caffeinate keeps the Mac awake while
# plugged in. Exit codes: 0 done, 3 zero-candidates, 4 budget stop.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0
export QF_SIGNAL_CACHE_MAX=64          # 8 GB Mac — bound the signal cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/l1h_local_run.log
echo "$(date -Is) L1H local supervisor start" >> "$LOG"
for i in $(seq 1 60); do
  ./venv/bin/python run_factor_evolution.py \
    --name L1H_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
    --children-per-deme 0 --population 16 --seed-ideas-per-group 12 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
    --retrieval graphrag --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 \
    --max-cost-usd 40 --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) run exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) L1H COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) terminal rc=$rc — stopping" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
echo "$(date -Is) supervisor gave up after 60 passes" >> "$LOG"
