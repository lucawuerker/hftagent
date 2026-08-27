#!/bin/bash
# Ideation-decomposition lane B on lagias: E0 LDU (no graph, no papers) —
# the fourth 2x2 corner. Seeding-only, WF schedule, resume-safe.
set -u
cd /root/QuantFundAgent || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=data/decomp_laneB.log
echo "$(date -Is) lane B start" >> "$LOG"
for i in $(seq 1 60); do
  ./venv/bin/python run_factor_evolution.py \
    --name LDU_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --retrieval none --mechanism-groups 1 --demes-per-group 4 \
    --children-per-deme 0 --population 16 --seed-ideas-per-group 96 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 \
    --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 \
    --max-cost-usd 40 --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) LDU exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) LDU COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) LDU terminal rc=$rc" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
