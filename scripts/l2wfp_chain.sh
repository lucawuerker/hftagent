#!/bin/bash
# 2026-08-12: (1) book analysis for L2WFB (the failed-parity 10-seed rerun —
# kept as a no-retrieval replicate), then (2) L2WFP_terra_s0 = the REAL
# seeding-parity no-retrieval arm, using the chunked seed-brainstorm fix in
# evolution/loop.py (12-idea calls with retry; the 86-idea single call that
# timed out and left L2WFB with 10/96 seeds can no longer happen).
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
LOG=data/l2wfp_chain.log
PY=./venv/bin/python
echo "$(date -Is) L2WFB book analysis" >> "$LOG"
$PY scripts/wf_arm_factor_analysis.py --arm L2WFB_terra_s0 \
  --out-root data/comparisons/wf_arm_analysis_local >> "$LOG" 2>&1
echo "$(date -Is) L2WFB analysis rc=$?" >> "$LOG"

echo "$(date -Is) L2WFP supervisor start" >> "$LOG"
for i in $(seq 1 80); do
  $PY run_factor_evolution.py \
    --name L2WFP_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --retrieval none --mechanism-groups 1 --demes-per-group 4 \
    --children-per-deme 12 --population 16 --seed-ideas-per-group 96 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 \
    --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 --max-cost-usd 120 \
    --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) run exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) L2WFP COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) terminal rc=$rc — stopping" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
