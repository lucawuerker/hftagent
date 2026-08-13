#!/bin/bash
# 2026-08-10: L2 rerun with SEEDING PARITY (user request) — the original
# L2WF_terra_s0 was seeded with only 12 ideas (1 group x default 12/group)
# vs 96 in every other ladder arm, confounding the retrieval ablation.
# L2WFB_terra_s0 = identical L2 configuration but --seed-ideas-per-group 96,
# so ONLY retrieval grounding + the group structure differ from L4WF.
# Waits for the L4IC chain to finish (sequential — one panel in RAM), then
# supervises the run. lagias SSH was down when this was written.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
LOG=data/l2wfb_local_run.log
CHAIN_LOG=data/ablation_analysis_l4ic.log
echo "$(date -Is) waiting for L4IC chain to finish" >> "$LOG"
while true; do
  grep -qE "L4IC COMPLETE|terminal rc=" "$CHAIN_LOG" 2>/dev/null && break
  pgrep -f ablation_analysis_then_l4ic.sh > /dev/null || break
  sleep 300
done
echo "$(date -Is) L2WFB supervisor start" >> "$LOG"
for i in $(seq 1 80); do
  ./venv/bin/python run_factor_evolution.py \
    --name L2WFB_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
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
    0) echo "$(date -Is) L2WFB COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) terminal rc=$rc — stopping" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
