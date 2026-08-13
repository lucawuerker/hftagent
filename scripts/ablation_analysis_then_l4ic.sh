#!/bin/bash
# 2026-08-10: sequential local pipeline (lagias SSH down, everything on the M2).
# Phase 1 — book analyses for the new ablation arms (per-factor block ICs,
#           diversity, static combined fits; PIT combiner race for L1HB/L4D).
# Phase 2 — launch the harness-ablation arm L4IC_terra_s0: L4 configuration but
#           --objective ic (standalone |VAL IC| only, no 4-axis Pareto) and NO
#           progressive reveal (classic IS/VAL/TEST split) on the to-2021 panel,
#           so 2021-26 stays an untouched holdout for post-hoc block scoring.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
LOG=data/ablation_analysis_l4ic.log
PY=./venv/bin/python
echo "$(date -Is) phase 1: arm analyses" >> "$LOG"

for ARM in L1H_terra_s0 L1HB_terra_s0 L4D_terra_s0 L0WF_gp_s0; do
  $PY scripts/wf_arm_factor_analysis.py --arm "$ARM" \
    --out-root data/comparisons/wf_arm_analysis_local >> "$LOG" 2>&1
  echo "$(date -Is) factor analysis $ARM rc=$?" >> "$LOG"
done

$PY scripts/wf_pit_combiner_study.py --arm L1HB_terra_s0 --availability full \
  --methods ic,ridge,lasso,lightgbm \
  --out-root data/comparisons/wf_arm_analysis_local/pit_combiners >> "$LOG" 2>&1
echo "$(date -Is) PIT race L1HB rc=$?" >> "$LOG"
$PY scripts/wf_pit_combiner_study.py --arm L4D_terra_s0 \
  --methods ic,ridge,lasso,lightgbm \
  --out-root data/comparisons/wf_arm_analysis_local/pit_combiners >> "$LOG" 2>&1
echo "$(date -Is) PIT race L4D rc=$?" >> "$LOG"

echo "$(date -Is) phase 2: L4IC supervisor start" >> "$LOG"
for i in $(seq 1 80); do
  $PY run_factor_evolution.py \
    --name L4IC_terra_s0 --config quant.config.nasdaq100_2010_to2021.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
    --children-per-deme 2 --population 16 --seed-ideas-per-group 12 \
    --objective ic \
    --graph-readonly --retrieval graphrag --curation archive \
    --selection-deflation on --archive-cap 40 --creative-frac 0.1 \
    --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 --max-cost-usd 180 \
    --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) L4IC run exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) L4IC COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) terminal rc=$rc — stopping" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
