#!/bin/bash
# M2 chain: E4 curated-book PIT races (L1HB + L1H, snapshots availability),
# then E3 L1HB_4omini_s0 (grounded seeding-only on gpt-4o-mini), then
# E6 L1HBD_terra_s0 (grounded seeding-only + debate). Resume-safe.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/m2_decomp_chain.log
PY=./venv/bin/python
echo "$(date -Is) E4: curated-book PIT races" >> "$LOG"
$PY scripts/wf_pit_combiner_study.py --arm L1HB_terra_s0 --label L1HB_curated \
  --methods ic,ridge,lasso,lightgbm \
  --out-root data/comparisons/wf_arm_analysis_local >> "$LOG" 2>&1
echo "$(date -Is) L1HB curated race rc=$?" >> "$LOG"
$PY scripts/wf_pit_combiner_study.py --arm L1H_terra_s0 --label L1H_curated \
  --methods ic,ridge,lasso,lightgbm \
  --out-root data/comparisons/wf_arm_analysis_local >> "$LOG" 2>&1
echo "$(date -Is) L1H curated race rc=$?" >> "$LOG"

run_arm () {
  local name="$1"; shift
  for i in $(seq 1 60); do
    $PY run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --generations 20 \
      --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
      --children-per-deme 0 --population 16 --seed-ideas-per-group 24 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
      --retrieval graphrag --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --llm-provider openai "$@" >> "$LOG" 2>&1
    rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; return 0 ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc" >> "$LOG"; return $rc ;;
    esac
    sleep 20
  done
}
echo "$(date -Is) E3: L1HB_4omini_s0 start" >> "$LOG"
run_arm L1HB_4omini_s0 --model gpt-4o-mini --max-cost-usd 15
echo "$(date -Is) E6: L1HBD_terra_s0 start" >> "$LOG"
run_arm L1HBD_terra_s0 --model gpt-5.6-terra --debate on --max-cost-usd 100
echo "$(date -Is) M2 chain done" >> "$LOG"
