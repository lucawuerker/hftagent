#!/bin/bash
# Chain (M2, 2026-08-09): (1) PIT combiner race for the finished L1H book
# (availability=full — every factor predates the reveals), then (2) launch
# L1HB_terra_s0: the archive-growth experiment — same selection-only setup
# but seed-ideas-per-group 24 (=192 ideas), targeting a ~40-50 factor final
# archive to answer the cost question empirically.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=32
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
rm -f data/comparisons/wf_arm_analysis_local/pit_combiners/L1H_terra_s0.jsonl
./venv/bin/python scripts/wf_pit_combiner_study.py --arm L1H_terra_s0 \
  --availability full --methods equal,ic,ridge,lasso,rf,autoalpha \
  --out-root data/comparisons/wf_arm_analysis_local >> data/l1h_pit.log 2>&1
echo "$(date -Is) L1H PIT race done rc=$?" >> data/l1h_pit.log

LOG=data/l1hb_local_run.log
echo "$(date -Is) L1HB local supervisor start" >> "$LOG"
for i in $(seq 1 60); do
  ./venv/bin/python run_factor_evolution.py \
    --name L1HB_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
    --children-per-deme 0 --population 16 --seed-ideas-per-group 24 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
    --retrieval graphrag --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 --max-cost-usd 60 \
    --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) run exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) L1HB COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) terminal rc=$rc — stopping" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
