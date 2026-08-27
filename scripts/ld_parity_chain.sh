#!/bin/bash
# E0b/E2b structural-parity reruns (user 2026-08-15): the no-graph 2x2
# corners with --neutral-groups — 8 unlabelled groups x 3 demes, 12 seeds
# each (=96), so archive CAPACITY matches the graph arms while the ideation
# prompt stays untouched. Waits for the running E6 persist to finish first.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
LOG=data/ld_parity_chain.log
PY=./venv/bin/python
while pgrep -f "run_factor_evolution.py --name L1HBD_terra_s[0]" >/dev/null; do sleep 60; done
run_arm () {
  local name="$1"; shift
  for i in $(seq 1 60); do
    $PY run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --model gpt-5.6-terra --generations 20 \
      --mechanism-groups 8 --mechanism-groups-mode max --neutral-groups \
      --demes-per-group 3 --children-per-deme 0 --population 16 \
      --seed-ideas-per-group 12 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 \
      --retrieval none --curation archive --selection-deflation on \
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
echo "$(date -Is) E0b: LDU8_terra_s0 (ungrounded, neutral 8x3)" >> "$LOG"
run_arm LDU8_terra_s0 --max-cost-usd 12
echo "$(date -Is) parity chain done (LDP8 runs on lagias)" >> "$LOG"
