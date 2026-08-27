#!/bin/bash
# Ideation-decomposition lane A on lagias: E2 LDP (random papers, no graph)
# then E1 LDG (graph briefs, no papers). Seeding-only (children 0), WF
# schedule, resume-safe. Run inside lagias-research.slice.
set -u
cd /root/QuantFundAgent || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=data/decomp_laneA.log
run_arm () {
  local name="$1"; shift
  for i in $(seq 1 60); do
    ./venv/bin/python run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --model gpt-5.6-terra --generations 20 \
      --children-per-deme 0 --population 16 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
      --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --max-cost-usd 40 --llm-provider openai "$@" >> "$LOG" 2>&1
    rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; return 0 ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc" >> "$LOG"; return $rc ;;
    esac
    sleep 20
  done
}
echo "$(date -Is) lane A start" >> "$LOG"
run_arm LDP_terra_s0 --retrieval none --mechanism-groups 1 \
  --demes-per-group 4 --seed-ideas-per-group 96 --seed-papers 24
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
run_arm LDG_terra_s0 --retrieval graphrag --seed-paperless \
  --mechanism-groups 8 --mechanism-groups-mode max --demes-per-group 3 \
  --seed-ideas-per-group 12
echo "$(date -Is) lane A done" >> "$LOG"
