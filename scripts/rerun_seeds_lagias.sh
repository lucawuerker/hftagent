#!/bin/bash
# Multi-seed replication lane on lagias (2026-08-21): L4WF_terra_s1 (arm 6,
# seed 1) then LDU8_terra_s2 (arm 1, seed 2).  Specs are byte-identical to the
# originals (matrix/terra_wf_ladder.yaml L4WF_terra_s0; scripts/ld_parity_chain.sh
# LDU8_terra_s0) except --seed.  Graphrag arms pin QF_GRAPH_PATH to the frozen
# ladder snapshot (mandatory for comparability since the 08-13 link-backs).
# The M2 runs the complementary chain (scripts/rerun_seeds_local.sh):
# LDU8_terra_s1 -> L1H_terra_s1 -> L4WF_terra_s2.
#
# RESOURCE DISCIPLINE: inside lagias-research.slice (CPUQuota=200%) via
# systemd-run --scope + nice 15; one arm at a time, OMP=2 = the quota.
set -u
cd /root/QuantFundAgent || exit 1
export PYTHONUNBUFFERED=1
export QF_USE_MCP=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/rerun_seeds_lagias.log

echo "$(date -Is) rerun-seeds lagias supervisor start (pid $$)" >> "$LOG"

run_arm () {
  local name="$1" seed="$2" cap="$3"; shift 3
  echo "$(date -Is) === arm $name (seed $seed) start ===" >> "$LOG"
  for i in $(seq 1 200); do
    systemd-run --quiet --collect --scope --slice=lagias-research.slice \
      nice -n 15 \
      ./venv/bin/python -u run_factor_evolution.py \
        --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
        --seed "$seed" --model gpt-5.6-terra --llm-provider openai \
        --generations 20 --population 16 \
        --mechanism-groups 8 --mechanism-groups-mode max \
        --demes-per-group 3 --seed-ideas-per-group 12 \
        --progressive-reveal --reveal-every 1 --test-frac 0.0 \
        --wf-blocks 10 --wf-block-bars 126 \
        --curation archive --selection-deflation on \
        --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
        --fixed-book data/prebooks/formulaic_101.json \
        --reference-book data/prebooks/formulaic_101.json \
        --n-tickers 0 --horizon 6 --llm-workers 8 \
        --max-cost-usd "$cap" "$@" >> "$LOG" 2>&1
    rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; return 0 ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc" >> "$LOG"; return $rc ;;
    esac
    sleep 60
  done
  echo "$(date -Is) $name gave up after 200 passes" >> "$LOG"; return 1
}

# arm 6: L4WF = full config, graphrag, children-per-deme 2
run_arm L4WF_terra_s1 1 165 --children-per-deme 2 --retrieval graphrag --graph-readonly
rc1=$?
# arm 1: LDU8 = ungrounded (no graph, no papers), neutral 8x3, seeding only
run_arm LDU8_terra_s2 2 20 --children-per-deme 0 --retrieval none --neutral-groups
rc2=$?
echo "$(date -Is) lagias rerun lane done: L4WF_terra_s1 rc=$rc1 LDU8_terra_s2 rc=$rc2" >> "$LOG"
