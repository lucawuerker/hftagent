#!/bin/bash
# Multi-seed replication chain on the M2 (2026-08-21), complementary to
# scripts/rerun_seeds_lagias.sh:  LDU8_terra_s1 (arm 1) -> L1H_terra_s1 (arm 4)
# -> L4WF_terra_s2 (arm 6).  Specs byte-identical to the originals except
# --seed (LDU8: scripts/ld_parity_chain.sh; L1H: scripts/rerun_l1h_ldg_overnight.sh;
# L4WF: matrix/terra_wf_ladder.yaml L4WF_terra_s0).  Graphrag arms pin
# QF_GRAPH_PATH to the frozen ladder snapshot.  Sequential, nice'd, RSS
# watchdog at 5.4 GB (kill -> checkpoint resume).  rc 0 done, 3 zero-cand,
# 4 budget stop.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0
export QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/rerun_seeds_local.log
RSS_LIMIT_KB=5400000

echo "$(date -Is) local rerun-seeds supervisor start (pid $$)" >> "$LOG"

run_arm () {
  local name="$1" seed="$2" cap="$3"; shift 3
  echo "$(date -Is) === arm $name (seed $seed) start ===" >> "$LOG"
  for i in $(seq 1 80); do
    nice -n 5 ./venv/bin/python run_factor_evolution.py \
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
      --max-cost-usd "$cap" "$@" >> "$LOG" 2>&1 &
    RUN_PID=$!
    while kill -0 "$RUN_PID" 2>/dev/null; do
      sleep 60
      rss=$(ps -o rss= -p "$RUN_PID" 2>/dev/null | tr -d ' ')
      if [ -n "${rss:-}" ] && [ "$rss" -gt "$RSS_LIMIT_KB" ]; then
        echo "$(date -Is) WATCHDOG: rss ${rss}KB > ${RSS_LIMIT_KB}KB — TERM $RUN_PID" >> "$LOG"
        kill -TERM "$RUN_PID"; sleep 30; kill -KILL "$RUN_PID" 2>/dev/null
      fi
    done
    wait "$RUN_PID"; rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; return 0 ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc — stopping arm" >> "$LOG"; return $rc ;;
    esac
    sleep 20
  done
  echo "$(date -Is) $name gave up after 80 passes" >> "$LOG"; return 1
}

run_arm LDU8_terra_s1 1 20 --children-per-deme 0 --retrieval none --neutral-groups
rc1=$?
run_arm L1H_terra_s1 1 40 --children-per-deme 0 --retrieval graphrag --graph-readonly
rc2=$?
run_arm L4WF_terra_s2 2 165 --children-per-deme 2 --retrieval graphrag --graph-readonly
rc3=$?
echo "$(date -Is) local rerun chain done: LDU8_terra_s1 rc=$rc1 L1H_terra_s1 rc=$rc2 L4WF_terra_s2 rc=$rc3" >> "$LOG"
