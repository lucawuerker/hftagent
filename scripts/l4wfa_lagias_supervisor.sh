#!/bin/bash
# L4WFA_terra_s0 — the L4WF ladder arm re-run on the FORMULAIC-ALPHA mechanism
# groups (2026-08-19).  Identical to matrix/terra_wf_ladder.yaml's L4WF_terra_s0
# in every respect except the upper population layer: instead of the graph's
# under-covered Louvain communities, the 8 mechanism groups are the mechanisms
# the 101 Kakushadze formulaic alphas occupy, ranked by the highest absolute
# fit-window IC among the alphas occupying them
# (scripts/alpha_mechanism_groups.py -> data/knowledge/alpha_mechanism_groups.json).
# Retrieval grounding still reads the FROZEN ladder graph snapshot, so the only
# changed variable is which mechanisms the demes are steered into.
#
# RESOURCE DISCIPLINE: runs inside lagias-research.slice (CPUQuota=200%) via
# systemd-run --scope + nice 15, exactly like scripts/ladder_lane.sh — this box
# also serves Lagias production and the Hostinger fair-use monitor counts
# absolute CPU.  One arm alone, so OMP=2 matches the 2-core quota (see the
# 2026-08-04 note in ladder_lane.sh: OpenMP spin-thrash inside the quota made a
# LightGBM fit 23s instead of 6s).
set -u
cd /root/QuantFundAgent || exit 1
export PYTHONUNBUFFERED=1
export QF_USE_MCP=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/l4wfa_run.log
NAME=L4WFA_terra_s0

echo "$(date -Is) L4WFA supervisor start (pid $$)" >> "$LOG"
for i in $(seq 1 200); do
  systemd-run --quiet --collect --scope --slice=lagias-research.slice \
    nice -n 15 \
    ./venv/bin/python -u run_factor_evolution.py \
      --name "$NAME" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed 0 --model gpt-5.6-terra --llm-provider openai \
      --generations 20 --population 16 \
      --mechanism-groups 8 --mechanism-groups-mode max \
      --mechanism-groups-file data/knowledge/alpha_mechanism_groups.json \
      --demes-per-group 3 --children-per-deme 2 --seed-ideas-per-group 12 \
      --retrieval graphrag --graph-readonly \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 \
      --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --max-cost-usd 165 >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) $NAME exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) $NAME COMPLETE" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) $NAME terminal rc=$rc" >> "$LOG"; exit $rc ;;
  esac
  sleep 60
done
echo "$(date -Is) $NAME gave up after 200 passes" >> "$LOG"
