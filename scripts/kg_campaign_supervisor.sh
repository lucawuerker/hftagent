#!/bin/bash
# Outer supervisor for the KG campaign chain: the environment can kill
# long-running background trees; the chain skips completed runs, so a
# relaunch resumes at the interrupted run (re-seeding it from scratch —
# per-run LLM cap bounds the loss). Max 60 passes.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
LOG=data/kg_campaign/chain.log
for i in $(seq 1 60); do
  [ -f data/kg_campaign/STOP ] && exit 0
  n_done=$(ls data/kg_campaign/run_*_summary.json 2>/dev/null | wc -l | tr -d " ")
  [ "$n_done" -ge 20 ] && { echo "$(date -Is) all 20 runs done — supervisor exit" >> "$LOG"; exit 0; }
  echo "$(date -Is) supervisor pass $i (done: $n_done/20)" >> "$LOG"
  bash scripts/kg_campaign_chain.sh
  sleep 30
done
