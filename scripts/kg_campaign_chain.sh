#!/bin/bash
# KG breadth campaign (user 2026-08-16): 20 sequential seeding-only runs
# against the LIVE, run-over-run evolving knowledge graph. Seeding + graph
# link-back run HERE (M2, owns the graph); the IC evaluation trails
# asynchronously on lagias (kg_ic_worker.py) via the queue markers.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0
LOG=data/kg_campaign/chain.log
mkdir -p data/kg_campaign
echo "$(date -Is) campaign chain start" >> "$LOG"
for N in $(seq 1 20); do
  [ -f data/kg_campaign/STOP ] && { echo "$(date -Is) STOP marker" >> "$LOG"; break; }
  if [ -f "data/kg_campaign/run_$(printf %02d $N)_summary.json" ]; then
    # re-ship idempotently: a kill during the ship step must not leave the
    # server with a partial book (run 13 incident, 2026-08-17)
    rsync -a quant_fund_agent/factors/researcher/ \
      lagias:/root/QuantFundAgent/quant_fund_agent/factors/researcher/ >> "$LOG" 2>&1
    rsync -a data/kg_campaign/cumulative_book.json \
      lagias:/root/QuantFundAgent/data/kg_campaign/cumulative_book.json >> "$LOG" 2>&1
    ssh lagias "touch /root/QuantFundAgent/data/kg_campaign/queue/run_$(printf %02d $N).ready" >> "$LOG" 2>&1
    echo "$(date -Is) run $N already done — re-shipped, skip" >> "$LOG"; continue
  fi
  echo "$(date -Is) === run $N seeding ===" >> "$LOG"
  ./venv/bin/python scripts/run_kg_seed_run.py --run-index "$N" --config quant.config.nasdaq100_2010_wf.yaml \
    --seed-ideas-per-group 12 --max-cost-usd 15 >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) run $N rc=$rc" >> "$LOG"
  [ "$rc" = 3 ] && { echo "$(date -Is) zero factors — aborting" >> "$LOG"; exit 3; }
  # ship the new state to the server and queue the IC evaluation
  rsync -a quant_fund_agent/factors/researcher/ \
    lagias:/root/QuantFundAgent/quant_fund_agent/factors/researcher/ >> "$LOG" 2>&1
  rsync -a "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/KG$(printf %02d $N)_terra_s0/" \
    "lagias:/root/QuantFundAgent/data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/KG$(printf %02d $N)_terra_s0/" >> "$LOG" 2>&1
  rsync -a data/kg_campaign/cumulative_book.json \
    lagias:/root/QuantFundAgent/data/kg_campaign/cumulative_book.json >> "$LOG" 2>&1
  ssh lagias "touch /root/QuantFundAgent/data/kg_campaign/queue/run_$(printf %02d $N).ready" >> "$LOG" 2>&1
  echo "$(date -Is) run $N queued for IC eval" >> "$LOG"
done
echo "$(date -Is) campaign chain done (seeding side)" >> "$LOG"
