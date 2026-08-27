#!/bin/bash
# When the local LDU8 arm completes, pull LDP8 home from lagias and continue
# it locally (user 2026-08-15: fastest path to all results). Restores the
# server's protective quota and removes its cron either way.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
LOG=data/ld_parity_chain.log
until grep -q "LDU8_terra_s0 COMPLETE\|LDU8_terra_s0 terminal" "$LOG" 2>/dev/null; do sleep 60; done
if ssh lagias 'grep -q "LDP8 COMPLETE" /root/QuantFundAgent/data/ldp8_server_run.log 2>/dev/null'; then
  echo "$(date -Is) LDP8 already complete on server — pulling results" >> "$LOG"
  rsync -a "lagias:/root/QuantFundAgent/data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/LDP8_terra_s0/" \
    "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/LDP8_terra_s0/"
  echo "$(date -Is) LDP8 results pulled — parity chain done" >> "$LOG"
  exit 0
fi
echo "$(date -Is) migrating LDP8 server->local" >> "$LOG"
ssh lagias 'tmux kill-session -t ldp8 2>/dev/null; pkill -f "ldp8_server_ru[n]"; pkill -f "run_factor_evolution.py --name LDP8_terra_s[0]"; sleep 2; systemctl set-property --runtime lagias-research.slice CPUQuota=200%; rm -f /etc/cron.d/restore-quota-ldp8; rm -f /root/QuantFundAgent/data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/LDP8_terra_s0/orchestrator.lock'
rsync -a "lagias:/root/QuantFundAgent/data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/LDP8_terra_s0/" \
  "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/LDP8_terra_s0/"
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=48 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
for i in $(seq 1 60); do
  ./venv/bin/python run_factor_evolution.py \
    --name LDP8_terra_s0 --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-5.6-terra --generations 20 \
    --mechanism-groups 8 --mechanism-groups-mode max --neutral-groups \
    --demes-per-group 3 --children-per-deme 0 --population 16 \
    --seed-ideas-per-group 12 --seed-papers 24 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 \
    --retrieval none --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 --max-cost-usd 15 \
    --llm-provider openai >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) LDP8(local) exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in
    0) echo "$(date -Is) LDP8_terra_s0 COMPLETE (local) — parity chain done" >> "$LOG"; exit 0 ;;
    3|4) echo "$(date -Is) LDP8 terminal rc=$rc" >> "$LOG"; exit $rc ;;
  esac
  sleep 20
done
