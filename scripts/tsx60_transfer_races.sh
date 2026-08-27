#!/bin/bash
# Diagnostic TSX 60 transfer races: Lasso record of the published books.
# Panel: quant.config.tsx60.yaml (current S&P/TSX 60, survivorship-biased).
# Isolated signal store — the cache is keyed by (factor_id, code hash) only.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent"
export QF_CONFIG_FILE=quant.config.tsx60.yaml
export QF_USE_MCP=0
export QF_SIGNAL_STORE_DIR="$PWD/data/comparisons/tsx60_transfer/signal_store"
OUT=data/comparisons/tsx60_transfer
LOG=$OUT/races.log
mkdir -p "$OUT"
echo "$(date -Is) TSX60 races start" > "$LOG"

race () {
  echo "$(date -Is) === $* ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py \
    --out-root "$OUT" --methods lasso "$@" >> "$LOG" 2>&1
  echo "$(date -Is) rc=$?" >> "$LOG"
}

race --arm LDU8_terra_s0  --label LDU8CUR_tsx   --availability full \
     --keep-fids data/comparisons/LDU8_terra_s0_archive_fids.json
race --arm L1H_terra_s0b  --label L1HCUR_tsx    --availability full \
     --keep-fids data/comparisons/L1H_terra_s0b_archive_fids.json
race --arm L1H_terra_s1   --label L1HCUR_s1_tsx --availability full \
     --keep-fids data/comparisons/L1H_terra_s1_archive_fids.json
race --arm L4WF_terra_s0  --label L4WF_tsx      --availability snapshots
race --arm zoo            --label zoo_tsx       --availability full
echo "$(date -Is) TSX60 races DONE" >> "$LOG"
