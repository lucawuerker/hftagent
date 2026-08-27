#!/bin/bash
# PIT combiner races for the 2026-08-18 replication reruns (book + pool),
# matching the thesis-ladder conventions:
#   <ARM>CUR_*  = curated book (factor_db==archive, restricted via --keep-fids)
#   <arm>       = kept pool (availability full)
# Linear methods only (equal/ic/lasso/ridge), sequential on the M2.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/rerun_pit_races_s0b.log
OUT=data/comparisons/wf_arm_analysis_local
M=equal,ic,lasso,ridge
echo "$(date -Is) s0b PIT races start" >> "$LOG"

race () {
  echo "$(date -Is) === $* ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py \
    --out-root "$OUT" --methods "$M" --availability full "$@" >> "$LOG" 2>&1
  echo "$(date -Is) rc=$?" >> "$LOG"
}

# --keep-fids takes a PATH to a JSON list, not inline JSON
race --arm L1H_terra_s0b --label L1HCUR_terra_s0b \
  --keep-fids data/comparisons/L1H_terra_s0b_archive_fids.json
race --arm LDG_terra_s0b --label LDGCUR_terra_s0b \
  --keep-fids data/comparisons/LDG_terra_s0b_archive_fids.json
race --arm L1H_terra_s0b --label L1H_terra_s0b
race --arm LDG_terra_s0b --label LDG_terra_s0b
echo "$(date -Is) s0b PIT races DONE" >> "$LOG"
