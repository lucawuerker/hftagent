#!/bin/bash
# Level-clean (rho_med < 0.9) pool PIT races for the two LDG arms, so the
# gpt-4o-mini vs Terra comparison is not driven by the persistent-level factor
# class.  Same protocol as the 2026-08-16 clean races (availability full is
# PIT-honest: both arms are children-per-deme 0, every member a gen-0 seed).
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/clean_ldg_races.log
for arm in LDGCLNB_4omini_s0 LDGCLNB_terra_s0; do
  echo "$(date +%FT%T) === $arm ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py \
    --out-root data/comparisons/wf_arm_analysis_local \
    --methods equal,ic,lasso,ridge --availability full \
    --arm "$arm" --label "$arm" >> "$LOG" 2>&1
  echo "$(date +%FT%T) $arm rc=$?" >> "$LOG"
done
echo "$(date +%FT%T) clean LDG races DONE" >> "$LOG"
