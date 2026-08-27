#!/bin/bash
# wf_arm_factor_analysis for the s0b replication arms (sequential, M2).
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/rerun_factor_analysis_s0b.log
echo "$(date -Is) s0b factor analysis start" >> "$LOG"
for arm in L1H_terra_s0b LDG_terra_s0b; do
  echo "$(date -Is) === $arm ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_arm_factor_analysis.py \
    --arm "$arm" --out-root data/comparisons/wf_arm_analysis_local >> "$LOG" 2>&1
  echo "$(date -Is) $arm rc=$?" >> "$LOG"
done
echo "$(date -Is) s0b factor analysis DONE" >> "$LOG"
