#!/bin/bash
# Supervisor for the KG nonlinear-combiner study: relaunch on crash
# (the study is resume-safe per (model, block)), stop on clean exit.
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
LOG=data/comparisons/kg_nonlinear_combiners/run.log
mkdir -p data/comparisons/kg_nonlinear_combiners
for attempt in $(seq 1 20); do
  echo "$(date -Is) attempt $attempt" >> "$LOG"
  ./venv/bin/python scripts/kg_nonlinear_combiners.py >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Is) rc=$rc" >> "$LOG"
  [ "$rc" = 0 ] && { echo "$(date -Is) DONE" >> "$LOG"; exit 0; }
  sleep 30
done
echo "$(date -Is) GAVE UP after 20 attempts" >> "$LOG"
exit 1
