#!/bin/bash
# Supervisor lane for the LLM-contribution ablation (matrix/ablation_qa.yaml):
# L1H (selection-only), L4D (deterministic evolution), L0WF (GP walk-forward).
# Same mechanics as ladder_lane.sh — per-arm lock lets N lanes self-partition;
# arms auto-resume from checkpoints on relaunch.  Runs inside the research
# slice so a restored CPUQuota re-binds it automatically.
LANE="${1:?lane name}"; DELAY="${2:-0}"
cd /root/QuantFundAgent || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="data/ablation_qa_${LANE}.log"
sleep "$DELAY"
while true; do
  systemd-run --quiet --collect --scope --slice=lagias-research.slice \
    nice -n 15 \
    ./venv/bin/python -u run_ablation_matrix.py --plan matrix/ablation_qa.yaml --no-probes >> "$LOG" 2>&1
  echo "[supervisor] ${LANE} orchestrator exited rc=$? at $(date -Is) — next pass in 300s" >> "$LOG"
  sleep 300
done
