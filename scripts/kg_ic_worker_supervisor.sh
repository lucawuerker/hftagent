#!/usr/bin/env bash
# Supervisor for the KG-campaign IC worker (lagias).
#
# The worker is resume-safe (results.csv is its done-set), so a crash costs at
# most the in-flight book.  It was OOM-killed once (2026-08-17, run 18's ~1.9k
# factor cumulative book); this loop brings it straight back and logs the exit
# code.  Stop with:  touch data/kg_campaign/STOP
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=data/kg_campaign/worker.log
STOP=data/kg_campaign/STOP
# leave headroom for the box's production containers; the ridge solve is BLAS
# bound and does not need all 8 cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

while [ ! -f "$STOP" ]; do
  echo "$(date -Is) [supervisor] launching kg_ic_worker" >>"$LOG"
  ./venv/bin/python scripts/kg_ic_worker.py >>"$LOG" 2>&1
  rc=$?
  echo "$(date -Is) [supervisor] worker exited rc=$rc" >>"$LOG"
  [ -f "$STOP" ] && break
  sleep 30
done
echo "$(date -Is) [supervisor] STOP marker — exiting" >>"$LOG"
