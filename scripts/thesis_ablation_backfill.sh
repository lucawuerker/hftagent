#!/bin/zsh
# Backfill the per-arm analyses + PIT lasso races the thesis ablation chapter
# needs for the arms that never got them locally (LDU8, LDP8, L0WF).
set -x
cd "$(dirname "$0")/.."
PY=./venv/bin/python
OUT=data/comparisons/wf_arm_analysis_local
METHODS=equal,ic,ridge,lasso,lightgbm

for arm in LDU8_terra_s0 LDP8_terra_s0 L0WF_gp_s0; do
  $PY scripts/wf_arm_factor_analysis.py --arm "$arm" --out-root "$OUT" \
    || echo "FACTOR_ANALYSIS_FAILED $arm"
done

$PY scripts/wf_pit_combiner_study.py --arm LDU8_terra_s0 --availability full \
  --methods $METHODS --out-root "$OUT" || echo "PIT_FAILED LDU8"
$PY scripts/wf_pit_combiner_study.py --arm LDP8_terra_s0 --availability full \
  --methods $METHODS --out-root "$OUT" || echo "PIT_FAILED LDP8"
$PY scripts/wf_pit_combiner_study.py --arm L0WF_gp_s0 --availability snapshots \
  --methods $METHODS --out-root "$OUT" || echo "PIT_FAILED L0WF"

echo BACKFILL_DONE
