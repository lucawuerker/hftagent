#!/bin/bash
# Waits for scripts/overnight_alpha_arms.sh to finish, then runs the standard
# ladder post-analysis for the two local arms and prints the comparison-table
# rows.  Per arm:
#   1. archive fids  — the curated book = the union of the group Pareto archives
#   2. wf_arm_factor_analysis  — per-factor block ICs, diversity, prequential
#   3. PIT combiner races      — curated book (<ARM>CUR, --keep-fids) + kept pool
#   4. scripts/alpha_arms_table.py — the comparison-table columns
# Both arms are children-per-deme 0, so every member is a generation-0 seed and
# `--availability full` is point-in-time honest (see thesis_ablation README).
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/alpha_arms_analysis.log
OUT=data/comparisons/wf_arm_analysis_local
PRERUNS=data/workspaces/fmp_archive_equity_nasdaq100pit/preruns
ARMS="LDG_4omini_s0b L1HA_terra_s0b"

while pgrep -f "overnight_alpha_arms.sh" > /dev/null; do sleep 120; done
echo "$(date -Is) runs finished — analysis start" >> "$LOG"

for arm in $ARMS; do
  if [ ! -f "$PRERUNS/$arm/evolution/state.json" ]; then
    echo "$(date -Is) $arm: no state.json — skipped" >> "$LOG"; continue
  fi
  echo "$(date -Is) === $arm: archive fids ===" >> "$LOG"
  ./venv/bin/python - "$arm" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
arm = sys.argv[1]
s = json.loads(Path(f"data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/{arm}/evolution/state.json").read_text())
fids = sorted({p["factor_id"]
               for grp in s.get("group_archives", [])
               for entry in grp
               for p in entry["genome"]["programs"]})
Path(f"data/comparisons/{arm}_archive_fids.json").write_text(json.dumps(fids))
print(f"{arm}: {len(fids)} archive factor ids")
PY

  echo "$(date -Is) === $arm: factor analysis ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_arm_factor_analysis.py \
    --arm "$arm" --out-root "$OUT" >> "$LOG" 2>&1
  echo "$(date -Is) $arm factor analysis rc=$?" >> "$LOG"

  echo "$(date -Is) === $arm: PIT race (curated book) ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py \
    --out-root "$OUT" --methods equal,ic,lasso,ridge --availability full \
    --arm "$arm" --label "${arm}CUR" \
    --keep-fids "data/comparisons/${arm}_archive_fids.json" >> "$LOG" 2>&1
  echo "$(date -Is) $arm book race rc=$?" >> "$LOG"

  echo "$(date -Is) === $arm: PIT race (kept pool) ===" >> "$LOG"
  nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py \
    --out-root "$OUT" --methods equal,ic,lasso,ridge --availability full \
    --arm "$arm" --label "$arm" >> "$LOG" 2>&1
  echo "$(date -Is) $arm pool race rc=$?" >> "$LOG"
done

echo "$(date -Is) === comparison table ===" >> "$LOG"
./venv/bin/python scripts/alpha_arms_table.py --arms \
  LDG_terra_s0b:LDGCUR_terra_s0b:"LDG-Terra (Baseline)" \
  LDG_4omini_s0b:LDG_4omini_s0bCUR:"LDG-4o-mini" \
  L1H_terra_s0b:L1HCUR_terra_s0b:"L1H (Baseline)" \
  L1HA_terra_s0b:L1HA_terra_s0bCUR:"L1H-Alpha-Mechanismen" \
  >> "$LOG" 2>&1
echo "$(date -Is) analysis DONE" >> "$LOG"
