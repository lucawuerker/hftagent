#!/bin/bash
# L1H_4omini_s0b — the missing corner of the {model} x {papers} 2x2:
#   Terra + papers    = L1H_terra_s0b        Terra  - papers = LDG_terra_s0b
#   4o-mini + papers  = THIS ARM             4o-mini- papers = LDG_4omini_s0b
# Same L1H spec, same frozen ladder graph snapshot, seed 0. Then the standard
# post-analysis + a level-rho profile so it can be compared level-clean.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/l1h_4omini_arm.log
NAME=L1H_4omini_s0b
OUT=data/comparisons/wf_arm_analysis_local

for i in $(seq 1 40); do
  nice -n 5 ./venv/bin/python run_factor_evolution.py \
    --name "$NAME" --config quant.config.nasdaq100_2010_wf.yaml \
    --seed 0 --model gpt-4o-mini --llm-provider openai --generations 20 \
    --children-per-deme 0 --population 16 \
    --progressive-reveal --reveal-every 1 --test-frac 0.0 \
    --wf-blocks 10 --wf-block-bars 126 --graph-readonly \
    --curation archive --selection-deflation on \
    --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
    --fixed-book data/prebooks/formulaic_101.json \
    --reference-book data/prebooks/formulaic_101.json \
    --n-tickers 0 --horizon 6 --llm-workers 8 --max-cost-usd 15 \
    --retrieval graphrag --mechanism-groups 8 --mechanism-groups-mode max \
    --demes-per-group 3 --seed-ideas-per-group 12 >> "$LOG" 2>&1
  rc=$?
  echo "$(date +%FT%T) $NAME exited rc=$rc (pass $i)" >> "$LOG"
  case $rc in 0|3|4) break ;; esac
  sleep 20
done

./venv/bin/python - "$NAME" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
arm = sys.argv[1]
s = json.loads(Path(f"data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/{arm}/evolution/state.json").read_text())
fids = sorted({p["factor_id"] for grp in s.get("group_archives", [])
               for e in grp for p in e["genome"]["programs"]})
Path(f"data/comparisons/{arm}_archive_fids.json").write_text(json.dumps(fids))
print(f"{arm}: {len(fids)} archive factor ids")
PY

nice -n 5 ./venv/bin/python scripts/wf_arm_factor_analysis.py --arm "$NAME" --out-root "$OUT" >> "$LOG" 2>&1
nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py --out-root "$OUT" \
  --methods equal,ic,lasso,ridge --availability full --arm "$NAME" --label "${NAME}CUR" \
  --keep-fids "data/comparisons/${NAME}_archive_fids.json" >> "$LOG" 2>&1
nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py --out-root "$OUT" \
  --methods equal,ic,lasso,ridge --availability full --arm "$NAME" --label "$NAME" >> "$LOG" 2>&1
./venv/bin/python scripts/pool_level_profile.py --arm "$NAME" >> "$LOG" 2>&1
./venv/bin/python scripts/build_clean_pool_prerun.py --arm "$NAME" --suffix CLNC --max-rho 0.7 >> "$LOG" 2>&1
nice -n 5 ./venv/bin/python scripts/wf_pit_combiner_study.py --out-root "$OUT" \
  --methods equal,ic,lasso,ridge --availability full \
  --arm L1HCLNC_4omini_s0 --label L1HCLNC_4omini_s0 >> "$LOG" 2>&1
echo "$(date +%FT%T) $NAME analysis DONE" >> "$LOG"
