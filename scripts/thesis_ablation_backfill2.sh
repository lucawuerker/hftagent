#!/bin/zsh
# Wave 2: curated-book PIT races for the ideation arms that only have pool
# races (LDU8, LDP8, LDG, L1HBD, L1HB_4omini).  Builds <arm>CUR pseudo-preruns
# (factor_db only, no evolution dir -> whole published book PIT-available,
# same trick as L1HCUR/L1HBCUR) and races them with the linear methods.
# Waits for wave 1 (thesis_ablation_backfill.sh) to finish first.
set -x
cd "$(dirname "$0")/.."
PY=./venv/bin/python
OUT=data/comparisons/wf_arm_analysis_local
LOG=data/thesis_ablation_backfill.log

while ! grep -q BACKFILL_DONE "$LOG" 2>/dev/null; do sleep 60; done

$PY - <<'EOF'
import json, shutil
from pathlib import Path
WS = Path('data/workspaces/fmp_archive_equity_nasdaq100pit/preruns')
PAIRS = {
    'LDU8_terra_s0': 'LDU8CUR_terra_s0',
    'LDP8_terra_s0': 'LDP8CUR_terra_s0',
    'LDG_terra_s0': 'LDGCUR_terra_s0',
    'L1HBD_terra_s0': 'L1HBDCUR_terra_s0',
    'L1HB_4omini_s0': 'L1HB4OMINICUR_s0',
}
for src, dst in PAIRS.items():
    d = WS / dst / 'factors'
    d.mkdir(parents=True, exist_ok=True)
    db = json.loads((WS / src / 'factors/factor_db.json').read_text())
    # published book is already the curated archive for these arms; keep as-is
    (d / 'factor_db.json').write_text(json.dumps(db, indent=1))
    srccode = WS / src / 'factors/code'
    if srccode.exists():
        shutil.copytree(srccode, d / 'code', dirs_exist_ok=True)
        # remap code paths into the pseudo-prerun so races resolve locally
        for r in db['factors']:
            p = Path(r['code_path'])
            local = d / 'code' / p.name
            if local.exists():
                r['code_path'] = str(local.resolve())
        (d / 'factor_db.json').write_text(json.dumps(db, indent=1))
    print('built', dst, len(db['factors']))
EOF

for arm in LDU8CUR_terra_s0 LDP8CUR_terra_s0 LDGCUR_terra_s0 \
           L1HBDCUR_terra_s0 L1HB4OMINICUR_s0; do
  $PY scripts/wf_pit_combiner_study.py --arm "$arm" \
    --methods equal,ic,ridge,lasso --out-root "$OUT" \
    || echo "PIT_FAILED $arm"
done

echo BACKFILL2_DONE
