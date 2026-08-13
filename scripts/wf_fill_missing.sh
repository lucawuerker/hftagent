#!/bin/bash
# One-shot completeness repair: once the queue is idle, find labels whose
# jsonl is missing (method, block) pairs — failed fits are never recorded, so
# resume alone won't retry them once the .done marker exists. Clearing .done
# (jsonl KEPT) makes the driver re-run only the missing pairs.
set -u
OUT=/root/QuantFundAgent/data/comparisons/wf_arm_analysis
MARK=$OUT/pit_combiners/fill_missing.done
LOG=$OUT/driver.log
[ -f "$MARK" ] && exit 0
while pgrep -f "wf_pit_combiner_stud[y]" >/dev/null \
   || pgrep -f "wf_arm_factor_analysi[s]" >/dev/null; do
  sleep 300
done
python3 - "$OUT" >> "$LOG" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]) / "pit_combiners"
METHODS = {"equal", "ic", "ridge", "lasso", "rf",
           "kakushadze", "kaku_reg", "autoalpha"}
BLOCKS = set(range(11, 21))
for j in sorted(out.glob("*.jsonl")):
    label = j.stem
    have = set()
    for line in j.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ic") is not None:
            have.add((r["method"], r["block_gen"]))
    missing = {(m, b) for m in METHODS for b in BLOCKS} - have
    if missing:
        print(f"fill_missing: {label}: {len(missing)} pairs missing "
              f"({sorted(missing)[:6]}...) -> clearing .done", flush=True)
        (out / f"{label}.done").unlink(missing_ok=True)
EOF
echo "$(date -Is) fill_missing scan complete (driver re-runs cleared labels)" >> "$LOG"
touch "$MARK"
