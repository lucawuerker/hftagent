#!/bin/bash
# One-shot artifact backfill: labels whose PIT run finished BEFORE artifact
# saving was deployed have IC rows but no saved weights/models/predictions.
# Once the primary queue is idle, clear those labels' jsonl+done so the
# existing driver re-runs them with the artifact-saving code (the shared
# signal store makes the re-run much cheaper than the first pass).
# Runs once, then marks itself done.
set -u
OUT=/root/QuantFundAgent/data/comparisons/wf_arm_analysis
MARK=$OUT/pit_combiners/backfill.done
LOG=$OUT/driver.log
[ -f "$MARK" ] && exit 0

# wait until no pit/factor job is running (primary queue drained)
while pgrep -f "wf_pit_combiner_stud[y]" >/dev/null \
   || pgrep -f "wf_arm_factor_analysi[s]" >/dev/null; do
  sleep 300
done

echo "$(date -Is) artifact backfill: queue idle, scanning" >> "$LOG"
n=0
for j in "$OUT"/pit_combiners/*.jsonl; do
  [ -e "$j" ] || continue
  label=$(basename "$j" .jsonl)
  # complete artifacts = final block g20 has a saved prediction
  if ! ls "$OUT/pit_combiners/artifacts/$label/g20/"pred_*.parquet >/dev/null 2>&1; then
    echo "$(date -Is) backfill: re-queueing $label (no artifacts)" >> "$LOG"
    mv "$j" "$j.pre_backfill"   # keep the first pass (incl. lightgbm rows)
    rm -f "$OUT/pit_combiners/$label.done"
    n=$((n+1))
  fi
done
echo "$(date -Is) artifact backfill: $n labels re-queued (driver picks them" \
     "up on its next pass)" >> "$LOG"
touch "$MARK"
