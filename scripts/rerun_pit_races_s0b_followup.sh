#!/bin/bash
# Wait for the running s0b race chain to finish, then relaunch the FIXED
# chain (resume-safe: completed pool blocks are skipped, the two CUR book
# races run for real this time).
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
until grep -q "s0b PIT races DONE" data/rerun_pit_races_s0b.log; do sleep 120; done
echo "$(date -Is) followup: relaunching fixed chain for CUR races" >> data/rerun_pit_races_s0b.log
exec ./scripts/rerun_pit_races_s0b.sh
