#!/bin/bash
# Level-rho profile for L1H_terra_s0b, queued behind the clean LDG races so the
# 8 GB M2 never holds two panels.  Needed to ask whether the LDG-vs-L1H gap
# survives removal of the persistent-level factor class.
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
while pgrep -f clean_ldg_races.sh > /dev/null; do sleep 60; done
export QF_USE_MCP=0 QF_SIGNAL_CACHE_MAX=64 OMP_NUM_THREADS=4
./venv/bin/python scripts/pool_level_profile.py --arm L1H_terra_s0b \
  >> data/level_profile_l1h_terra.log 2>&1
echo "$(date +%FT%T) L1H profile rc=$?" >> data/level_profile_l1h_terra.log
