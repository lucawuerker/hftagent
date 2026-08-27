#!/bin/zsh
# Sequential follow-up chain (8 GB box -> one heavy process at a time):
#   wait for wave 1 (LDU8/LDP8/L0WF analyses + pool/snapshot PIT races)
#   -> L1H deep dive (user priority)
#   -> wave 2 (curated-book PIT races)
#   -> re-derive tables + re-render figures
set -x
cd "$(dirname "$0")/.."
PY=./venv/bin/python

while ! grep -q BACKFILL_DONE data/thesis_ablation_backfill.log 2>/dev/null; do
  sleep 60
done

$PY scripts/thesis_ablation_l1h_deepdive.py \
  > data/thesis_l1h_deepdive.log 2>&1 || echo DEEPDIVE_FAILED

zsh scripts/thesis_ablation_backfill2.sh \
  > data/thesis_ablation_backfill2.log 2>&1

$PY scripts/thesis_ablation_derive.py >> data/thesis_ablation_backfill2.log 2>&1
$PY scripts/thesis_ablation_figures.py >> data/thesis_ablation_backfill2.log 2>&1
echo CHAIN_DONE
