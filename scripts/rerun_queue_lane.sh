#!/bin/bash
# Shared replication queue for chapter arms 1 (LDU8), 4 (L1H), 6 (L4WF):
# target 5 runs per arm (user 2026-08-22).  ONE script for both boxes —
# `rerun_queue_lane.sh local` on the M2, `rerun_queue_lane.sh lagias` on the
# server.  Jobs are claimed atomically through a directory on lagias
# (mkdir data/rerun_queue/<job>; the M2 claims over ssh), so whichever box is
# free pulls the next job and the faster box automatically takes more.
# Jobs pinned to a box (a checkpoint that already lives there) are only
# claimed by that box.  Each lane first resumes its pinned L4WF run.
# Completion markers (data/rerun_queue/<job>/{lane,COMPLETE,rc}) drive
# scripts/rerun_seeds_postanalysis.sh.
#
# Specs byte-identical to the originals except --seed; graphrag arms pin
# QF_GRAPH_PATH to the frozen ladder snapshot.  L4WF cap raised to $300 (a
# full L4WF costs ~$250-280; the s0 usage file undercounted across resumes).
# NOTE: on resume --max-cost-usd is ADDED to the preloaded prior spend, so the
# two pinned resumes carry the remainder (~$140-150) rather than the full cap.
set -u
LANE="${1:?local|lagias}"
if [ "$LANE" = lagias ]; then
  cd /root/QuantFundAgent || exit 1
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  QDIR=data/rerun_queue
  claim () { mkdir "$QDIR/$1" 2>/dev/null && echo "$LANE" > "$QDIR/$1/lane"; }
  mark  () { echo "$2" > "$QDIR/$1/rc"; [ "$2" = 0 ] && touch "$QDIR/$1/COMPLETE"; }
  launch () { systemd-run --quiet --collect --scope --slice=lagias-research.slice nice -n 15 ./venv/bin/python -u "$@"; }
else
  cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QF_SIGNAL_CACHE_MAX=64
  QDIR=/root/QuantFundAgent/data/rerun_queue
  claim () { ssh -n -o ConnectTimeout=30 lagias "mkdir $QDIR/$1 2>/dev/null && echo $LANE > $QDIR/$1/lane"; }
  mark  () { ssh -n -o ConnectTimeout=30 lagias "echo $2 > $QDIR/$1/rc; [ $2 = 0 ] && touch $QDIR/$1/COMPLETE"; }
  launch () { nice -n 5 ./venv/bin/python "$@"; }
fi
export PYTHONUNBUFFERED=1 QF_USE_MCP=0
export QF_GRAPH_PATH=data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json
LOG=data/rerun_queue_${LANE}.log
RSS_LIMIT_KB=5400000
echo "$(date -Is) queue lane $LANE start (pid $$)" >> "$LOG"

# job table: name seed cap kind pin      kind: evo (L4WF) | seed (children 0)
JOBS="
L4WF_terra_s1 1 140 L4WF lagias
L4WF_terra_s2 2 150 L4WF local
LDU8_terra_s3 3 20 LDU8 -
L1H_terra_s2 2 40 L1H -
L4WF_terra_s3 3 300 L4WF -
LDU8_terra_s4 4 20 LDU8 -
L1H_terra_s3 3 40 L1H -
L4WF_terra_s4 4 300 L4WF -
"

arm_flags () {
  case "$1" in
    L4WF) echo "--children-per-deme 2 --retrieval graphrag --graph-readonly" ;;
    L1H)  echo "--children-per-deme 0 --retrieval graphrag --graph-readonly" ;;
    LDU8) echo "--children-per-deme 0 --retrieval none --neutral-groups" ;;
  esac
}

run_job () {  # name seed cap kind
  local name="$1" seed="$2" cap="$3" kind="$4" rc=1
  echo "$(date -Is) === $name (seed $seed, cap $cap) start ===" >> "$LOG"
  for i in $(seq 1 200); do
    launch run_factor_evolution.py \
      --name "$name" --config quant.config.nasdaq100_2010_wf.yaml \
      --seed "$seed" --model gpt-5.6-terra --llm-provider openai \
      --generations 20 --population 16 \
      --mechanism-groups 8 --mechanism-groups-mode max \
      --demes-per-group 3 --seed-ideas-per-group 12 \
      --progressive-reveal --reveal-every 1 --test-frac 0.0 \
      --wf-blocks 10 --wf-block-bars 126 \
      --curation archive --selection-deflation on \
      --archive-cap 40 --creative-frac 0.1 --marginal-model lightgbm \
      --fixed-book data/prebooks/formulaic_101.json \
      --reference-book data/prebooks/formulaic_101.json \
      --n-tickers 0 --horizon 6 --llm-workers 8 \
      --max-cost-usd "$cap" $(arm_flags "$kind") >> "$LOG" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      if [ "$LANE" = local ]; then
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "${rss:-}" ] && [ "$rss" -gt "$RSS_LIMIT_KB" ]; then
          echo "$(date -Is) WATCHDOG rss ${rss}KB — TERM $pid" >> "$LOG"
          kill -TERM "$pid"; sleep 30; kill -KILL "$pid" 2>/dev/null
        fi
      fi
    done
    wait "$pid"; rc=$?
    echo "$(date -Is) $name exited rc=$rc (pass $i)" >> "$LOG"
    case $rc in
      0) echo "$(date -Is) $name COMPLETE" >> "$LOG"; break ;;
      3|4) echo "$(date -Is) $name terminal rc=$rc" >> "$LOG"; break ;;
    esac
    sleep 30
  done
  mark "$name" "$rc"
}

while true; do
  got=0
  while read -r name seed cap kind pin; do
    [ -z "$name" ] && continue
    if [ "$pin" != "-" ] && [ "$pin" != "$LANE" ]; then continue; fi
    if claim "$name"; then
      run_job "$name" "$seed" "$cap" "$kind"
      got=1; break   # re-scan from the top so pinned/ordered jobs come first
    fi
  done <<< "$JOBS"
  [ "$got" = 0 ] && break
done
echo "$(date -Is) queue lane $LANE: no unclaimed jobs left — done" >> "$LOG"
