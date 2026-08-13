#!/bin/bash
# Autonomous WF post-analysis driver (runs on the Lagias server in tmux).
#
# For every finished WF-ladder arm (orchestrator status ok + factor book on
# disk) plus the 101-alpha zoo:
#   1. scripts/wf_arm_factor_analysis.py  — per-factor block ICs, diversity,
#      static combined fits, prequential record
#   2. scripts/wf_pit_combiner_study.py   — PIT walk-forward combiner race
#      (arm alone and arm+zoo; plus the all-arms union once all members done)
# then rebuilds the cross-arm summary. Rescans every 30 min so arms that
# finish later (L7WF_terra_s0, the s1 arms) are picked up automatically;
# exits once L7WF_terra_s0 has been analysed.
#
# Jobs are independent → run ANALYSIS_PAR of them concurrently (default 4),
# each single-threaded (OMP=1): ~5 busy cores incl. the ladder arm while the
# quota is lifted, and no OpenMP spin-thrash if the 200% quota returns.
set -u
cd /root/QuantFundAgent || exit 1
# if RAM runs out the kernel must sacrifice analysis workers, never the
# ladder arm — children inherit this at fork
echo 500 > /proc/$$/oom_score_adj 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export QF_USE_MCP=0
PY=./venv/bin/python
WS=data/workspaces/fmp_archive_equity_nasdaq100pit/preruns
OUT=data/comparisons/wf_arm_analysis
PAR=${ANALYSIS_PAR:-4}
mkdir -p "$OUT/pit_combiners"
LOG=$OUT/driver.log
echo "$(date -Is) driver start (pid $$, par=$PAR)" >> "$LOG"

finished() {  # arm has an ok status and a materialised factor book
  [ "$1" = zoo ] && return 0
  [ -f "$WS/$1/factors/factor_db.json" ] || return 1
  grep -q '"status": "ok"' "$WS/$1/orchestrator_status.json" 2>/dev/null
}

summarise() {
  $PY - <<'EOF' >> "$LOG" 2>&1
import glob, json, pandas as pd
from pathlib import Path
out = Path("data/comparisons/wf_arm_analysis")
frames = [pd.read_csv(f) for f in sorted(glob.glob(str(out / "pit_combiners/*_summary.csv")))]
lines = ["# WF post-analysis — cross-arm summary", "",
         "PIT walk-forward combiner race: refit at each 126-bar block start on all",
         "prior bars, factors restricted to those existing at that date (evolution",
         "arms: archive snapshot; oneshot/zoo: full book). Statistic = mean of the",
         "10 per-block pooled per-underlying ICs.", ""]
if frames:
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "ALL_PIT_SUMMARY.csv", index=False)
    lines.append("| book | method | WF blockmean | std | hit | blocks | avg N |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["label", "blockmean"], ascending=[True, False]).iterrows():
        lines.append(f"| {r.label} | {r.method} | {r.blockmean:.4f} | "
                     f"{(r.blockstd if pd.notna(r.blockstd) else 0):.4f} | "
                     f"{r.hit:.0%} | {r.n_blocks} | {r.mean_n_factors:.0f} |")
statics = [pd.read_csv(f) for f in sorted(glob.glob(str(out / "*/combined_static.csv")))]
if statics:
    sd = pd.concat(statics, ignore_index=True)
    sd.to_csv(out / "ALL_STATIC_COMBINED.csv", index=False)
    lines += ["", "## Static combined fits (fit once < 2021-07-20)", "",
              "| arm | model | n | IS blockmean | WF blockmean | hit |", "|---|---|---|---|---|---|"]
    for _, r in sd.iterrows():
        lines.append(f"| {r.arm} | {r.model} | {r.n_factors} | "
                     f"{r.ic_is_blockmean:.4f} | {r.ic_wf_blockmean:.4f} | {r.wf_hit_rate:.0%} |")
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("summary rebuilt:", len(frames), "pit files,", len(statics), "static files")
EOF
}

while :; do
  # discover finished arms (evolution WF arms + oneshot books)
  ARMS="zoo"
  for d in "$WS"/L*WF*; do
    a=$(basename "$d")
    finished "$a" && ARMS="$ARMS $a"
  done
  echo "$(date -Is) pass over: $ARMS" >> "$LOG"

  # build the work queue (independent jobs, one line each), LONGEST FIRST:
  # job cost scales with book size, so the union/+zoo monsters must start
  # immediately or they alone define the finish time. Cheap factor analyses
  # are interleaved right after the widest jobs so their REPORTs land early.
  QUEUE=$(mktemp)
  pit() {  # $1 = arm spec, $2 = label
    [ -f "$OUT/pit_combiners/$2.done" ] || echo \
      "nice -n 10 $PY scripts/wf_pit_combiner_study.py --arm '$1' --label '$2' >> '$LOG' 2>&1 && touch '$OUT/pit_combiners/$2.done'" >> "$QUEUE"
  }
  UNION=""
  for a in L2WF_terra_s0 L4WF_terra_s0 L5WF_terra_s0 L6WF_terra_s0; do
    finished "$a" && UNION="${UNION:+$UNION+}$a"
  done
  if [ "$UNION" = "L2WF_terra_s0+L4WF_terra_s0+L5WF_terra_s0+L6WF_terra_s0" ]; then
    pit "$UNION+zoo" "union_wf_s0_plus_zoo"
    pit "$UNION" "union_wf_s0"
  fi
  for a in $ARMS; do            # widest first: oneshot books are the biggest
    case "$a" in L1WF*) pit "$a+zoo" "${a}_plus_zoo";; esac
  done
  for a in $ARMS; do
    [ -f "$OUT/$a/REPORT.md" ] || echo \
      "nice -n 10 $PY scripts/wf_arm_factor_analysis.py --arm '$a' >> '$LOG' 2>&1" >> "$QUEUE"
  done
  for a in $ARMS; do
    case "$a" in zoo|L1WF*) pit "$a" "$a";; esac
  done
  for a in $ARMS; do
    case "$a" in zoo|L1WF*) ;; *) pit "$a+zoo" "${a}_plus_zoo";; esac
  done
  for a in $ARMS; do
    case "$a" in zoo|L1WF*) ;; *) pit "$a" "$a";; esac
  done

  N=$(wc -l < "$QUEUE")
  echo "$(date -Is) queue: $N jobs, running $PAR-wide" >> "$LOG"
  if [ "$N" -gt 0 ]; then
    xargs -P "$PAR" -I{} -d '\n' bash -c '{}' < "$QUEUE"
  fi
  rm -f "$QUEUE"

  summarise

  if [ -f "$OUT/pit_combiners/L7WF_terra_s0.done" ]; then
    echo "$(date -Is) L7WF analysed — driver exiting" >> "$LOG"
    break
  fi
  sleep 1800
done
