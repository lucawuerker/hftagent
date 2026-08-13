#!/bin/bash
# Push the local L1H arm's progress to the phone dashboard every 3 min:
#   http://31.97.141.166:8899/wf-2b505d86a0f2/l1h_local.html
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
OUT=/tmp/l1h_local.html
while :; do
  python3 - > "$OUT" <<'EOF'
import html, json, subprocess, time
from pathlib import Path
evo = Path("data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/L1H_terra_s0/evolution")
rows = []
gq = evo / "gen_quality.jsonl"
if gq.exists():
    rows = [json.loads(l) for l in gq.read_text().splitlines()]
cost = "?"
try:
    cost = f"${json.load(open(evo / 'llm_usage.json'))['total']['cost_usd']:.2f}"
except Exception:
    pass
prq = evo / "prequential.jsonl"
prq_rows = []
if prq.exists():
    prq_rows = [json.loads(l) for l in prq.read_text().splitlines()]
alive = subprocess.run(["pgrep", "-f", "run_factor_evolution.py --name L1H"],
                       capture_output=True).returncode == 0
log_tail = ""
lg = Path("data/l1h_local_run.log")
if lg.exists():
    log_tail = "\n".join(lg.read_text().splitlines()[-12:])
print("<!doctype html><meta charset=utf-8><meta name=viewport "
      "content='width=device-width,initial-scale=1'>"
      "<meta http-equiv=refresh content=180><title>L1H local</title>"
      "<body style='background:#111;color:#ddd;font-family:monospace;font-size:13px;padding:12px'>")
print(f"<p>L1H_terra_s0 (local M2) — rendered "
      f"{time.strftime('%H:%M:%S UTC', time.gmtime())} — "
      f"{'RUNNING' if alive else 'NOT RUNNING'} — spend {cost}</p>")
if rows:
    r = rows[-1]
    print(f"<p>generation <b>{r.get('generation')}</b>/20 — archive "
          f"{r.get('archive_size_total')} — kept {r.get('kept_pool_size')} — "
          f"trials {r.get('n_trials')}</p>")
if prq_rows:
    print("<p>prequential (honest OOS per revealed block):</p><pre>")
    for r in prq_rows[-12:]:
        ic = r.get("combined_oos_ic")
        print(html.escape(f"g{r.get('generation'):>2}  {str(r.get('start'))[:10]} -> "
              f"{str(r.get('end'))[:10]}  IC {ic:+.4f}" if ic is not None else
              f"g{r.get('generation'):>2}  skipped: {r.get('skipped')}"))
    print("</pre>")
print(f"<p>log tail:</p><pre>{html.escape(log_tail)}</pre>")
EOF
  scp -q "$OUT" lagias:/root/status/wf-2b505d86a0f2/l1h_local.html 2>/dev/null
  sleep 180
done
