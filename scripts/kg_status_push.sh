#!/bin/bash
# Live tracker for the KG breadth campaign -> phone dashboard every 2 min:
#   http://31.97.141.166:8899/wf-2b505d86a0f2/kg.html
set -u
cd "/Users/lucawurker/Desktop/Imperial/Master Thesis/QuantFundAgent" || exit 1
OUT=/tmp/kg.html
while :; do
  scp -q lagias:/root/QuantFundAgent/data/kg_campaign/results.csv /tmp/kg_results.csv 2>/dev/null || true
  python3 - > "$OUT" <<'PY'
import glob, html, json, subprocess, time
from pathlib import Path
camp = Path("data/kg_campaign")
sums = sorted(glob.glob(str(camp / "run_*_summary.json")))
alive = subprocess.run(["pgrep","-f","kg_campaign_chain"],capture_output=True).returncode==0
print("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
      "<meta http-equiv=refresh content=120><title>KG campaign</title>"
      "<body style='background:#111;color:#ddd;font-family:monospace;font-size:13px;padding:12px'>")
print(f"<p><b>KG breadth campaign</b> — {time.strftime('%H:%M:%S UTC',time.gmtime())} — "
      f"chain {'RUNNING' if alive else 'STOPPED'} — runs done: {len(sums)}/20</p>")
tot_new = tot_cost = 0
print("<pre>run  ideas  valid  dedup  kept  cost")
for f in sums:
    s = json.load(open(f))
    tot_new += s.get("n_persisted",0); tot_cost += s.get("llm_cost_usd") or 0
    print(f"{s['run']:>3}  {s.get('n_ideas_requested','?'):>5}  {s.get('n_validated','?'):>5}  "
          f"{s.get('n_deduped','?'):>5}  {s.get('n_persisted','?'):>4}  ${s.get('llm_cost_usd') or 0:.2f}")
print(f"TOTAL new factors: {tot_new}   LLM spend: ${tot_cost:.2f}</pre>")
res = Path("/tmp/kg_results.csv")
if res.exists() and len(res.read_text().splitlines())>1:
    print("<p>WF-IC (mean of 10 block refits, 2021-26):</p><pre>run scope   method  N     mean    hit")
    for line in res.read_text().splitlines()[1:]:
        r,sc,m,n,bm,bs,hit,nb = line.split(",")
        print(f"{r:>3} {sc:<6} {m:<6} {n:>5} {float(bm):+0.4f} {float(hit):.0%}")
    print("</pre>")
tail = Path(camp/"chain.log")
if tail.exists():
    lines=[l for l in tail.read_text().splitlines() if "===" in l or "rc=" in l or "queued" in l][-8:]
    print("<p>chain log:</p><pre>"+html.escape("\n".join(lines))+"</pre>")
PY
  scp -q "$OUT" lagias:/root/status/wf-2b505d86a0f2/kg.html 2>/dev/null || true
  sleep 120
done
