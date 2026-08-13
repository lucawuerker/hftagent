#!/bin/bash
# Live WF post-analysis dashboard, rendered every 60 s to the phone httpd:
#   http://<host>:8899/wf-2b505d86a0f2/analysis.html
# Shows per-job progress (factor analyses + PIT combiner runs: blocks done,
# current best method), the L7WF_terra_s0 ladder arm, memory, and the full
# cross-arm SUMMARY.md once available.
set -u
OUT=/root/QuantFundAgent/data/comparisons/wf_arm_analysis
WS=/root/QuantFundAgent/data/workspaces/fmp_archive_equity_nasdaq100pit/preruns
WEB=/root/status/wf-2b505d86a0f2
while :; do
  python3 - "$OUT" "$WS" "$WEB" <<'EOF' || true
import glob, html, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

out, ws, web = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if not web.is_dir():
    sys.exit(0)
now = time.time()
stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

ARMS = ["zoo", "L1WF_oneshot_terra_s0", "L2WF_terra_s0", "L4WF_terra_s0",
        "L5WF_terra_s0", "L6WF_terra_s0", "L7WF_terra_s0"]
PITS = list(dict.fromkeys(
    ["union_wf_s0_plus_zoo", "union_wf_s0", "L1WF_oneshot_terra_s0_plus_zoo"]
    + ARMS[:6]
    + [f"{a}_plus_zoo" for a in ARMS[1:6]]
    + ["L7WF_terra_s0", "L7WF_terra_s0_plus_zoo"]))

# labels of PIT jobs that are ACTUALLY running right now (exact, via ps)
running_labels = set()
try:
    ps = subprocess.run(["pgrep", "-af", "wf_pit_combiner_study"],
                        capture_output=True, text=True, timeout=10).stdout
    for line in ps.splitlines():
        if "--label" in line:
            running_labels.add(line.split("--label")[1].split()[0].strip("'\""))
except Exception:  # noqa: BLE001
    pass

def age(p):
    try:
        return now - p.stat().st_mtime
    except OSError:
        return None

rows_fact = []
for a in ARMS[:6] + ["L7WF_terra_s0"]:
    d = out / a
    if (d / "REPORT.md").exists():
        st = "done"
    elif d.exists() and any((age(f) or 9e9) < 300 for f in d.glob("*")):
        st = "running"
    elif a == "L7WF_terra_s0":
        st = "waits for run"
    else:
        st = "pending"
    rows_fact.append((a, st))

rows_pit = []
for label in PITS:
    j = out / "pit_combiners" / f"{label}.jsonl"
    done = (out / "pit_combiners" / f"{label}.done").exists()
    live = label in running_labels
    if not j.exists():
        st = ("done" if done else
              "running (signals)" if live else "queued")
        rows_pit.append((label, st, "", "", ""))
        continue
    per_m = defaultdict(list)
    n_avail = ""
    for line in j.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ic") is not None:
            per_m[r["method"]].append((r["block_gen"], r["ic"]))
        n_avail = r.get("n_factors_avail", n_avail)
    if not per_m:
        rows_pit.append((label, "running" if live else "queued", "", "", ""))
        continue
    max_block = max(g for v in per_m.values() for g, _ in v)
    n_rows = sum(len(v) for v in per_m.values())
    best = max(per_m.items(),
               key=lambda kv: sum(ic for _, ic in kv[1]) / len(kv[1]))
    best_txt = (f"{best[0]} {sum(ic for _, ic in best[1]) / len(best[1]):+.4f}"
                f" ({len(best[1])} bl)")
    st = ("done" if done else
          "running" if live else "queued (partial)")
    rows_pit.append((label, st, f"g{max_block}/g20", str(n_rows), best_txt))

# ladder arm
l7 = "?"
try:
    gq = (ws / "L7WF_terra_s0/evolution/gen_quality.jsonl")
    gen = json.loads(gq.read_text().splitlines()[-1]).get("generation", "?")
    usage = json.loads((ws / "L7WF_terra_s0/evolution/llm_usage.json").read_text())
    cost = usage.get("total_cost_usd") or usage.get("cost_usd") or "?"
    l7 = f"generation {gen}/20, spent ${cost:.2f}" if isinstance(cost, float) else f"generation {gen}/20"
except Exception as e:  # noqa: BLE001
    l7 = f"status unreadable ({type(e).__name__})"

mem = ""
try:
    mem = subprocess.run(["free", "-g"], capture_output=True, text=True,
                         timeout=10).stdout.splitlines()
    mem = " | ".join(x.split()[0] + " " + "/".join(x.split()[2:4])
                     for x in mem[1:3])
except Exception:  # noqa: BLE001
    pass

CSS = ("body{background:#111;color:#ddd;font-family:monospace;font-size:13px;"
       "margin:12px}table{border-collapse:collapse;margin:8px 0}"
       "td,th{border:1px solid #333;padding:3px 8px;text-align:left}"
       "th{color:#8cf}.done{color:#7d7}.running{color:#fd7}"
       ".pending{color:#777}.queued{color:#777}"
       "a{color:#8cf}h2{font-size:14px;margin:14px 0 4px}")

def tr(cells, cls=None):
    tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
    return f"<tr class='{cls}'>{tds}</tr>" if cls else f"<tr>{tds}</tr>"

parts = [
    f"<!doctype html><meta charset=utf-8><meta name=viewport "
    f"content='width=device-width,initial-scale=1'>"
    f"<meta http-equiv=refresh content=60><title>WF analysis live</title>"
    f"<style>{CSS}</style><body>",
    f"<p>rendered {stamp} — refresh 60s — mem(u/f)GB: {html.escape(mem)}"
    f" — <a href='analysis_pit.csv'>pit csv</a></p>",
    f"<h2>ladder (only remaining run)</h2><p>L7WF_terra_s0: {html.escape(l7)}"
    f"; s1 arms ON HOLD</p>",
    "<h2>PIT combiner runs</h2><table><tr><th>book</th><th>state</th>"
    "<th>block</th><th>rows</th><th>best method (blockmean)</th></tr>"]
for label, st, blk, nr, best in rows_pit:
    parts.append(tr([label, st, blk, nr, best], cls=st.split()[0]))
parts.append("</table><h2>factor analyses</h2><table>"
             "<tr><th>arm</th><th>state</th></tr>")
for a, st in rows_fact:
    parts.append(tr([a, st], cls=st.split()[0]))
parts.append("</table>")
summ = out / "SUMMARY.md"
if summ.exists():
    parts.append("<h2>cross-arm summary (rebuilt each driver pass)</h2>"
                 f"<pre style='overflow-x:auto'>{html.escape(summ.read_text())}"
                 "</pre>")
(web / "analysis.html").write_text("".join(parts))
allp = out / "ALL_PIT_SUMMARY.csv"
if allp.exists():
    import shutil
    shutil.copy(allp, web / "analysis_pit.csv")
EOF
  sleep 60
done
