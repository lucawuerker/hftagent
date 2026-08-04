#!/usr/bin/env python
"""Live status of an ablation-matrix run — text table + optional HTML page.

Reads only the on-disk checkpoints the orchestrator/entrypoints already write
(orchestrator_status.json, evolution/state.json, evolution/llm_usage.json,
evolution/prequential.jsonl, the per-arm logs), so it is safe to run at any
time from anywhere::

    ./venv/bin/python scripts/matrix_status.py --plan matrix/terra_wf_ladder.yaml
    ./venv/bin/python scripts/matrix_status.py --plan ... --html /root/status/x/index.html

On the run server a tmux loop regenerates the HTML every minute and a
``python3 -m http.server`` serves it, so the page is readable from a phone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as html_mod
import json
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNNING_HEARTBEAT_SEC = 20 * 60


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — a half-written checkpoint must not kill status
        return {}


def _process_alive(arm_name: str) -> bool:
    """True when an entrypoint process for this arm is running on this host."""
    import subprocess
    try:
        # "--" stops pgrep's own option parsing (the pattern starts with "--")
        return subprocess.run(
            ["pgrep", "-f", "--", f"-name {arm_name}"],
            capture_output=True, timeout=5).returncode == 0
    except Exception:  # noqa: BLE001 — status must never crash on a probe
        return False


def _mtime_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _fmt_age(sec: float | None) -> str:
    if sec is None:
        return "-"
    if sec < 90:
        return f"{int(sec)}s"
    if sec < 5400:
        return f"{int(sec / 60)}m"
    return f"{sec / 3600:.1f}h"


def arm_status(sdir: Path, arm: dict, plan: dict) -> dict:
    name = arm["name"]
    entry = arm.get("entrypoint", "evolution")
    orch = _read_json(sdir / "orchestrator_status.json")
    state = _read_json(sdir / "evolution" / "state.json")
    usage = _read_json(sdir / "evolution" / "llm_usage.json")
    if not usage and entry == "oneshot":
        usage = (_read_json(sdir / "manifest.json").get("llm_usage") or {})
    cost = float((usage.get("total") or {}).get("cost_usd", 0.0))

    gens_total = (arm.get("flags", {}).get("generations")
                  or (plan.get("defaults") or {}).get("generations"))
    gen = state.get("generation")
    archive = len(state.get("archive") or [])
    trials = state.get("n_trials")

    preq_rows: list[dict] = []
    preq_path = sdir / "evolution" / "prequential.jsonl"
    if preq_path.exists():
        for line in preq_path.read_text().splitlines():
            try:
                preq_rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    scored = [r for r in preq_rows if r.get("combined_oos_ic") is not None]
    wf_blocks = 0
    if entry == "evolution":
        wf_blocks = (arm.get("flags", {}).get("wf-blocks")
                     or (plan.get("defaults") or {}).get("wf-blocks") or 0)
    p1_gens = (gens_total - wf_blocks) if (gens_total and wf_blocks) else None
    wf_scored = ([r for r in scored if p1_gens and r["generation"] > p1_gens]
                 if p1_gens else [])

    # Heartbeat: freshest of the arm's checkpoint writes.  The transcript and
    # the orchestrator's per-arm log are the ONLY live signals for a oneshot
    # arm (it writes manifest.json/llm_usage only at the end).
    heartbeat = min((a for a in (
        _mtime_age(sdir / "evolution" / "state.json"),
        _mtime_age(sdir / "evolution" / "llm_usage.json"),
        _mtime_age(sdir / "evolution" / "lineage.jsonl"),
        _mtime_age(sdir / "evolution" / "llm_transcript.jsonl"),
        _mtime_age(REPO / "data" / "comparisons" / "final_matrix" / "logs"
                   / f"{name}.log"),
        _mtime_age(sdir / "manifest.json"),
    ) if a is not None), default=None)

    if orch.get("status") == "ok":
        status = "ok"
    elif orch.get("status") == "failed":
        status = "FAILED"
    elif _process_alive(name):
        # The strongest signal: the arm's entrypoint process exists.  Long
        # silent evaluation phases (~1h without a log/checkpoint write) are
        # normal, so file ages alone must not demote a live arm.
        status = "running"
    elif heartbeat is not None and heartbeat < RUNNING_HEARTBEAT_SEC:
        status = "running"
    elif heartbeat is not None:
        status = "stalled?"
    else:
        status = "pending"

    return {
        "name": name, "entry": entry, "status": status,
        "gen": gen, "gens_total": gens_total,
        "archive": archive, "trials": trials, "cost": cost,
        "heartbeat": heartbeat,
        "elapsed_min": (orch.get("elapsed_sec") or 0) / 60 or None,
        "preq_n": len(scored),
        "preq_last": scored[-1].get("combined_oos_ic") if scored else None,
        "preq_last_window": (f"{scored[-1]['start'][:10]}→{scored[-1]['end'][:10]}"
                             if scored else None),
        "wf_n": len(wf_scored), "wf_blocks": wf_blocks or None,
        "wf_mean": (sum(r["combined_oos_ic"] for r in wf_scored) / len(wf_scored)
                    if wf_scored else None),
    }


def collect(plan_path: Path, scope_root: Path) -> dict:
    plan = yaml.safe_load(plan_path.read_text())
    rows = [arm_status(scope_root / a["name"], a, plan) for a in plan["arms"]]
    spent = sum(r["cost"] for r in rows)
    return {"plan": plan_path.name, "budget": plan.get("budget_usd"),
            "spent": spent, "rows": rows,
            "generated": dt.datetime.now().isoformat(timespec="seconds")}


def render_text(rep: dict) -> str:
    out = [f"plan {rep['plan']}   spent ${rep['spent']:.2f}"
           + (f" of ${rep['budget']:.0f}" if rep["budget"] else "")
           + f"   ({rep['generated']})"]
    hdr = (f"{'arm':<24}{'status':<10}{'gen':<8}{'archive':<9}{'trials':<8}"
           f"{'cost$':<9}{'wf-blocks':<11}{'last preq IC':<24}{'beat':<6}")
    out += [hdr, "-" * len(hdr)]
    for r in rep["rows"]:
        gen = (f"{r['gen']}/{r['gens_total']}"
               if r["gen"] is not None and r["gens_total"] else (r["gen"] or "-"))
        wf = (f"{r['wf_n']}/{r['wf_blocks']}" if r["wf_blocks"] else "-")
        ic = (f"{r['preq_last']:+.4f} {r['preq_last_window']}"
              if r["preq_last"] is not None else "-")
        out.append(f"{r['name']:<24}{r['status']:<10}{gen:<8}{r['archive']:<9}"
                   f"{r['trials'] if r['trials'] is not None else '-':<8}"
                   f"{r['cost']:<9.2f}{wf:<11}{ic:<24}{_fmt_age(r['heartbeat']):<6}")
    return "\n".join(out)


_CSS = """
body{font-family:-apple-system,system-ui,sans-serif;margin:1rem;background:#fff;color:#111}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}
 table td,table th{border-color:#444!important}.bar{background:#333!important}}
h2{margin:.2rem 0}small{opacity:.7}
table{border-collapse:collapse;width:100%;margin-top:.8rem;font-size:.85rem}
td,th{border:1px solid #ccc;padding:.35rem .5rem;text-align:left;white-space:nowrap}
.ok{color:#2e7d32;font-weight:600}.FAILED{color:#c62828;font-weight:700}
.running{color:#1565c0;font-weight:600}.stalled\\?{color:#e65100;font-weight:600}
.pending{opacity:.55}
.bar{background:#eee;border-radius:4px;height:10px;margin:.3rem 0;max-width:480px}
.bar>div{background:#1565c0;height:10px;border-radius:4px}
.wrap{overflow-x:auto}
"""


def render_html(rep: dict) -> str:
    pct = (100 * rep["spent"] / rep["budget"]) if rep.get("budget") else 0
    rows = []
    for r in rep["rows"]:
        gen = (f"{r['gen']}/{r['gens_total']}"
               if r["gen"] is not None and r["gens_total"] else (r["gen"] or "–"))
        wf = (f"{r['wf_n']}/{r['wf_blocks']}" if r["wf_blocks"] else "–")
        ic = (f"{r['preq_last']:+.4f}<br><small>{r['preq_last_window']}</small>"
              if r["preq_last"] is not None else "–")
        wfm = f"{r['wf_mean']:+.4f}" if r["wf_mean"] is not None else "–"
        rows.append(
            f"<tr><td>{html_mod.escape(r['name'])}</td>"
            f"<td class='{r['status']}'>{r['status']}</td><td>{gen}</td>"
            f"<td>{r['archive'] or '–'}</td><td>{r['trials'] or '–'}</td>"
            f"<td>${r['cost']:.2f}</td><td>{wf}</td><td>{wfm}</td><td>{ic}</td>"
            f"<td>{_fmt_age(r['heartbeat'])}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>WF ladder status</title>
<style>{_CSS}</style></head><body>
<h2>Terra WF ladder</h2>
<small>{rep['generated']} · auto-refresh 60s</small>
<div>spent <b>${rep['spent']:.2f}</b> of ${rep['budget']:.0f}
<div class="bar"><div style="width:{min(pct, 100):.1f}%"></div></div></div>
<div class="wrap"><table>
<tr><th>arm</th><th>status</th><th>gen</th><th>archive</th><th>trials</th>
<th>cost</th><th>WF blocks</th><th>WF mean IC</th><th>last preq IC</th><th>beat</th></tr>
{''.join(rows)}
</table></div>
<p><small>WF blocks = prequentially traded ~6-month forward blocks (honest OOS).
"beat" = age of the arm's newest checkpoint write.</small></p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="matrix/terra_wf_ladder.yaml")
    ap.add_argument("--scope-dir",
                    default="data/workspaces/fmp_archive_equity_nasdaq100pit/preruns")
    ap.add_argument("--html", default=None,
                    help="Also write a self-refreshing HTML page here.")
    args = ap.parse_args()
    rep = collect(REPO / args.plan if not Path(args.plan).is_absolute()
                  else Path(args.plan),
                  REPO / args.scope_dir if not Path(args.scope_dir).is_absolute()
                  else Path(args.scope_dir))
    print(render_text(rep))
    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(render_html(rep))
        tmp.replace(out)


if __name__ == "__main__":
    main()
