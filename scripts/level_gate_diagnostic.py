"""Stationarity ("level-signal") diagnostic over existing factor books.

Motivated by the L0WF finding (2026-08-09): the GP arm's record was driven by
``log(high)`` — a raw price level whose per-underlying IC telescopes short-
horizon mean reversion into a huge spurious statistic. This script MEASURES
(never re-runs) every archive factor of the chosen books on the dev window:

  * rho_med   — median across names of the signal's lag-1 autocorrelation
  * halflife  — ln2 / (1 - rho_med), capped at 10_000 bars
  * lvl_corr  — median across names of |corr(signal, log close)| (price-level
                proxy detector)

and reports, per book, how many factors a candidate level gate would reject
(rho_med thresholds 0.995 / 0.999, halflife > 126). Output:
``data/comparisons/wf_arm_analysis/level_gate_diagnostic.csv`` + stdout table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("level_gate")

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
DEV_END = "2021-07-20"


def book_codes(arm: str) -> dict[str, str]:
    if arm == "zoo":
        pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
        return {m["factor_id"]: m["code"] for m in pb["members"]}
    db = json.loads((WS / arm / "factors/factor_db.json").read_text())
    codes = {}
    state_codes = {}
    for sub in ("evolution", "gp"):
        st = WS / arm / sub / "state.json"
        if st.exists():
            for eg in json.loads(st.read_text()).get("kept_pool", []):
                for prog in eg["genome"]["programs"]:
                    state_codes[prog["factor_id"]] = prog["code"]
            break
    for r in db["factors"]:
        p = Path(r["code_path"])
        if not p.exists():
            p = REPO / "quant_fund_agent/factors/researcher" / p.name
        if p.exists():
            codes[r["id"]] = p.read_text()
        elif r["id"] in state_codes:
            codes[r["id"]] = state_codes[r["id"]]
    return codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="zoo,L1WF_oneshot_terra_s0,L2WF_terra_s0,"
                    "L4WF_terra_s0,L5WF_terra_s0,L6WF_terra_s0,L7WF_terra_s0,"
                    "L1H_terra_s0,L0WF_gp_s0")
    ap.add_argument("--out", default=str(
        REPO / "data/comparisons/wf_arm_analysis/level_gate_diagnostic.csv"))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from wf_common import load_or_compute_signal
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    dev = np.asarray(idx < pd.Timestamp(DEV_END))
    logc = np.log(close.where(close > 0)).to_numpy()[dev]

    rows = []
    for arm in args.arms.split(","):
        codes = book_codes(arm)
        log.info("[%s] %d factors", arm, len(codes))
        for fid, code in codes.items():
            try:
                sig = load_or_compute_signal(fid, code, panel, idx,
                                             close.columns).astype(float)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s failed: %s", arm, fid, e)
                continue
            x = sig.to_numpy()[dev]
            rhos, lvls = [], []
            for j in range(x.shape[1]):
                v = x[:, j]
                m = np.isfinite(v)
                if m.sum() < 100:
                    continue
                v = v[m]
                if np.std(v) < 1e-12:
                    rhos.append(1.0)
                    continue
                a, b = v[:-1] - v[:-1].mean(), v[1:] - v[1:].mean()
                den = np.sqrt((a * a).sum() * (b * b).sum())
                if den > 0:
                    rhos.append(float((a * b).sum() / den))
                lc = logc[m, j] if m.shape[0] == logc.shape[0] else None
                if lc is not None and np.isfinite(lc).all() and np.std(lc) > 0:
                    c = np.corrcoef(v, lc)[0, 1]
                    if np.isfinite(c):
                        lvls.append(abs(float(c)))
            if not rhos:
                continue
            rho = float(np.median(rhos))
            hl = min(10_000.0, float(np.log(2) / max(1e-6, 1.0 - rho)))
            rows.append({"arm": arm, "factor_id": fid, "rho_med": round(rho, 5),
                         "halflife": round(hl, 1),
                         "lvl_corr": round(float(np.median(lvls)), 3) if lvls else None,
                         "n_names": len(rhos)})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summ = (df.assign(f995=df.rho_med > 0.995, f999=df.rho_med > 0.999,
                      fhl=df.halflife > 126)
              .groupby("arm")
              .agg(n=("factor_id", "size"), rho_median=("rho_med", "median"),
                   fail_rho995=("f995", "sum"), fail_rho999=("f999", "sum"),
                   fail_hl126=("fhl", "sum")))
    log.info("\n%s", summ.to_string())
    log.info("wrote %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main()
