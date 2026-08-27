"""Post-hoc level-rho profile of an arm's KEPT POOL (median per-name lag-1
autocorrelation per factor signal on the dev window), for runs that predate
the harness's level_rho diagnostic.  Usage: --arm L1H_terra_s0 [--arm ...]
Writes data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv
(appends; one row per arm+factor)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("poolrho")

import numpy as np
import pandas as pd

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
OUT = REPO / "data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    args = ap.parse_args()

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    dev = np.asarray(idx < pd.Timestamp("2021-07-20"))

    rows = []
    for arm in args.arm:
        st = json.load((WS / arm / "evolution/state.json").open())
        seen = set()
        pool = st["kept_pool"] + st["archive"]
        for e in pool:
            prog = e["genome"]["programs"][0]
            fid = prog["factor_id"]
            if fid in seen:
                continue
            seen.add(fid)
            try:
                cls = compile_factor(prog["code"], fid)
                sig = compute_signal(cls, panel).reindex(index=idx,
                                                         columns=close.columns)
            except Exception as ex:  # noqa: BLE001
                log.warning("%s/%s failed (%s)", arm, fid, ex)
                continue
            v = sig.to_numpy(dtype=float)[dev]
            rhos = []
            for j in range(v.shape[1]):
                x = v[:, j]
                ok = np.isfinite(x)
                if ok.sum() < 100:
                    continue
                x = x[ok]
                if np.std(x) == 0:
                    rhos.append(1.0)
                    continue
                r = np.corrcoef(x[:-1], x[1:])[0, 1]
                if np.isfinite(r):
                    rhos.append(r)
            if rhos:
                rows.append({"arm": arm, "factor_id": fid,
                             "rho_med": float(np.median(rhos))})
        got = [r for r in rows if r["arm"] == arm]
        med = np.median([r["rho_med"] for r in got]) if got else float("nan")
        n9 = sum(r["rho_med"] > 0.9 for r in got)
        n995 = sum(r["rho_med"] > 0.995 for r in got)
        log.info("%s: pool n=%d median rho %.3f, >0.9: %d, >0.995: %d",
                 arm, len(got), med, n9, n995)

    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.arm.isin(df.arm.unique())], df])
    df.to_csv(OUT, index=False)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
