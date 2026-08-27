"""Clean pool (rho < 0.9) of the KG-campaign cumulative book.

For every factor in data/kg_campaign/cumulative_book.json: ensure its signal
parquet exists in the shared signal store (compute locally if the lagias sync
did not cover it), then compute rho_med = median per-name lag-1
autocorrelation on the dev window (< 2021-07-20) and keep rho_med < 0.9
(the CLN-race convention).  Writes:

  data/kg_campaign/clean_book.json        [{factor_id, signal_key, rho_med}]
  data/kg_campaign/clean_pool_profiles.csv one row per factor (kept or not)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_clean_pool")

import numpy as np
import pandas as pd

from wf_common import SIGNAL_STORE, load_or_compute_signal, signal_key

CAMP = REPO / "data/kg_campaign"
MAX_RHO = 0.9
DEV_END = pd.Timestamp("2021-07-20")


def col_lag1_rho(v: np.ndarray) -> list[float]:
    """Per-column lag-1 autocorrelation with pairwise-finite handling."""
    rhos = []
    a, b = v[:-1], v[1:]
    m = np.isfinite(a) & np.isfinite(b)
    for j in range(v.shape[1]):
        mj = m[:, j]
        if mj.sum() < 100:
            continue
        x, y = a[mj, j], b[mj, j]
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            rhos.append(1.0)
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r):
            rhos.append(r)
    return rhos


def main() -> None:
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx, cols = close.index, close.columns
    dev = np.asarray(idx < DEV_END)

    cum = json.loads((CAMP / "cumulative_book.json").read_text())
    codes: dict[str, str] = {}
    for e in cum:
        p = Path(e["code_path"])
        if not p.exists():
            p = REPO / "quant_fund_agent/factors/researcher" / p.name
        if p.exists():
            codes[e["factor_id"]] = p.read_text()

    rows, kept = [], []
    n_fail = 0
    for i, (fid, code) in enumerate(sorted(codes.items())):
        key = signal_key(fid, code)
        p = SIGNAL_STORE / f"{key}.parquet"
        try:
            if p.exists():
                sig = pd.read_parquet(p).reindex(index=idx, columns=cols)
            else:
                sig = load_or_compute_signal(fid, code, panel, idx, cols)
        except Exception as ex:  # noqa: BLE001
            log.warning("%s failed: %s", fid, ex)
            n_fail += 1
            continue
        v = sig.to_numpy(dtype=np.float32)[dev]
        rhos = col_lag1_rho(v)
        if not rhos:
            n_fail += 1
            rows.append({"factor_id": fid, "rho_med": "", "kept": 0})
            continue
        rho = float(np.median(rhos))
        keep = rho < MAX_RHO
        rows.append({"factor_id": fid, "rho_med": f"{rho:.4f}",
                     "kept": int(keep)})
        if keep:
            kept.append({"factor_id": fid, "signal_key": key,
                         "rho_med": rho})
        if (i + 1) % 200 == 0:
            log.info("%d/%d profiled (kept %d, failed %d)",
                     i + 1, len(codes), len(kept), n_fail)

    with (CAMP / "clean_pool_profiles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["factor_id", "rho_med", "kept"])
        w.writeheader()
        w.writerows(rows)
    (CAMP / "clean_book.json").write_text(json.dumps(kept, indent=1))
    log.info("DONE: %d/%d kept (rho<%.1f), %d failed/degenerate",
             len(kept), len(codes), MAX_RHO, n_fail)


if __name__ == "__main__":
    main()
