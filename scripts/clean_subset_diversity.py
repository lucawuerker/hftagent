#!/usr/bin/env python
"""Diversity (mean |rho|, participation-ratio effective N) for an arbitrary
SUBSET of an arm's factors, using exactly the convention of
scripts/wf_arm_factor_analysis.py (shared parquet signal store, fit window
< 2021-07-20, stride-thinned to <=400 rows, inf-safe).

Usage: --arm LDG_4omini_s0b --max-rho 0.7 [--scope book|pool]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import os
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

from wf_common import load_or_compute_signal            # noqa: E402
from wf_arm_factor_analysis import WF_START, load_book  # noqa: E402

PROFILES = REPO / "data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--max-rho", type=float, default=0.7)
    ap.add_argument("--scope", choices=["book", "pool"], default="book")
    args = ap.parse_args()

    prof = pd.read_csv(PROFILES)
    rho = prof[prof.arm == args.arm].set_index("factor_id")["rho_med"]
    clean = set(rho[rho < args.max_rho].index)

    if args.scope == "book":
        codes = load_book(args.arm)
    else:
        st = json.loads((WS / args.arm / "evolution/state.json").read_text())
        codes = {}
        for e in st.get("kept_pool", []) + st.get("archive", []):
            for p in e["genome"]["programs"]:
                codes.setdefault(p["factor_id"], p["code"])
    codes = {f: c for f, c in codes.items() if f in clean}
    print(f"{args.arm} {args.scope}: {len(codes)} factors with rho<{args.max_rho}")

    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.data import usable_fields
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx, cols = close.index, close.columns
    fit_mask = np.asarray(idx < pd.Timestamp(WF_START))
    fit_pos = np.flatnonzero(fit_mask)

    sigs = []
    for fid, code in sorted(codes.items()):
        try:
            sig = load_or_compute_signal(fid, code, panel, idx, cols)
            if sig is None:
                continue
            sigs.append(sig[0] if isinstance(sig, tuple) else sig)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [skip] {fid}: {exc}")
    print(f"  usable signals: {len(sigs)}")

    stride = max(1, int(fit_mask.sum()) // 400)
    rows_sel = fit_pos[::stride]
    mat = np.column_stack([
        np.nan_to_num(s.astype(float).to_numpy()[rows_sel].ravel()) for s in sigs])
    c = np.corrcoef(mat, rowvar=False)
    n = c.shape[0]
    off = c[np.triu_indices(n, 1)]
    off = off[np.isfinite(off)]
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(c)), 0, None)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    print(json.dumps({"n_factors": n, "mean_abs_corr": float(np.mean(np.abs(off))),
                      "max_abs_corr": float(np.max(np.abs(off))),
                      "effective_n_participation_ratio": pr}, indent=2))


if __name__ == "__main__":
    main()
