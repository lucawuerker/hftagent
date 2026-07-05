#!/usr/bin/env python3
"""Build a Lasso-selected fixed factor book from a workspace."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="data/workspaces/yfinance_equity_sp100")
    p.add_argument("--output", default=None)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--n-tickers", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=50_000)
    p.add_argument("--max-members", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--fields", default="open,high,low,close,volume")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))

    logging.basicConfig(level=logging.INFO, format="%(name)-24s %(message)s")

    from quant_fund_agent.data import load_panel
    from quant_fund_agent.research_eval.prebook import (
        fit_lasso_prebook,
        load_workspace_programs,
        save_prebook,
    )

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    programs, dropped_at_load = load_workspace_programs(args.workspace)
    panel = load_panel(args.data_dir, fields=fields, n_tickers=args.n_tickers)
    prebook = fit_lasso_prebook(
        programs,
        panel,
        target_horizon=args.horizon,
        max_rows=args.max_rows,
        max_members=args.max_members,
        seed=args.seed,
    )
    prebook["workspace"] = args.workspace
    prebook["load_dropped"] = dropped_at_load
    prebook["fields"] = fields
    prebook["n_tickers"] = args.n_tickers

    out = args.output or str(Path(args.workspace) / "prebooks" / "lasso_prebook.json")
    save_prebook(prebook, out)
    print(json.dumps({
        "ok": prebook.get("ok"),
        "output": out,
        "n_candidates": prebook.get("n_candidates"),
        "n_usable": prebook.get("n_usable"),
        "n_lasso_nonzero": prebook.get("n_lasso_nonzero"),
        "n_selected": prebook.get("n_selected"),
        "selected_factor_ids": prebook.get("selected_factor_ids"),
        "n_dropped": len(prebook.get("dropped", [])) + len(dropped_at_load),
    }, indent=2))


if __name__ == "__main__":
    main()
