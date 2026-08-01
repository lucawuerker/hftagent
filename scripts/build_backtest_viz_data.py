#!/usr/bin/env python
"""Assemble backtest_viz_data.json for plot_strategy_backtests.py (any catalog).

Pools the combined-book backtest of a prerun and a catalog's persona sample
strategies into the viz schema the figure scripts read:
{val_start, oos_start, cost_rate_bp, strategies: {name: {returns, metrics,
winner, n_factors, mean_daily_turnover, ann_cost_drag, forward_ic,
forward_gross_sharpe, dsr}}}.

Usage:
  ./venv/bin/python scripts/build_backtest_viz_data.py \
      --catalog terra_l4_v1 --prerun L4_terra_s0 \
      [--construction cross_sectional] [--val-start 2022-07-28] \
      [--oos-start 2024-07-28]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = 252


def ann_stats(net: pd.Series) -> dict:
    net = net.dropna()
    if net.empty:
        return {"sharpe": None, "ann_ret": None, "ann_vol": None,
                "max_dd": None, "n_bars": 0}
    mu, sd = float(net.mean()), float(net.std(ddof=0))
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": (mu / sd * math.sqrt(ANN)) if sd > 0 else None,
            "ann_ret": mu * ANN, "ann_vol": sd * math.sqrt(ANN),
            "max_dd": dd, "n_bars": int(len(net))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--prerun", required=True)
    p.add_argument("--workspace", default="fmp_archive_equity_nasdaq100pit")
    p.add_argument("--construction", default="cross_sectional",
                   choices=["cross_sectional", "per_underlying"])
    p.add_argument("--val-start", default="2022-07-28")
    p.add_argument("--oos-start", default="2024-07-28")
    p.add_argument("--cost-rate-bp", type=int, default=5)
    args = p.parse_args()

    cat_dir = ROOT / "data/books" / f"catalog_{args.catalog}"
    prerun_dir = (ROOT / "data/workspaces" / args.workspace / "preruns"
                  / args.prerun)
    pu = args.construction == "per_underlying"
    strat_dir = cat_dir / ("sample_strategies_per_underlying" if pu
                           else "sample_strategies")
    book_report = prerun_dir / "book_backtest" / (
        "combined_book_per_underlying_report.json" if pu
        else "combined_book_report.json")
    book_returns = prerun_dir / "book_backtest" / (
        "combined_book_per_underlying_net_returns.csv" if pu
        else "combined_book_net_returns.csv")

    def load(returns_csv: Path, report_json: Path) -> dict:
        rep = json.loads(report_json.read_text())
        ser = pd.read_csv(returns_csv, index_col=0,
                          parse_dates=True).iloc[:, 0]
        vs, os_ = pd.Timestamp(args.val_start), pd.Timestamp(args.oos_start)
        metrics = {
            "IS": ann_stats(ser[ser.index < vs]),
            "VAL": ann_stats(ser[(ser.index >= vs) & (ser.index < os_)]),
            "OOS": ann_stats(ser[ser.index >= os_]),
        }
        fwd, fwd_g = rep.get("forward") or {}, rep.get("forward_gross") or {}
        turn = rep.get("mean_daily_turnover") or 0.0
        return {
            "returns": {d.strftime("%Y-%m-%d"):
                        (None if pd.isna(v) else float(v))
                        for d, v in ser.items()},
            "metrics": metrics,
            "winner": rep.get("winner"),
            "n_factors": rep.get("n_factors"),
            "mean_daily_turnover": turn,
            "ann_cost_drag": turn * args.cost_rate_bp / 1e4 * ANN,
            "forward_ic": fwd.get("ic"),
            "forward_gross_sharpe": fwd_g.get("sharpe"),
            "dsr": fwd.get("deflated_sharpe_prob"),
        }

    strategies = {"combined_book": load(book_returns, book_report)}
    for rep_path in sorted(strat_dir.glob("*_report.json")):
        rep = json.loads(rep_path.read_text())
        name = rep.get("label") or rep_path.stem.replace("_report", "")
        for suffix in ("_per_underlying",):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        ret_path = Path(str(rep_path).replace("_report.json",
                                              "_net_returns.csv"))
        strategies[name] = load(ret_path, rep_path)

    out = {"val_start": args.val_start, "oos_start": args.oos_start,
           "cost_rate_bp": args.cost_rate_bp, "strategies": strategies}
    out_path = cat_dir / ("backtest_viz_data_pu.json" if pu
                          else "backtest_viz_data.json")
    out_path.write_text(json.dumps(out, default=str))
    print(f"wrote {out_path} ({len(strategies)} strategies)")


if __name__ == "__main__":
    main()
