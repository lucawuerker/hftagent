#!/usr/bin/env python
"""Construction lab: re-trade a walk-forward prediction track five ways.

Uses the prequential-deployment ``stitched_prediction.csv`` (honest walk-forward
combined-model predictions, fit only on then-visible data) and compares position
constructions on the SAME signal — isolating the signal→portfolio step:

  baseline_top10       exact current protocol (top10/bottom10 equal-weight,
                       dollar-neutral) — sanity-checked against the report
  full_book            signal-proportional whole-cross-section book
                       (per-day z of pred, clipped ±2, demeaned, gross 1.0)
  top10_diversified    top-10 per side with a max-2-per-GICS-sector cap
                       (greedy down the ranking), equal weight, dollar-neutral
  top10_hedged         long top-10 (equal weight, 50%) vs SHORT equal-weight
                       universe proxy (50%) — the "one-instrument QQQ hedge"
  product_shape        diversified top-10 long, inverse-vol weights, vs the
                       50% universe-proxy short (the sellable K-name shape)

All books use the same 6-bar tranche hold, ±50% forward-return clip and
5 bp × turnover costs as the existing protocol.  Analysis-only: no LLM, no
refit — pure pandas over stored predictions + the close/sector panel.

Usage:
  QF_CONFIG_FILE=quant.config.nasdaq100_2010_forward.yaml QF_USE_MCP=0 \
  ./venv/bin/python scripts/walkforward_construction_lab.py \
      --run L4_terra_s0 [--n-side 10] [--sector-cap 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QF_USE_MCP", "0")

import numpy as np
import pandas as pd

H = 6
COST = 5e-4
ANN = 252


def ann_stats(net: pd.Series) -> dict:
    r = net.dropna()
    if len(r) < 20 or r.std(ddof=0) == 0:
        return {"sharpe": float("nan"), "ann_ret": float("nan"),
                "ann_vol": float("nan"), "max_dd": float("nan"), "n": len(r)}
    eq = (1 + r).cumprod()
    return {"sharpe": float(math.sqrt(ANN) * r.mean() / r.std(ddof=0)),
            "ann_ret": float(r.mean() * ANN),
            "ann_vol": float(r.std(ddof=0) * math.sqrt(ANN)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "n": int(len(r))}


def trade(target: pd.DataFrame, fwd1: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Tranche-hold a daily target book; net returns + daily turnover."""
    book = target.rolling(H, min_periods=1).mean()
    pnl = (book * fwd1).sum(axis=1)
    turnover = (book - book.shift(1)).abs().sum(axis=1)
    net = pnl - COST * turnover
    net.iloc[:H] = np.nan
    return net.iloc[:-1], turnover.iloc[:-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="L4_terra_s0")
    ap.add_argument("--workspace", default="fmp_archive_equity_nasdaq100pit")
    ap.add_argument("--n-side", type=int, default=10)
    ap.add_argument("--sector-cap", type=int, default=2)
    args = ap.parse_args()
    K = args.n_side

    run_dir = ROOT / "data/workspaces" / args.workspace / "preruns" / args.run
    dep = run_dir / "prequential_deployment"
    pred = pd.read_csv(dep / "stitched_prediction.csv", index_col=0,
                       parse_dates=True)
    report = json.loads((dep / "report.json").read_text())

    from quant_fund_agent.mcp import research_service as svc
    panel = svc._load_panel_cached("ticker_data", ["close", "sector"],
                                   n_tickers=None)
    close = panel["close"]
    pred = pred.reindex(index=close.index, columns=close.columns)
    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)

    # static sector per ticker (last observed label)
    sec_frame = panel.get("sector")
    sector = {}
    if sec_frame is not None:
        for t in close.columns:
            col = sec_frame[t].dropna() if t in sec_frame else pd.Series(dtype=object)
            sector[t] = str(col.iloc[-1]) if len(col) else "UNKNOWN"
    sec_arr = np.array([sector.get(t, "UNKNOWN") for t in close.columns])

    # 60d inverse-vol weights (causal)
    vol = close.pct_change().rolling(60, min_periods=20).std()

    P = pred.to_numpy(dtype=float)
    n_dates, n_names = P.shape
    order = np.argsort(-P, axis=1)          # descending prediction per day

    def greedy_side(row_order, valid_row, cap, want, side):
        """Greedy pick down (side=+1) or up (side=-1) the ranking with a
        per-sector cap; returns column indices."""
        picks, counts = [], {}
        it = row_order if side > 0 else row_order[::-1]
        for j in it:
            if not valid_row[j]:
                continue
            s = sec_arr[j]
            if counts.get(s, 0) >= cap:
                continue
            picks.append(j)
            counts[s] = counts.get(s, 0) + 1
            if len(picks) == want:
                break
        return picks

    valid = ~np.isnan(P)
    top_m = np.zeros_like(P, dtype=bool)
    bot_m = np.zeros_like(P, dtype=bool)
    div_top = np.zeros_like(P, dtype=bool)
    div_bot = np.zeros_like(P, dtype=bool)
    for i in range(n_dates):
        if valid[i].sum() < 4 * K:
            continue
        row = order[i]
        vr = valid[i]
        picks = [j for j in row if vr[j]]
        top_m[i, picks[:K]] = True
        bot_m[i, picks[-K:]] = True
        div_top[i, greedy_side(row, vr, args.sector_cap, K, +1)] = True
        div_bot[i, greedy_side(row, vr, args.sector_cap, K, -1)] = True

    idx, cols = close.index, close.columns
    top_m = pd.DataFrame(top_m, idx, cols)
    bot_m = pd.DataFrame(bot_m, idx, cols)
    div_top = pd.DataFrame(div_top, idx, cols)
    div_bot = pd.DataFrame(div_bot, idx, cols)
    any_day = top_m.any(axis=1)

    books: dict[str, pd.DataFrame] = {}

    # A) baseline: current protocol
    books["baseline_top10"] = (top_m.astype(float) - bot_m.astype(float)) / (2 * K)

    # B) full-book proportional: per-day cross-sectional z, clip ±2, demean,
    #    gross 1.0
    mu = pred.mean(axis=1)
    sd = pred.std(axis=1).replace(0.0, np.nan)
    z = pred.sub(mu, axis=0).div(sd, axis=0).clip(-2, 2)
    z = z.sub(z.mean(axis=1), axis=0)
    gross = z.abs().sum(axis=1).replace(0.0, np.nan)
    books["full_book"] = z.div(gross, axis=0).fillna(0.0)

    # C) diversified top-K, dollar-neutral
    books["top10_diversified"] = (div_top.astype(float)
                                  - div_bot.astype(float)) / (2 * K)

    # D) top-K long vs universe-proxy short (one-instrument hedge):
    #    the short leg spreads -50% over every name with data that day —
    #    return-equivalent to shorting an equal-weight universe ETF.
    n_valid = valid.sum(axis=1)
    proxy_w = pd.DataFrame(
        np.where(valid, 1.0, 0.0) / np.maximum(n_valid, 1)[:, None], idx, cols)
    proxy_w[~any_day] = 0.0
    books["top10_hedged"] = top_m.astype(float) / (2 * K) - 0.5 * proxy_w

    # E) product shape: diversified long book, inverse-vol sizing, hedged
    iv = (1.0 / vol).where(div_top, 0.0).replace([np.inf, -np.inf], np.nan) \
        .fillna(0.0)
    iv_sum = iv.sum(axis=1).replace(0.0, np.nan)
    long_w = iv.div(iv_sum, axis=0).fillna(0.0) * 0.5
    books["product_shape"] = long_w - 0.5 * proxy_w

    # ── NO-LEVERAGE variants: gross ≤ 1.0, ≤ 20 positions, no hedge ──────────
    # F) per-underlying threshold book, CAPPED at 20 names (strongest |z|),
    #    1/20 each → gross ≤ 1.0 (the old book leveraged up whenever more than
    #    20 names crossed the threshold)
    from quant_fund_agent.backtesting.positions import (
        directional_positions,
        zscore_over_time,
    )
    zpu = zscore_over_time(pred, "expanding", 500)
    raw = directional_positions(zpu, "threshold", 1.0)
    absz = zpu.abs().where(raw != 0.0, 0.0).fillna(0.0)
    keep_rank = absz.rank(axis=1, ascending=False, method="first")
    capped = raw.where(keep_rank <= 20, 0.0)
    books["pu_capped20"] = capped * (1.0 / 20)

    # G) long-only product WITHOUT hedge: diversified top-K, inverse-vol,
    #    fully invested (weights sum to 1.0) — max K positions, no leverage
    books["product_longonly"] = iv.div(iv_sum, axis=0).fillna(0.0)

    # H) long-only equal-weight diversified top-K, fully invested
    books["top10_longonly_ew"] = div_top.astype(float) / K

    # I) benchmark: equal-weight universe buy-and-hold (context for the
    #    long-only rows — how much is just market beta?)
    n_valid = valid.sum(axis=1)
    bench_w = pd.DataFrame(
        np.where(valid, 1.0, 0.0) / np.maximum(n_valid, 1)[:, None], idx, cols)
    books["benchmark_universe"] = bench_w

    # periods from the deployment report
    seg = report["segments"]
    blocks_start = pd.Timestamp(min(s["eval_start"] for s in seg
                                    if s["segment"].startswith("block")))
    test_start = pd.Timestamp([s for s in seg
                               if s["segment"] == "test_tail"][0]["eval_start"])
    fwd_start = pd.Timestamp([s for s in seg
                              if s["segment"] == "forward_reserve"][0]["eval_start"])
    periods = {
        "blocks_2016_2021": lambda d: (d >= blocks_start) & (d < test_start),
        "test_2021_2024": lambda d: (d >= test_start) & (d < fwd_start),
        "forward_2024_2026": lambda d: d >= fwd_start,
        "full_track": lambda d: d >= blocks_start,
    }

    rows = []
    for name, target in books.items():
        net, turn = trade(target, fwd1)
        net = net[any_day.reindex(net.index).fillna(False)]
        for pname, mask_fn in periods.items():
            sub = net[mask_fn(net.index)]
            st = ann_stats(sub)
            rows.append({"construction": name, "period": pname, **st,
                         "turnover": float(
                             turn.reindex(sub.index).mean())})
    out = pd.DataFrame(rows)
    out_path = dep / "construction_lab.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    for pname in periods:
        print(f"\n== {pname} ==")
        sub = out[out.period == pname][
            ["construction", "sharpe", "ann_ret", "ann_vol", "max_dd",
             "turnover", "n"]]
        print(sub.to_string(index=False,
                            float_format=lambda x: f"{x:+.3f}"))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
