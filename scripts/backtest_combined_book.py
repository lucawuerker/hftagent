"""Combined-book strategy backtest with a model race + one-shot forward OOS.

Protocol (agreed 2026-07-29):
- Factor set: a run's final published book (default: L2's 18-factor front) or a
  named factor list / persona selection (used by simulate_user_strategies.py).
- Panel: the FORWARD config (2010 -> 2026-07). Fit/selection may only look at
  rows <= 2024-07-27 (the research panel); 2024-07-28 -> 2026-07-27 is the
  single forward-validation frame, touched exactly once per final strategy.
- Model race: ridge / elastic_net / random_forest / lightgbm, each fit on IS
  (panel rows before VAL_START), selected on VAL (VAL_START -> 2024-07-27) by
  combined IC; N_trials = number of models raced feeds the deflated Sharpe.
  CSCV PBO is computed across the racers' VAL strategy-return series.
- Strategy ("old sp100 style", cross-sectional): each bar rank the combined
  prediction across names with data; long the top 10, short the bottom 10,
  equal-weight dollar-neutral (gross 1.0), tranche-held over the 6-bar horizon
  (book = mean of the last 6 target books), marked to market on 1-bar forward
  returns, net of cost_rate * turnover (5 bp default).
- The winning model is REFIT on the full in-panel window (2010 -> 2024-07-27,
  the Architect's refit-on-full-IS convention) and evaluated ONCE on the
  forward frame.

Outputs <out>/report.json + returns CSVs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backtest_combined_book")

MODELS = ("ridge", "elastic_net", "random_forest", "lightgbm")
H = 6
COST_RATE = 5e-4
N_SIDE = 10          # 10 long / 10 short (agreed)
PANEL_END = "2024-07-28"      # first forward bar (exclusive fit bound)
VAL_YEARS = 2                 # last 2 in-panel years = model-selection VAL


def cross_sectional_returns(pred, close, cost_rate=COST_RATE, n_side=N_SIDE, h=H):
    """Daily net returns of the tranche-held long/short top-N book."""
    import numpy as np
    import pandas as pd

    # 1-bar forward return, clipped to +/-50% — early archive years contain a
    # handful of unadjusted-split glitches that would otherwise produce single
    # fictitious -90% book days (data hygiene, not signal shaping).
    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)
    ranks = pred.rank(axis=1, method="first")
    n_valid = pred.notna().sum(axis=1)
    long_m = ranks.ge(n_valid - n_side + 1, axis=0) & pred.notna()
    short_m = ranks.le(n_side, axis=0) & pred.notna()
    target = (long_m.astype(float) - short_m.astype(float))
    per_name = 1.0 / (2 * n_side)                  # gross 1.0 dollar-neutral
    target = target * per_name
    book = target.rolling(h, min_periods=1).mean()  # tranche book
    pnl = (book * fwd1).sum(axis=1)
    turnover = (book - book.shift(1)).abs().sum(axis=1)
    net = pnl - cost_rate * turnover
    # warmup: the first h bars hold a partial tranche book
    net.iloc[:h] = float("nan")
    pnl.iloc[:h] = float("nan")
    return net.iloc[:-1], turnover.iloc[:-1], pnl.iloc[:-1]


def per_underlying_returns(pred, close, cost_rate=COST_RATE, h=H,
                           max_positions=20):
    """Daily net returns of the deployment-faithful PER-UNDERLYING book.

    Uses the SHARED primitives (backtesting.positions): causal expanding
    z-score of each name's prediction over its own history -> threshold band
    (+1 above +1z, -1 below -1z, flat between) -> each active name sized
    1/max_positions (directional, net market exposure possible). Same 6-bar
    tranche hold and cost treatment as the cross-sectional book.
    """
    from quant_fund_agent.backtesting.positions import (
        directional_positions,
        zscore_over_time,
    )

    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)
    z = zscore_over_time(pred, "expanding", 500)
    raw = directional_positions(z, "threshold", 1.0)
    target = raw * (1.0 / max_positions)
    book = target.rolling(h, min_periods=1).mean()
    pnl = (book * fwd1).sum(axis=1)
    turnover = (book - book.shift(1)).abs().sum(axis=1)
    net = pnl - cost_rate * turnover
    net.iloc[:h] = float("nan")
    pnl.iloc[:h] = float("nan")
    return net.iloc[:-1], turnover.iloc[:-1], pnl.iloc[:-1]


CONSTRUCTIONS = {"cross_sectional": cross_sectional_returns,
                 "per_underlying": per_underlying_returns}


def ann_stats(returns):
    import numpy as np
    r = returns.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": None, "ann_ret": None, "ann_vol": None, "max_dd": None}
    sharpe = float(np.sqrt(252) * r.mean() / r.std())
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": sharpe, "ann_ret": float(r.mean() * 252),
            "ann_vol": float(r.std() * np.sqrt(252)), "max_dd": dd,
            "n_bars": int(len(r))}


def run_book_backtest(factor_ids, codes, out_dir, label,
                      extra_trials: int = 0,
                      construction: str = "cross_sectional") -> dict:
    """The full protocol for one factor set; returns the report dict."""
    import numpy as np
    import pandas as pd
    import types

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.research_eval.harness import _combined_prediction, _pooled_ic
    from quant_fund_agent.research_eval.deflation import (
        deflated_sharpe_ratio,
        pbo_cscv,
    )

    fields = sorted(usable_fields())
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    forward_start = pd.Timestamp(PANEL_END)
    val_start = forward_start - pd.DateOffset(years=VAL_YEARS)
    is_mask = np.asarray(idx < val_start)
    val_mask = np.asarray((idx >= val_start) & (idx < forward_start))
    fit_mask = np.asarray(idx < forward_start)     # refit window for the winner
    fwd_mask = np.asarray(idx >= forward_start)
    log.info("[%s] IS<%s VAL<%s FORWARD %s->%s", label, val_start.date(),
             forward_start.date(), idx[fwd_mask][0].date(), idx[-1].date())

    discover_factors()
    signals = []
    used_ids = []
    for fid in factor_ids:
        try:
            cls = get_factor_class(fid) or compile_factor(codes[fid], fid)
            signals.append(compute_signal(cls, panel))
            used_ids.append(fid)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] factor %s failed to compute (%s) — excluded", label, fid, e)
    log.info("[%s] %d/%d factors computable", label, len(used_ids), len(factor_ids))

    returns_fn = CONSTRUCTIONS[construction]
    cfg = types.SimpleNamespace(target_horizon=H, fit_standardize="per_underlying")
    race = {}
    val_return_series = {}
    for model in MODELS:
        pred = _combined_prediction(signals, close, is_mask, cfg, model)
        if pred is None:
            race[model] = {"val_ic": None}
            continue
        val_ic = _pooled_ic(pred, close, H, val_mask, is_mask, is_mask | val_mask)[0]
        vr, _, _ = returns_fn(pred, close)
        vr = vr[val_mask[:-1]]
        race[model] = {"val_ic": val_ic, **{f"val_{k}": v for k, v in ann_stats(vr).items()}}
        val_return_series[model] = vr
        log.info("[%s] %s: VAL IC=%.4f sharpe=%s", label, model,
                 val_ic if val_ic is not None else float("nan"),
                 race[model].get("val_sharpe"))

    n_trials = len([m for m in race if race[m]["val_ic"] is not None]) + extra_trials
    winner = max((m for m in race if race[m]["val_ic"] is not None),
                 key=lambda m: race[m]["val_ic"])
    log.info("[%s] winner on VAL IC: %s (n_trials=%d)", label, winner, n_trials)

    pbo = None
    try:
        M = pd.DataFrame(val_return_series).dropna()
        if len(M) > 100 and M.shape[1] >= 2:
            pbo = pbo_cscv(M.to_numpy(), n_groups=10).get("pbo")
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] PBO failed: %s", label, e)

    # ── refit winner on the FULL in-panel window; ONE forward pass ──
    pred_full = _combined_prediction(signals, close, fit_mask, cfg, winner)
    fwd_ic = _pooled_ic(pred_full, close, H, fwd_mask, fit_mask,
                        fit_mask | fwd_mask)[0]
    net, turnover, gross = returns_fn(pred_full, close)
    fwd_ret = net[fwd_mask[:-1]]
    in_ret = net[fit_mask[:-1]]
    fwd_gross = gross[fwd_mask[:-1]]
    fwd = ann_stats(fwd_ret)
    dsr = None
    try:
        r = fwd_ret.dropna()
        if fwd["sharpe"] is not None and len(r) > 30:
            from scipy import stats as _sps
            per_bar_sr = float(r.mean() / r.std())
            # variance of the racers' per-bar VAL Sharpes = the trial spread
            val_srs = [s.mean() / s.std() for s in val_return_series.values()
                       if s.std() > 0]
            sr_var = float(np.var(val_srs)) if len(val_srs) > 1 else None
            dsr = deflated_sharpe_ratio(
                per_bar_sr, n_obs=len(r), n_trials=max(2, n_trials),
                sr_variance=(sr_var if sr_var and sr_var > 0 else 1e-6),
                skew=float(_sps.skew(r)), kurtosis=float(_sps.kurtosis(r, fisher=False)))
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] DSR failed: %s", label, e)

    out_dir.mkdir(parents=True, exist_ok=True)
    net.to_csv(out_dir / f"{label}_net_returns.csv")
    report = {
        "label": label,
        "construction": construction,
        "factor_ids": used_ids,
        "n_factors": len(used_ids),
        "models_raced": list(MODELS),
        "race": race,
        "winner": winner,
        "n_trials_strategy": n_trials,
        "pbo_cscv_val": pbo,
        "forward_gross": ann_stats(fwd_gross),
        "forward": {"ic": fwd_ic, **fwd,
                    "window": [str(idx[fwd_mask][0].date()), str(idx[-1].date())],
                    "deflated_sharpe_prob": dsr},
        "in_panel_net": ann_stats(in_ret),
        "mean_daily_turnover": float(turnover.mean()),
    }
    (out_dir / f"{label}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="quant.config.nasdaq100_2010_forward.yaml")
    p.add_argument("--prerun", default="L2_opus5_s0",
                   help="Run whose final archive book is backtested.")
    p.add_argument("--construction", choices=list(CONSTRUCTIONS), default="cross_sectional")
    p.add_argument("--extra-trials", type=int, default=0,
                   help="Additional configurations already evaluated forward (deflation).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.environ["QF_CONFIG_FILE"] = args.config
    os.environ.setdefault("QF_USE_MCP", "0")

    st = json.loads((REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
                     / args.prerun / "evolution/state.json").read_text())
    fids = [a["genome"]["programs"][0]["factor_id"] for a in st["archive"]]
    codes = {a["genome"]["programs"][0]["factor_id"]: a["genome"]["programs"][0]["code"]
             for a in st["archive"]}
    out = Path(args.out or (REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit"
                            / "preruns" / args.prerun / "book_backtest"))
    label = "combined_book" if args.construction == "cross_sectional" else f"combined_book_{args.construction}"
    report = run_book_backtest(fids, codes, out, label,
                               extra_trials=args.extra_trials,
                               construction=args.construction)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
