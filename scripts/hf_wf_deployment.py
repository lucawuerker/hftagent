"""Walk-forward deployment of a factor book on single-name HF panels.

Answers "how would the book have traded, refit as time passes?" with the SAME
protocol as the evolution run's prequential phase: the panel's last
``--n-blocks`` blocks of ``--block-bars`` bars are traded in order, and before
each block every combined model is refit on ALL data strictly before the block
(expanding window — nothing after the block start ever touches the fit or the
feature standardisation stats).  Per block it records the combined signal's
OOS IC (at ``--horizon``, honouring ``QF_EXECUTION_LAG_BARS``); the per-block
P&L series are chained into one walk-forward equity curve per model.

Positions are the comparison harness's construction, verbatim: per-underlying
expanding z-score -> threshold(+-1) -> staggered tranche book
(rolling ``--holding-period`` mean) marked on the 1-bar forward return (gross,
no costs — same convention as run_model_comparison's bruteforce track).

The factor book is taken from a prerun's factor DB but signals are computed on
WHATEVER tickers are requested — so a book mined on GLD can be deployed
out-of-universe on SPY or CORN unchanged.

Usage (from repo root):
  QF_EXECUTION_LAG_BARS=3 ./venv/bin/python scripts/hf_wf_deployment.py \
      --prerun L4WF_gld_s0 --config-scope lobster_equity_gld_hf \
      --tickers GLD,SPY,CORN --block-bars 42000 --n-blocks 10 \
      --horizon 60 --holding-period 60 \
      --models linear_regression,random_forest,xgboost,lightgbm \
      --out data/comparisons/gld_hf_wf_deployment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hf_wf_deployment")

BARS_PER_YEAR = 2340 * 252  # 10s bars, regular session


def _book_factor_ids(scope_dir: Path) -> list[str]:
    db = json.loads((scope_dir / "factors" / "factor_db.json").read_text())
    factors = db.get("factors", db)
    if isinstance(factors, dict):
        factors = list(factors.values())
    ids = [f["id"] for f in factors
           if str(f.get("source", "")).lower().endswith("researcher")]
    return ids or [f["id"] for f in factors]


def _load_panel_for(ticker: str, fields: list[str]):
    # The data-layer loader (NOT the raw CSV loader): synthesizes vwap/returns
    # exactly like the evolution run did.  QF_DATA_TICKERS overrides the
    # config's ticker list per call.
    import os

    from quant_fund_agent.data.panel import load_panel
    os.environ.setdefault("QF_CONFIG_FILE", "quant.config.gld_hf.yaml")
    os.environ["QF_DATA_TICKERS"] = ticker
    try:
        return load_panel("ticker_data", fields=fields)
    finally:
        os.environ.pop("QF_DATA_TICKERS", None)


def _signals(factor_ids: list[str], panel) -> dict[str, pd.DataFrame]:
    from quant_fund_agent.factors import discover_factors, instantiate_factor
    discover_factors()  # import every factor module so the registry is populated
    close = panel["close"]
    out = {}
    for fid in factor_ids:
        try:
            sig = (instantiate_factor(fid).calc(panel)
                   .reindex(index=close.index, columns=close.columns)
                   .replace([np.inf, -np.inf], np.nan))
            if sig.notna().to_numpy().mean() < 0.05:
                log.warning("  %s: <5%% coverage on this ticker — skipped", fid)
                continue
            out[fid] = sig
        except Exception as e:  # noqa: BLE001 — a factor that can't run here is skipped
            log.warning("  %s failed on this ticker (%s) — skipped", fid, e)
    return out


def run_ticker(ticker: str, factor_ids: list[str], args, rng) -> pd.DataFrame:
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.backtesting.positions import (directional_positions,
                                                        zscore_over_time)
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.modeling.catalog import build_estimator

    fields = ["close", "open", "high", "low", "volume", "vwap", "returns",
              "trade", "orderFlow", "hidden", "auction", "spread", "effSpread",
              "lobImb", "effLobImb", "trdLiq", "ofLiq", "depth",
              "nbEvents", "nbHidden", "nbTrades"] + [
        f"{side}{kind}{lvl}" for side in ("ask", "bid")
        for kind in ("Price", "Depth") for lvl in range(1, 6)]
    t0 = time.time()
    panel = _load_panel_for(ticker, fields)
    close = panel["close"]
    idx = close.index
    log.info("[%s] panel %d bars (%.0fs)", ticker, len(idx), time.time() - t0)

    sigs = _signals(factor_ids, panel)
    log.info("[%s] %d/%d factors computable", ticker, len(sigs), len(factor_ids))
    fids = sorted(sigs)

    y = forward_returns(close, horizon=args.horizon).to_numpy(dtype=float).ravel()
    n = len(idx)
    total = args.n_blocks * args.block_bars
    if total >= n - 50_000:
        raise SystemExit(f"[{ticker}] panel too short for {args.n_blocks}x{args.block_bars} blocks")
    wf_start = n - total
    blocks = [(wf_start + k * args.block_bars, wf_start + (k + 1) * args.block_bars)
              for k in range(args.n_blocks)]

    rows = []
    chained: dict[str, list[pd.Series]] = {m: [] for m in args.models}
    for k, (b0, b1) in enumerate(blocks):
        fit_end = b0  # expanding window: everything strictly before the block
        is_idx = idx[:fit_end]
        # per-underlying standardisation on pre-block stats only (no leakage)
        feats = {fid: per_underlying_zscore(sigs[fid], is_idx) for fid in fids}
        X = np.column_stack([feats[fid].to_numpy(dtype=float).ravel() for fid in fids])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        train = np.flatnonzero(np.isfinite(y[:fit_end]))
        # forward-return labels within `horizon+lag` of the block start peek into
        # the block — drop them from the fit
        import os
        lag = int(os.environ.get("QF_EXECUTION_LAG_BARS", "0") or 0)
        train = train[train < fit_end - (args.horizon + lag)]
        if len(train) > args.max_train_rows:
            train = np.sort(rng.choice(train, size=args.max_train_rows, replace=False))

        block_dates = f"{idx[b0].date()}->{idx[b1-1].date()}"
        for model in args.models:
            t1 = time.time()
            est = build_estimator(model, None)
            try:
                est.fit(X[train], y[train])
                pred = est.predict(X)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] block %d %s failed (%s)", ticker, k, model, e)
                continue
            sig = pd.DataFrame(pred.reshape(len(idx), -1), index=idx, columns=close.columns)
            # per-block OOS IC of the combined signal at the forecast horizon
            b_sig = sig.iloc[b0:b1, 0].to_numpy()
            b_y = y[b0:b1]
            ok = np.isfinite(b_sig) & np.isfinite(b_y)
            ic = float(np.corrcoef(b_sig[ok], b_y[ok])[0, 1]) if ok.sum() > 100 else np.nan
            # positions: causal expanding z-score over the signal's own history,
            # threshold book, tranche-averaged, marked on the 1-bar return
            z = zscore_over_time(sig.iloc[:b1], "expanding", 500)
            pos = directional_positions(z, mode="threshold", threshold=1.0)
            book = pos.rolling(args.holding_period, min_periods=1).mean()
            pnl = (book * forward_returns(close.iloc[:b1], horizon=1)).iloc[b0:b1, 0]
            chained[model].append(pnl)
            blk_sharpe = (float(pnl.mean() / pnl.std() * np.sqrt(BARS_PER_YEAR))
                          if pnl.std() and pnl.std() > 0 else np.nan)
            rows.append(dict(ticker=ticker, block=k, dates=block_dates, model=model,
                             oos_ic=ic, block_sharpe=blk_sharpe,
                             n_train=len(train), fit_s=round(time.time() - t1, 1)))
            log.info("[%s] block %d/%d %s: IC %+0.4f sharpe %+.2f (%.0fs)",
                     ticker, k + 1, args.n_blocks, model, ic if np.isfinite(ic) else float("nan"),
                     blk_sharpe if np.isfinite(blk_sharpe) else float("nan"), time.time() - t1)

    per_block = pd.DataFrame(rows)
    # chained walk-forward record per model
    summary = []
    for model in args.models:
        if not chained[model]:
            continue
        pnl = pd.concat(chained[model])
        ics = per_block.query("model == @model")["oos_ic"]
        summary.append(dict(
            ticker=ticker, model=model,
            wf_sharpe=float(pnl.mean() / pnl.std() * np.sqrt(BARS_PER_YEAR)),
            wf_ann_return=float(pnl.mean() * BARS_PER_YEAR),
            wf_max_drawdown=float(((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min()),
            mean_oos_ic=float(ics.mean()), median_oos_ic=float(ics.median()),
            pos_blocks=f"{int((ics > 0).sum())}/{len(ics)}",
        ))
    return per_block, pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prerun", default="L4WF_gld_s0")
    p.add_argument("--config-scope", default="lobster_equity_gld_hf")
    p.add_argument("--tickers", default="GLD,SPY,CORN")
    p.add_argument("--block-bars", type=int, default=42000)
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--holding-period", type=int, default=60)
    p.add_argument("--models", default="linear_regression,random_forest,xgboost,lightgbm")
    p.add_argument("--max-train-rows", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/comparisons/gld_hf_wf_deployment")
    args = p.parse_args()
    args.models = [m for m in args.models.split(",") if m]

    scope_dir = Path("data/workspaces") / args.config_scope / "preruns" / args.prerun
    factor_ids = _book_factor_ids(scope_dir)
    log.info("book: %d factors from %s", len(factor_ids), scope_dir)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_blocks, all_summaries = [], []
    for ticker in args.tickers.split(","):
        per_block, summary = run_ticker(ticker.strip(), factor_ids, args, rng)
        all_blocks.append(per_block)
        all_summaries.append(summary)
        # checkpoint after every ticker so a crash keeps finished tickers
        pd.concat(all_blocks).to_csv(out / "wf_blocks.csv", index=False)
        pd.concat(all_summaries).to_csv(out / "wf_summary.csv", index=False)

    summary = pd.concat(all_summaries)
    print("\n" + "=" * 90)
    print(summary.to_string(index=False))
    print("=" * 90)
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
