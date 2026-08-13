"""Signal->trade construction lab for the PIT walk-forward composites.

Goal (user 2026-08-07): capitalise the LINEAR combiners' high per-underlying
IC as Sharpe, with explicit turnover control. The composite predictions are
the saved PIT artifacts (pred_<method>.parquet per block, stitched 2021-07 ->
2026-07); every transformation here is causal, so the stitched backtest keeps
the walk-forward's honesty.

Construction (per-name directional — the per-underlying IC convention):
  1. causal per-name standardisation of the composite (expanding or rolling)
  2. optional EWMA smoothing of the score (halflife ~ horizon)
  3. position map: continuous clip(z,±2)/2 or banded sign
  4. optional inverse-vol sizing (63d causal per-name vol), gross-normalised
  5. partial adjustment toward target: W_t = W_{t-1} + lam*(T_t - W_{t-1})
     (lam is THE turnover knob; 1.0 = trade to target daily)
  6. optional no-trade band: skip updates smaller than eps*typical weight
  7. causal vol targeting to 10% ann (63d realised, leverage capped 3x)
  8. costs: 5 bps * daily turnover

Honest protocol: the parameter grid is scored on the TUNE half (2021-07 ->
2023-07-21) only; the chosen configs are then evaluated once on the
VALIDATION half (-> 2026-07) and full period. No validation-half information
enters the choice.

Usage:
  QF_CONFIG_FILE=quant.config.nasdaq100_2010_wf.yaml QF_USE_MCP=0 \
    ./venv/bin/python scripts/wf_pit_strategy_lab.py \
      --pred-root <dir with artifacts/<label>/g*/pred_<m>.parquet> \
      --labels union_wf_s0,L4WF_terra_s0 --methods lasso,ic,ridge,rf
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

TUNE_END = "2023-07-21"
COST_BPS = 5.0
VOL_TARGET = 0.10
ANN = 252


def stitch(pred_root, label, method):
    import pandas as pd
    files = sorted(glob.glob(f"{pred_root}/{label}/g*/pred_{method}.parquet"),
                   key=lambda p: int(p.split("/g")[-1].split("/")[0]))
    if not files:
        return None
    pred = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    return pred[~pred.index.duplicated()].astype(float)


def run_construction(pred, close, r1, *, z_mode, ewma_hl, pos_map, inv_vol,
                     lam, band):
    """Causal per-name construction -> daily net returns + diagnostics."""
    import numpy as np

    ok = close.loc[pred.index].notna() & r1.loc[pred.index].notna()
    p = pred.reindex(columns=close.columns).where(ok)

    if z_mode == "expanding":
        mu, sd = p.expanding(60).mean(), p.expanding(60).std()
    else:  # rolling 252
        mu, sd = p.rolling(252, min_periods=60).mean(), p.rolling(252, min_periods=60).std()
    z = (p - mu) / sd.replace(0.0, np.nan)
    if ewma_hl:
        z = z.ewm(halflife=ewma_hl, min_periods=1).mean()

    if pos_map == "linear":
        u = z.clip(-2, 2) / 2.0
    else:  # band: sign outside |z|>0.5
        u = np.sign(z.where(z.abs() > 0.5, 0.0))
    u = u.fillna(0.0)

    if inv_vol:
        vol = r1.loc[u.index].rolling(63, min_periods=20).std()
        iv = (1.0 / vol.replace(0.0, np.nan)).fillna(0.0)
        u = u * iv
    gross = u.abs().sum(axis=1).replace(0.0, np.nan)
    tgt = u.div(gross, axis=0).fillna(0.0)          # gross 1 target book

    # partial adjustment + optional no-trade band (iterative, causal)
    T = tgt.to_numpy()
    W = np.zeros_like(T)
    w = np.zeros(T.shape[1])
    typical = 1.0 / max(1, int((np.abs(T) > 0).sum(axis=1).mean() or 1))
    eps = band * typical
    for t in range(T.shape[0]):
        step = lam * (T[t] - w)
        if eps > 0:
            step[np.abs(step) < eps] = 0.0
        w = w + step
        W[t] = w
    import pandas as pd
    W = pd.DataFrame(W, index=tgt.index, columns=tgt.columns)

    pnl = (W * r1.loc[W.index]).sum(axis=1)
    # causal vol targeting on the pre-scaled book
    rv = pnl.rolling(63, min_periods=20).std() * np.sqrt(ANN)
    scale = (VOL_TARGET / rv).clip(upper=3.0).shift(1).fillna(1.0)
    Ws = W.mul(scale, axis=0)
    pnl = (Ws * r1.loc[Ws.index]).sum(axis=1)
    to = Ws.diff().abs().sum(axis=1)
    to.iloc[0] = Ws.iloc[0].abs().sum()
    net = (pnl - COST_BPS / 1e4 * to).iloc[:-1]
    return net, to.iloc[:-1], Ws


def stats(net, to=None):
    import numpy as np
    if len(net) < 40 or net.std() == 0:
        return None
    sh = float(net.mean() / net.std() * np.sqrt(ANN))
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    out = {"sharpe": round(sh, 2), "ann_ret": round(float(net.mean() * ANN * 100), 1),
           "ann_vol": round(float(net.std() * np.sqrt(ANN) * 100), 1),
           "maxdd": round(dd * 100, 1)}
    if to is not None:
        out["turnover"] = round(float(to.mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--labels", default="union_wf_s0,L4WF_terra_s0")
    ap.add_argument("--methods", default="lasso,ic,ridge,rf")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    r1 = close.shift(-1) / close - 1.0
    tune_end = pd.Timestamp(TUNE_END)

    GRID = list(itertools.product(
        ["expanding", "rolling"],      # z_mode
        [0, 3, 6, 12],                 # ewma_hl
        ["linear", "band"],            # pos_map
        [True, False],                 # inv_vol
        [1.0, 0.33, 0.15],             # lam
        [0.0, 0.25],                   # no-trade band (x typical weight)
    ))
    results = []
    for label in args.labels.split(","):
        for method in args.methods.split(","):
            pred = stitch(args.pred_root, label, method)
            if pred is None:
                continue
            rows = []
            for z_mode, hl, pm, iv, lam, band in GRID:
                net, to, _ = run_construction(
                    pred, close, r1, z_mode=z_mode, ewma_hl=hl, pos_map=pm,
                    inv_vol=iv, lam=lam, band=band)
                s_tune = stats(net[net.index <= tune_end],
                               to[to.index <= tune_end])
                if not s_tune:
                    continue
                rows.append({"label": label, "method": method, "z": z_mode,
                             "hl": hl, "map": pm, "invvol": iv, "lam": lam,
                             "band": band, "tune": s_tune,
                             "_net": net, "_to": to})
            rows.sort(key=lambda r: -r["tune"]["sharpe"])
            for r in rows[:args.top]:
                net, to = r.pop("_net"), r.pop("_to")
                r["valid"] = stats(net[net.index > tune_end],
                                   to[to.index > tune_end])
                r["full"] = stats(net, to)
                results.append(r)
                print(f"[{label}/{method}] z={r['z']} hl={r['hl']} map={r['map']} "
                      f"invvol={r['invvol']} lam={r['lam']} band={r['band']}\n"
                      f"   tune  {r['tune']}\n   valid {r['valid']}\n"
                      f"   full  {r['full']}")
            for r in rows[args.top:]:
                r.pop("_net", None), r.pop("_to", None)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
