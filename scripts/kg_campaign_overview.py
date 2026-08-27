"""KG-campaign overview computations (correlations + missing worker rows).

Stage A (--stage a): one dev/WF assembly of every valid cumulative-book
factor; produces
  overview/factor_corr_runpair.csv   21x21 mean |pairwise factor corr|
  overview/lasso_signal_corr.csv     21x21 corr of per-run combined lasso
                                     signals (fit on dev, predicted on WF)
  overview/lasso_wf_ic.csv           single-fit WF pooled IC per run book
Stage B (--stage b): replicates the kg_ic_worker per-block refit protocol
locally for the rows lagias hasn't produced yet (cum ridge runs 18-20,
run-scope ridge+lasso runs 19-20) -> overview/local_results.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_overview")

import numpy as np
import pandas as pd

from wf_common import SIGNAL_STORE, signal_key

CAMP = REPO / "data/kg_campaign"
OV = CAMP / "overview"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
REF_PREQUENTIAL = WS / "L4WF_terra_s0/evolution/prequential.jsonl"
H = 6
RIDGE_ALPHA = 1e4
DEV_END = pd.Timestamp("2021-07-20")


def load_book():
    cum = json.loads((CAMP / "cumulative_book.json").read_text())
    out = []
    for e in cum:
        p = Path(e["code_path"])
        if not p.exists():
            p = REPO / "quant_fund_agent/factors/researcher" / p.name
        if not p.exists():
            continue
        key = signal_key(e["factor_id"], p.read_text())
        if (SIGNAL_STORE / f"{key}.parquet").exists():
            out.append((e["factor_id"], key, e["run"]))
    return out


def setup_panel():
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", ["close"], n_tickers=None)
    close = panel["close"]
    y_all = forward_returns(close, horizon=H).to_numpy(dtype=np.float64).ravel()
    return close, y_all


def rows_ic(pred, y, tids, n_assets):
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return None
    t, p, q = tids[m], pred[m].astype(np.float64), y[m].astype(np.float64)
    n = np.bincount(t, minlength=n_assets).astype(np.float64)
    sx = np.bincount(t, weights=p, minlength=n_assets)
    sy = np.bincount(t, weights=q, minlength=n_assets)
    sxx = np.bincount(t, weights=p * p, minlength=n_assets)
    syy = np.bincount(t, weights=q * q, minlength=n_assets)
    sxy = np.bincount(t, weights=p * q, minlength=n_assets)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx)
                                          * (n * syy - sy * sy))
    ok = (n >= 3) & np.isfinite(r)
    return float((r[ok] * n[ok]).sum() / n[ok].sum()) if ok.any() else None


def assemble(book, close, bar_mask, stat_mask, dtype=np.float16):
    """float array (rows of bar_mask x len(book)); z-stats over stat_mask."""
    idx, cols = close.index, close.columns
    n_assets = len(cols)
    rows = np.flatnonzero(np.repeat(bar_mask, n_assets))
    X = np.empty((len(rows), len(book)), dtype=dtype)
    for j, (fid, key, _run) in enumerate(book):
        sig = pd.read_parquet(SIGNAL_STORE / f"{key}.parquet").reindex(
            index=idx, columns=cols)
        v = np.array(sig.to_numpy(dtype=np.float32), copy=True)
        v[~np.isfinite(v)] = np.nan
        ref = v[stat_mask]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(ref, axis=0)
            sd = np.nanstd(ref, axis=0, ddof=1)
        sd[sd == 0] = np.nan
        z = ((v - mu) / sd)[bar_mask].ravel()
        X[:, j] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if (j + 1) % 400 == 0:
            log.info("  %d/%d", j + 1, len(book))
    return X, rows


def gram(X, chunk=16384):
    n, N = X.shape
    G = np.zeros((N, N), dtype=np.float64)
    s = np.zeros(N, dtype=np.float64)
    for a in range(0, n, chunk):
        Xc = X[a:a + chunk].astype(np.float64)
        G += Xc.T @ Xc
        s += Xc.sum(axis=0)
    return G, s, n


def corr_from_gram(G, s, n):
    cov = n * G - np.outer(s, s)
    d = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return cov / np.outer(d, d)


def stage_a():
    book = load_book()
    runs = sorted({r for _, _, r in book})
    close, y_all = setup_panel()
    idx, cols = close.index, close.columns
    n_assets = len(cols)
    dev_mask = np.asarray(idx < DEV_END)
    wf_mask = ~dev_mask
    log.info("valid factors: %d, runs: %s", len(book), runs)

    log.info("assembling X_dev ...")
    X_dev, dev_rows = assemble(book, close, dev_mask, dev_mask)
    log.info("factor corr (gram %dx%d over %d rows)", len(book), len(book),
             len(X_dev))
    C = corr_from_gram(*gram(X_dev))
    run_of = np.array([r for _, _, r in book])
    K = len(runs)
    M = np.zeros((K, K))
    absC = np.abs(C)
    for i, a in enumerate(runs):
        ia = np.flatnonzero(run_of == a)
        for j, b in enumerate(runs):
            ib = np.flatnonzero(run_of == b)
            sub = absC[np.ix_(ia, ib)]
            if a == b:
                v = sub[np.triu_indices_from(sub, k=1)]
            else:
                v = sub.ravel()
            M[i, j] = float(np.nanmean(v))
    pd.DataFrame(M, index=runs, columns=runs).to_csv(
        OV / "factor_corr_runpair.csv")
    log.info("factor corr saved")

    # per-run combined lasso: fit on dev (stride 2), predict WF
    from sklearn.linear_model import LassoCV
    from quant_fund_agent.research_eval.harness import _label_available_mask

    lab_ok = _label_available_mask(dev_mask, H)
    fit_bar = np.repeat(dev_mask & lab_ok, n_assets)
    fit_sel = fit_bar[dev_rows] & np.isfinite(y_all[dev_rows])
    fit_i = np.flatnonzero(fit_sel)[::2]
    yf = y_all[dev_rows][fit_sel][::2]
    coefs, inters, nnz = {}, {}, {}
    for r in runs:
        cols_r = np.flatnonzero(run_of == r)
        Xf = X_dev[fit_i][:, cols_r].astype(np.float32)
        t0 = time.time()
        est = LassoCV(cv=3, random_state=42, precompute=False, n_jobs=4)
        est.fit(Xf, yf)
        coefs[r], inters[r] = est.coef_.copy(), float(est.intercept_)
        nnz[r] = int((est.coef_ != 0).sum())
        log.info("run %02d lasso fit: %d F., %d nonzero (%.0fs)",
                 r, len(cols_r), nnz[r], time.time() - t0)
    del X_dev
    log.info("assembling X_wf ...")
    X_wf, wf_rows = assemble(book, close, wf_mask, dev_mask)
    tids_wf = np.tile(np.arange(n_assets), int(wf_mask.sum()))
    y_wf = y_all[wf_rows]
    preds, ics = {}, {}
    for r in runs:
        cols_r = np.flatnonzero(run_of == r)
        pred = X_wf[:, cols_r].astype(np.float64) @ coefs[r] + inters[r]
        preds[r] = pred.astype(np.float32)
        ics[r] = rows_ic(pred, y_wf, tids_wf, n_assets)
        log.info("run %02d WF IC %.4f", r, ics[r] or float("nan"))
    P = np.column_stack([preds[r] for r in runs])
    pc = np.corrcoef(P, rowvar=False)
    pd.DataFrame(pc, index=runs, columns=runs).to_csv(
        OV / "lasso_signal_corr.csv")
    pd.Series(ics).to_csv(OV / "lasso_wf_ic.csv", header=["wf_ic"])
    log.info("stage A done")


def stage_b():
    book = load_book()
    close, y_all = setup_panel()
    idx, cols = close.index, close.columns
    n_assets = len(cols)
    run_of = np.array([r for _, _, r in book])
    from sklearn.linear_model import LassoCV
    from quant_fund_agent.research_eval.harness import (_label_available_mask,
                                                        _pooled_ic)

    blocks = []
    for line in REF_PREQUENTIAL.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((r["generation"], pd.Timestamp(r["start"]),
                           pd.Timestamp(r["end"])))
    blocks.sort()

    # (scope-label, factor column subset, methods)
    jobs = [("cum18", run_of <= 18, ("ridge",)),
            ("cum19", run_of <= 19, ("ridge",)),
            ("cum20", run_of <= 20, ("ridge",)),
            ("run19", run_of == 19, ("ridge", "lasso")),
            ("run20", run_of == 20, ("ridge", "lasso"))]
    out = {(lbl, m): [] for lbl, _, ms in jobs for m in ms}
    for g, start, end in blocks:
        fit_mask = np.asarray(idx < start)
        block_mask = np.asarray((idx >= start) & (idx < end))
        lab_ok = _label_available_mask(fit_mask, H)
        log.info("block g%d: assembling ...", g)
        Xfit, fit_rows = assemble(book, close, fit_mask & lab_ok, fit_mask)
        keep = np.isfinite(y_all[fit_rows])
        Xfit = Xfit[keep]
        yf = y_all[fit_rows][keep]
        Xblk, blk_rows = assemble(book, close, block_mask, fit_mask)
        for lbl, sel, methods in jobs:
            cix = np.flatnonzero(sel)
            for m in methods:
                if m == "ridge":
                    G = np.zeros((len(cix), len(cix)))
                    c = np.zeros(len(cix))
                    for a in range(0, len(Xfit), 16384):
                        Xc = Xfit[a:a + 16384, cix].astype(np.float64)
                        G += Xc.T @ Xc
                        c += Xc.T @ yf[a:a + 16384]
                    G[np.diag_indices_from(G)] += RIDGE_ALPHA
                    w = np.linalg.solve(G, c)
                    pred = Xblk[:, cix].astype(np.float64) @ w
                else:
                    est = LassoCV(cv=3, random_state=42, precompute=False)
                    est.fit(Xfit[::2, cix].astype(np.float64), yf[::2])
                    pred = (Xblk[:, cix].astype(np.float64) @ est.coef_
                            + est.intercept_)
                full = np.full(len(idx) * n_assets, np.nan)
                full[blk_rows] = pred
                pred_df = pd.DataFrame(full.reshape(len(idx), n_assets),
                                       index=idx, columns=cols)
                ic, _ = _pooled_ic(pred_df, close, H, row_mask=block_mask,
                                   available_mask=block_mask)
                out[(lbl, m)].append(ic)
                log.info("  %s/%s g%d IC %.4f", lbl, m, g, ic or float("nan"))
        del Xfit, Xblk
        with (OV / "local_results_partial.json").open("w") as f:
            json.dump({f"{l}:{m}": v for (l, m), v in out.items()}, f)
    rows = []
    for (lbl, m), ics in out.items():
        ok = [i for i in ics if i is not None]
        rows.append({"label": lbl, "method": m,
                     "n_factors": int((run_of <= int(lbl[3:])).sum())
                     if lbl.startswith("cum") else
                     int((run_of == int(lbl[3:])).sum()),
                     "blockmean": float(np.mean(ok)),
                     "blockstd": float(np.std(ok, ddof=1)),
                     "hit": float(np.mean([i > 0 for i in ok])),
                     "ics": ics})
    pd.DataFrame(rows).to_csv(OV / "local_results.csv", index=False)
    log.info("stage B done")


if __name__ == "__main__":
    OV.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    a = ap.parse_args()
    (stage_a if a.stage == "a" else stage_b)()
