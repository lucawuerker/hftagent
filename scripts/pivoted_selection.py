#!/usr/bin/env python
"""Structural distinct-alpha selection via PIVOTED CHOLESKY on the correlation
matrix (supervisor follow-up 2026-08-22).

Given a book's fit-window correlation matrix C (per-underlying z-scored
signals, < 2021-07-20, the diversity convention), select k = round(participation
ratio) factors that are maximally independent USING ONLY C:

  Pivoted (column-pivoted) Cholesky: at step j pick the factor with the largest
  diagonal entry of the current Schur complement of C — i.e. the factor with
  the largest residual variance after regressing it on the already-selected
  factors (1 - R^2).  Update the Schur complement, repeat.  This greedily
  maximises det(C_SS) of the selected submatrix (maximum-volume subset) and is
  the Gram-side equivalent of column-pivoted QR on the data matrix.

Pure structure: the target/IC is never consulted.  For context the output also
reports each pick's fit-window IC and residual-variance share, and evaluates
the selected set with the SAME static-OLS walk-forward convention as
scripts/distinct_alpha_count.py (fit on the full fit window, mean of the 10
held-out block ICs) next to the full book.

Output: data/comparisons/distinct_alphas/<label>_pivoted.json.
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
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pivoted")

import numpy as np
import pandas as pd

OUT = REPO / "data/comparisons/distinct_alphas"
WF_START = pd.Timestamp("2021-07-20")
H = 6


def pivoted_cholesky_order(C: np.ndarray, k_max: int, tol: float = 1e-8):
    """Greedy max-residual-variance pivots on correlation matrix C.

    Returns (order, residual_var_at_pick): order[j] is the j-th selected
    column; residual_var_at_pick[j] is its Schur-complement diagonal (the
    share of its variance NOT explained by the j previously selected factors;
    1.0 for the first pick).
    """
    n = C.shape[0]
    d = np.array(np.diag(C), dtype=float).copy()      # current Schur diagonal
    L = np.zeros((n, k_max))
    order, resvar = [], []
    for j in range(min(k_max, n)):
        i = int(np.argmax(d))
        if d[i] < tol:
            break                                      # numerically rank-complete
        order.append(i)
        resvar.append(float(d[i]))
        piv = np.sqrt(d[i])
        # column of the Cholesky factor for pivot i
        col = (C[:, i] - L[:, :j] @ L[i, :j]) / piv
        L[:, j] = col
        d = d - col ** 2
        d[i] = -np.inf                                 # never repick
    return order, resvar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="override; default round(participation ratio)")
    ap.add_argument("--max-extra", type=int, default=10,
                    help="also report the order up to k + this many pivots")
    args = ap.parse_args()
    label = args.label or args.book.replace("+", "_plus_")
    OUT.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dac", REPO / "scripts/distinct_alpha_count.py")
    dac = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["x", "--book", args.book]
    spec.loader.exec_module(dac)
    sys.argv = saved

    from wf_common import load_or_compute_signal
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import (_label_available_mask,
                                                        _pooled_ic)

    book: dict[str, str] = {}
    for part in args.book.split("+"):
        book.update(dac.load_book(part))
    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    n_cols = close.shape[1]
    y_all = forward_returns(close, horizon=H).to_numpy(dtype=float).ravel()
    fit_mask = np.asarray(idx < WF_START)
    lab_ok = _label_available_mask(fit_mask, H)
    fit_idx = idx[fit_mask]

    fids, cols, ic_fit = [], [], {}
    for fid, code in sorted(book.items()):
        try:
            sig = load_or_compute_signal(fid, code, panel, idx, close.columns)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", fid, e)
            continue
        z = per_underlying_zscore(sig.astype(float), fit_idx)
        z = np.nan_to_num(z.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                          neginf=0.0).astype(np.float32)
        ic = _pooled_ic(sig.astype(float), close, H, row_mask=fit_mask,
                        available_mask=fit_mask)[0]
        if ic is None:
            continue
        ic_fit[fid] = float(ic)
        fids.append(fid)
        cols.append(z.ravel())
    X = np.column_stack(cols)
    del cols

    # correlation on the fit window (diversity convention: strided rows)
    fit_rows = np.repeat(np.asarray(fit_mask), n_cols)
    fr = np.flatnonzero(fit_rows)
    fr = fr[:: max(1, len(fr) // (400 * n_cols))]
    C = np.corrcoef(X[fr].astype(np.float64), rowvar=False)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 1.0)

    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    pr = float(eig.sum() ** 2 / (eig ** 2).sum())
    k = args.k or int(round(pr))
    order, resvar = pivoted_cholesky_order(C, k + args.max_extra)
    sel = order[:k]
    log.info("[%s] n=%d PR=%.2f -> k=%d; first pivots: %s",
             label, len(fids), pr, k, [fids[i] for i in sel[:5]])

    # static-OLS WF evaluation of the structural set (same convention as greedy)
    train = np.flatnonzero(np.repeat(np.asarray(fit_mask) & lab_ok, n_cols)
                           & np.isfinite(y_all))
    train = train[:: max(1, len(train) // 400_000)]
    blocks = []
    src = (REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
           / "L4WF_terra_s0/evolution/prequential.jsonl")
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"]))
                           & (idx < pd.Timestamp(r["end"])))
            blocks.append(m)

    def wf_mean(cols_sel: list[int]) -> float:
        Xf = X[train][:, cols_sel].astype(np.float64)
        w, *_ = np.linalg.lstsq(np.column_stack([Xf, np.ones(len(Xf))]),
                                y_all[train], rcond=None)
        pred = (X[:, cols_sel].astype(np.float64) @ w[:-1] + w[-1]).reshape(len(idx), n_cols)
        pred_df = pd.DataFrame(pred, index=idx, columns=close.columns)
        ics = [ic for m in blocks
               if (ic := _pooled_ic(pred_df, close, H, row_mask=m,
                                    available_mask=m)[0]) is not None]
        return float(np.mean(ics))

    wf_sel = wf_mean(sel)
    wf_full = wf_mean(list(range(len(fids))))
    res = {
        "book": args.book, "n_members": len(fids), "participation_ratio": pr,
        "k": k, "wf_pivoted_set": wf_sel, "wf_full_book": wf_full,
        "pivot_order": [{"factor_id": fids[i], "residual_var": round(rv, 4),
                         "ic_fit": round(ic_fit[fids[i]], 4)}
                        for i, rv in zip(order, resvar)],
        "selected_set": [fids[i] for i in sel],
    }
    (OUT / f"{label}_pivoted.json").write_text(json.dumps(res, indent=1))
    print(f"\n=== {label}: n={len(fids)} PR={pr:.2f} k={k} ===")
    print(f"WF mean block IC: pivoted set {wf_sel:.4f} vs full book {wf_full:.4f}")
    for i, rv in list(zip(order, resvar))[:k]:
        print(f"  {fids[i]:55s} resvar={rv:.3f} ic_fit={ic_fit[fids[i]]:+.4f}")


if __name__ == "__main__":
    main()
