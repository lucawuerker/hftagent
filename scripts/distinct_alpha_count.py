#!/usr/bin/env python
"""How many DISTINCT alphas does a book contain — and which ones?

Two complementary, deterministic answers per book (supervisor question
2026-08-22):

1. **Greedy performance saturation** (the headline).  Factors are added one at
   a time by greedy forward selection; at each step the candidate that most
   improves the combined validation IC is picked.  Selection uses ONLY the
   pre-walk-forward fit window (train = first 80 % of the fit bars, selection
   scored on the last 20 % = tail-validation, OLS on per-underlying z-scored
   signals).  Every prefix book of the resulting order is then frozen, refit
   on the FULL fit window and scored on the 10 held-out walk-forward blocks
   (the chapter's static-combiner convention: mean of the 10 block ICs).
   k* = the smallest k whose WF mean block IC reaches 95 % of the full book's.
   The first k* picks are the requested representative high-performer set.

2. **Correlation clustering** (structural).  Average-linkage hierarchical
   clustering on 1-|rho| of the fit-window signals; the number of clusters at
   |rho| = 0.7 and 0.5 thresholds, with the highest-|fit-IC| member per
   cluster as representative.

Output: data/comparisons/distinct_alphas/<book>.json + summary printout.
Books resolved like the PIT study (zoo / prerun factor_db books).
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
log = logging.getLogger("distinct")

import numpy as np
import pandas as pd

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
OUT = REPO / "data/comparisons/distinct_alphas"
WF_START = pd.Timestamp("2021-07-20")
H = 6


def load_book(arm: str) -> dict[str, str]:
    if arm == "zoo":
        pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
        return {m["factor_id"]: m["code"] for m in pb["members"]}
    db = json.loads((WS / arm / "factors/factor_db.json").read_text())
    out = {}
    for r in db["factors"]:
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        out[r["id"]] = path.read_text()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True,
                    help="'zoo', a prerun name, or a '+'-joined union")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--saturation", type=float, default=0.95)
    args = ap.parse_args()
    label = args.label or args.book.replace("+", "_plus_")
    OUT.mkdir(parents=True, exist_ok=True)

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
        book.update(load_book(part))
    discover_factors()

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    n_cols = close.shape[1]
    y_all = forward_returns(close, horizon=H).to_numpy(dtype=float).ravel()
    fit_mask = np.asarray(idx < WF_START)
    lab_ok = _label_available_mask(fit_mask, H)

    # z-scored signal columns (fit-window stats), float32
    fit_idx = idx[fit_mask]
    fids, cols = [], []
    ic_fit = {}
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
    log.info("[%s] %d usable members", label, len(fids))

    # ── greedy forward selection on the fit window (train 80 / tail-val 20) ──
    fit_pos = np.flatnonzero(np.asarray(fit_mask) & lab_ok)
    cut = fit_pos[int(len(fit_pos) * 0.8)]
    tr_rows = np.repeat((np.arange(len(idx)) <= cut) & np.asarray(fit_mask) & lab_ok, n_cols)
    va_rows = np.repeat((np.arange(len(idx)) > cut) & np.asarray(fit_mask) & lab_ok, n_cols)
    tr = np.flatnonzero(tr_rows & np.isfinite(y_all))
    va = np.flatnonzero(va_rows & np.isfinite(y_all))
    tr = tr[:: max(1, len(tr) // 200_000)]      # stride: greedy scan speed
    ytr, yva = y_all[tr], y_all[va]

    def val_ic(sel: list[int]) -> float:
        Xtr = X[tr][:, sel].astype(np.float64)
        w, *_ = np.linalg.lstsq(np.column_stack([Xtr, np.ones(len(Xtr))]),
                                ytr, rcond=None)
        pred = X[va][:, sel].astype(np.float64) @ w[:-1] + w[-1]
        m = np.isfinite(yva)
        c = np.corrcoef(pred[m], yva[m])[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    order: list[int] = []
    curve_val = []
    remaining = list(range(len(fids)))
    best_prev = -np.inf
    while remaining and len(order) < args.max_k:
        scored = [(val_ic(order + [j]), j) for j in remaining]
        best, j = max(scored)
        order.append(j)
        remaining.remove(j)
        curve_val.append(best)
        log.info("[%s] k=%d  +%s  val IC %.4f", label, len(order), fids[j], best)
        best_prev = best

    # ── WF evaluation of every prefix (static convention) ────────────────────
    train_full = np.flatnonzero(np.repeat(np.asarray(fit_mask) & lab_ok, n_cols)
                                & np.isfinite(y_all))
    train_full = train_full[:: max(1, len(train_full) // 400_000)]
    blocks = []
    src = WS / "L4WF_terra_s0/evolution/prequential.jsonl"
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"]))
                           & (idx < pd.Timestamp(r["end"])))
            blocks.append((r["generation"], m))

    def wf_mean(sel: list[int]) -> float:
        Xf = X[train_full][:, sel].astype(np.float64)
        w, *_ = np.linalg.lstsq(np.column_stack([Xf, np.ones(len(Xf))]),
                                y_all[train_full], rcond=None)
        pred = (X[:, sel].astype(np.float64) @ w[:-1] + w[-1]).reshape(len(idx), n_cols)
        pred_df = pd.DataFrame(pred, index=idx, columns=close.columns)
        ics = [ic for _, m in blocks
               if (ic := _pooled_ic(pred_df, close, H, row_mask=m,
                                    available_mask=m)[0]) is not None]
        return float(np.mean(ics))

    ks = sorted(set(list(range(1, min(16, len(order) + 1)))
                    + [20, 25, 30, 40, len(order)]))
    ks = [k for k in ks if k <= len(order)]
    curve_wf = {k: wf_mean(order[:k]) for k in ks}
    full_wf = wf_mean(list(range(len(fids)))) if len(fids) > len(order) else curve_wf[len(order)]
    k_star = next((k for k in ks if curve_wf[k] >= args.saturation * full_wf),
                  len(order))
    log.info("[%s] full-book WF %.4f; k*=%d (%.0f%% saturation)",
             label, full_wf, k_star, args.saturation * 100)

    # ── correlation clustering ───────────────────────────────────────────────
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    stride = max(1, X.shape[0] // (400 * n_cols))
    c = np.corrcoef(X[::stride].astype(np.float64), rowvar=False)
    c = np.nan_to_num(c)
    dist = 1.0 - np.abs(c)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    clusters = {}
    for thr in (0.7, 0.5):
        lab_arr = fcluster(Z, t=1.0 - thr, criterion="distance")
        reps = {}
        for ci in np.unique(lab_arr):
            members = [fids[i] for i in np.flatnonzero(lab_arr == ci)]
            reps[int(ci)] = max(members, key=lambda f: abs(ic_fit[f]))
        clusters[str(thr)] = {"n_clusters": int(lab_arr.max()),
                              "representatives": sorted(reps.values())}

    res = {
        "book": args.book, "n_members": len(fids),
        "greedy_order": [fids[j] for j in order],
        "greedy_val_curve": curve_val,
        "wf_curve": {str(k): v for k, v in curve_wf.items()},
        "full_book_wf": full_wf, "k_star": int(k_star),
        "saturation": args.saturation,
        "representative_set": [fids[j] for j in order[:k_star]],
        "clusters": clusters,
    }
    (OUT / f"{label}.json").write_text(json.dumps(res, indent=1))
    print(f"\n=== {label}: {len(fids)} members ===")
    print(f"full-book WF mean block IC {full_wf:.4f}")
    print("WF curve:", {k: round(v, 4) for k, v in curve_wf.items()})
    print(f"k* ({args.saturation:.0%} saturation) = {k_star}")
    print("representative set:", res["representative_set"])
    for thr, cl in clusters.items():
        print(f"clusters at |rho|<{thr}: {cl['n_clusters']}")


if __name__ == "__main__":
    main()
