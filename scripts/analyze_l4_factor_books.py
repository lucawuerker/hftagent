"""Factor-level analysis of the two big evolution books vs the 101 formulaic alphas.

Books:
- L4_terra_s0  (GPT-5.6 Terra, full L4 config, 44 published factors)
- L2_opus5_s0  (Claude Opus 5, 20-gen evolution, retrieval none, 18 factors)
  [there is NO Opus L4 run — the Opus ladder budget was exhausted after L2]
- formulaic_101 prebook (Kakushadze 101 alphas) as the reference zoo.

Panel: quant.config.nasdaq100_2010_forward.yaml (2010 -> 2026-07-27).
Windows (the runs' own convention, fractions on the research panel
2010 -> 2024-07-27): IS 60% | VAL 20% | TEST 20%; DEV = IS|VAL is what the
evolution searched on, TEST was never revealed, FORWARD (2024-07-28 ->
2026-07-27) was outside the research panel entirely.

Per factor: pooled per-underlying Pearson IC at h=6 (the harness statistic)
plus classic cross-sectional mean daily IC + ICIR, on DEV / TEST / FORWARD.
Combined books: lightgbm (the runs' marginal model) fit on DEV, scored on all
three windows, for {terra, opus, zoo, terra+zoo, opus+zoo}.
Correlations: cross-sectionally z-scored signals on DEV, pairwise Pearson;
effective number of factors (participation ratio); max |corr| vs the zoo.

Writes JSON/CSV + figure-ready data to data/comparisons/l4_factor_analysis/.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ["QF_CONFIG_FILE"] = "quant.config.nasdaq100_2010_forward.yaml"
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("l4_factor_analysis")

H = 6
RESEARCH_END = "2024-07-28"
OUT = REPO / "data/comparisons/l4_factor_analysis"

PRERUNS = {
    "terra": "L4_terra_s0",
    "opus": "L2_opus5_s0",
}
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def load_book(prerun: str):
    db = json.loads((WS / prerun / "factors/factor_db.json").read_text())
    out = {}
    meta = {}
    for rec in db["factors"]:
        code = Path(rec["code_path"]).read_text()
        out[rec["id"]] = code
        meta[rec["id"]] = {
            "name": rec.get("name"),
            "category": rec.get("category"),
            "required_inputs": rec.get("required_inputs"),
            "horizon": rec.get("prediction_horizon"),
        }
    return out, meta


def load_zoo():
    pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
    return {m["factor_id"]: m["code"] for m in pb["members"]}


def cs_daily_ic(sig, fwd):
    """Per-bar cross-sectional Pearson IC series (numpy, NaN-aware)."""
    import numpy as np
    x = sig.to_numpy(dtype=float)
    y = fwd.to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = np.where(m, x, np.nan)
    y = np.where(m, y, np.nan)
    n = m.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.nanmean(x, axis=1, keepdims=True)
        my = np.nanmean(y, axis=1, keepdims=True)
        xc = x - mx
        yc = y - my
        cov = np.nansum(xc * yc, axis=1)
        sx = np.sqrt(np.nansum(xc * xc, axis=1))
        sy = np.sqrt(np.nansum(yc * yc, axis=1))
        ic = cov / (sx * sy)
    ic[n < 10] = np.nan
    return ic


def main():
    import numpy as np
    import pandas as pd

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.backtesting.strategy_backtester import normalise_factor_signals
    from quant_fund_agent.research_eval.harness import _pooled_ic, _combined_prediction
    from quant_fund_agent.research_eval.splits import three_way_split

    OUT.mkdir(parents=True, exist_ok=True)

    fields = sorted(usable_fields())
    log.info("loading panel (%d fields)...", len(fields))
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    log.info("panel %d bars x %d tickers  %s -> %s", len(idx), close.shape[1],
             idx[0].date(), idx[-1].date())

    research_mask = np.asarray(idx < pd.Timestamp(RESEARCH_END))
    n_res = int(research_mask.sum())
    sp = three_way_split(n_res, is_frac=0.6, val_frac=0.2)
    dev_mask = np.zeros(len(idx), dtype=bool)
    test_mask = np.zeros(len(idx), dtype=bool)
    dev_mask[:n_res][sp.is_mask | sp.val_mask] = True
    test_mask[:n_res][sp.test_mask] = True
    fwd_mask = ~research_mask
    windows = {"dev": dev_mask, "test": test_mask, "forward": fwd_mask}
    for w, m in windows.items():
        log.info("window %-7s %s -> %s (%d bars)", w, idx[m][0].date(), idx[m][-1].date(),
                 int(m.sum()))

    # ── compute signals ──────────────────────────────────────────────────────
    discover_factors()
    books = {}
    metas = {}
    for label, prerun in PRERUNS.items():
        books[label], metas[label] = load_book(prerun)
    books["zoo"] = load_zoo()
    metas["zoo"] = {}

    signals = {}      # (book, fid) -> DataFrame
    failed = {}
    for label, codes in books.items():
        ok = 0
        for fid, code in codes.items():
            try:
                cls = get_factor_class(fid) or compile_factor(code, fid)
                sig = compute_signal(cls, panel)
                sig = sig.reindex(index=idx, columns=close.columns).astype("float32")
                cov = float(np.isfinite(sig.to_numpy()).mean())
                if cov < 0.01:
                    raise ValueError(f"coverage {cov:.3%}")
                signals[(label, fid)] = sig
                ok += 1
            except Exception as e:  # noqa: BLE001
                failed.setdefault(label, []).append(f"{fid}: {e}")
                log.warning("[%s] %s failed: %s", label, fid, e)
        log.info("[%s] %d/%d factors computed", label, ok, len(codes))

    fwd_h = forward_returns(close, horizon=H)
    fwd1_ret = fwd_h  # h-bar forward return frame for CS IC

    # ── per-factor ICs ──────────────────────────────────────────────────────
    rows = []
    for (label, fid), sig in signals.items():
        sigf = sig.astype(float)
        ics = cs_daily_ic(sigf, fwd_h)
        rec = {"book": label, "factor_id": fid,
               **{f"{k}": v for k, v in (metas[label].get(fid) or {}).items()}}
        for w, m in windows.items():
            pooled, nobs = _pooled_ic(sigf, close, H, row_mask=m, available_mask=m)
            s = ics[m]
            s = s[np.isfinite(s)]
            rec[f"ic_pooled_{w}"] = pooled
            rec[f"ic_cs_{w}"] = float(s.mean()) if len(s) else None
            rec[f"icir_cs_{w}"] = float(s.mean() / s.std()) if len(s) > 2 and s.std() > 0 else None
        rows.append(rec)
        log.info("IC %s/%s pooled dev=%.4f test=%.4f fwd=%.4f", label, fid,
                 rec["ic_pooled_dev"] or float("nan"),
                 rec["ic_pooled_test"] or float("nan"),
                 rec["ic_pooled_forward"] or float("nan"))
    per_factor = pd.DataFrame(rows)
    per_factor.to_csv(OUT / "per_factor_ic.csv", index=False)
    log.info("wrote per_factor_ic.csv (%d rows)", len(per_factor))

    # ── combined books (lightgbm fit on DEV) ────────────────────────────────
    import types
    cfg = types.SimpleNamespace(target_horizon=H, fit_standardize="per_underlying")

    def book_signals(*labels):
        out, keys = [], []
        for lb in labels:
            for (l2, fid), s in signals.items():
                if l2 == lb:
                    out.append(s.astype(float))
                    keys.append(f"{l2}:{fid}")
        return out, keys

    combos = {
        "terra_book": ["terra"],
        "opus_book": ["opus"],
        "zoo_101": ["zoo"],
        "terra_plus_zoo": ["terra", "zoo"],
        "opus_plus_zoo": ["opus", "zoo"],
        "terra_plus_opus": ["terra", "opus"],
        "terra_plus_opus_zoo": ["terra", "opus", "zoo"],
    }
    combined_rows = []
    combined_ic_series = {}
    for name, labels in combos.items():
        sigs, keys = book_signals(*labels)
        log.info("combined fit %s (%d signals)...", name, len(sigs))
        pred = _combined_prediction(sigs, close, dev_mask, cfg, "lightgbm")
        if pred is None:
            log.warning("combined %s: fit returned None", name)
            continue
        ics = cs_daily_ic(pred, fwd_h)
        combined_ic_series[name] = ics
        rec = {"combo": name, "n_factors": len(sigs)}
        for w, m in windows.items():
            pooled, nobs = _pooled_ic(pred, close, H, row_mask=m, available_mask=m)
            s = ics[m]
            s = s[np.isfinite(s)]
            rec[f"ic_pooled_{w}"] = pooled
            rec[f"ic_cs_{w}"] = float(s.mean()) if len(s) else None
            rec[f"icir_cs_{w}"] = float(s.mean() / s.std()) if len(s) > 2 and s.std() > 0 else None
        combined_rows.append(rec)
        log.info("combined %s: pooled dev=%.4f test=%.4f fwd=%.4f", name,
                 rec["ic_pooled_dev"] or float("nan"),
                 rec["ic_pooled_test"] or float("nan"),
                 rec["ic_pooled_forward"] or float("nan"))
        del sigs
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(OUT / "combined_book_ic.csv", index=False)
    pd.DataFrame(combined_ic_series, index=idx).to_csv(OUT / "combined_cs_ic_series.csv")

    # ── correlation structure (cross-sectional z, DEV window) ───────────────
    log.info("building correlation matrix...")
    cols = {}
    dev_idx_pos = np.flatnonzero(dev_mask)
    for (label, fid), sig in signals.items():
        z = normalise_factor_signals({"_": sig.astype(float)})["_"]
        cols[f"{label}:{fid}"] = z.to_numpy(dtype="float32")[dev_idx_pos].ravel()
    M = pd.DataFrame(cols)
    corr = M.corr(min_periods=200)
    corr.to_csv(OUT / "signal_corr_dev.csv")
    del M, cols
    log.info("corr matrix %s", corr.shape)

    # degenerate (variance-free) signals yield all-NaN correlation rows — drop
    # them from the summary/nearest statistics but keep them in the saved CSV.
    nan_frac = corr.isna().mean(axis=1)
    degenerate = list(nan_frac[nan_frac > 0.9].index)
    keep = [c for c in corr.columns if c not in degenerate]
    corr = corr.loc[keep, keep]

    def eff_n(sub):
        c = sub.fillna(0.0).to_numpy(dtype=float).copy()
        np.fill_diagonal(c, 1.0)
        ev = np.linalg.eigvalsh(c)
        ev = np.clip(ev, 0, None)
        return float((ev.sum() ** 2) / (ev ** 2).sum())

    def block(a, b):
        ka = [c for c in corr.columns if c.startswith(a + ":")]
        kb = [c for c in corr.columns if c.startswith(b + ":")]
        return corr.loc[ka, kb]

    summary = {"failed": failed, "degenerate_signals": degenerate, "windows": {
        w: [str(idx[m][0].date()), str(idx[m][-1].date()), int(m.sum())]
        for w, m in windows.items()}}
    for lb in ["terra", "opus", "zoo"]:
        sub = block(lb, lb)
        off = sub.to_numpy()[~np.eye(len(sub), dtype=bool)]
        off = off[np.isfinite(off)]
        summary[f"{lb}_n"] = len(sub)
        summary[f"{lb}_eff_n"] = eff_n(sub)
        summary[f"{lb}_mean_abs_corr"] = float(np.abs(off).mean())
        summary[f"{lb}_frac_abs_gt_0.5"] = float((np.abs(off) > 0.5).mean())
    for a, b in [("terra", "opus"), ("terra", "zoo"), ("opus", "zoo")]:
        x = block(a, b).to_numpy().ravel()
        x = x[np.isfinite(x)]
        summary[f"{a}_x_{b}_mean_abs_corr"] = float(np.abs(x).mean())
        summary[f"{a}_x_{b}_max_abs_corr"] = float(np.abs(x).max())
    for combo in [["terra", "zoo"], ["opus", "zoo"], ["terra", "opus", "zoo"]]:
        ks = [c for c in corr.columns if c.split(":")[0] in combo]
        summary["eff_n_" + "+".join(combo)] = eff_n(corr.loc[ks, ks])

    # per-book-factor max |corr| vs zoo (novelty vs the formulaic library)
    nearest = []
    for lb in ["terra", "opus"]:
        bz = block(lb, "zoo").abs()
        for fid in bz.index:
            j = bz.loc[fid].idxmax()
            nearest.append({"book": lb, "factor_id": fid.split(":", 1)[1],
                            "nearest_alpha": j.split(":", 1)[1],
                            "max_abs_corr_zoo": float(bz.loc[fid, j]),
                            "max_abs_corr_own_book": float(
                                block(lb, lb).abs().loc[fid].drop(fid).max())})
    pd.DataFrame(nearest).to_csv(OUT / "nearest_zoo_corr.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary: %s", json.dumps({k: v for k, v in summary.items()
                                        if k != "failed"}, indent=1, default=str))
    log.info("DONE -> %s", OUT)


if __name__ == "__main__":
    main()
