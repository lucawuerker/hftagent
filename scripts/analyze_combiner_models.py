"""Combiner-model study: is the DEV->OOS collapse of the combined-book IC a
property of the fitted model (LightGBM default) and how do regularised
alternatives behave?

Two panels (run the script once per --panel; separate processes because the
panel cache is config-blind):

  --panel forward   quant.config.nasdaq100_2010_forward.yaml
                    books: L4_terra_s0 (terra), L2_opus5_s0 (opus), zoo 101
                    fit = DEV (first 80% of research panel, ->2021-08-25)
                    oos = TEST (->2024-07-26), FORWARD (->2026-07-27)

  --panel wf        quant.config.nasdaq100_2010_wf.yaml
                    books: L2WF_terra_s0 (l2wf), L4WF_terra_s0 (l4wf), zoo 101
                    fit = bars < 2021-07-20 (phase-1 span of the WF schedule)
                    oos = 2021-07-20 -> 2026-07-27 (the prequential span),
                    plus per-block ICs on the 10 phase-2 blocks (dates from
                    prequential.jsonl) for comparison with the runs' honest
                    prequential record.

Models: ridge, lasso (catalog, CV-self-tuned), lightgbm (catalog default),
lgbm_l2n (reg_lambda = 1*n_factors), lgbm_l2 (reg_lambda = 10*n_factors).

Results appended incrementally to <out>/combiner_<panel>.jsonl (resume-safe).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

p = argparse.ArgumentParser()
p.add_argument("--panel", choices=["forward", "wf"], required=True)
ARGS = p.parse_args()

CFG = {"forward": "quant.config.nasdaq100_2010_forward.yaml",
       "wf": "quant.config.nasdaq100_2010_wf.yaml"}[ARGS.panel]
os.environ["QF_CONFIG_FILE"] = CFG
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("combiner")

H = 6
OUT = REPO / "data/comparisons/combiner_models"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
MODELS = ("ridge", "lasso", "lightgbm", "lgbm_l2n", "lgbm_l2")


def load_book(prerun):
    db = json.loads((WS / prerun / "factors/factor_db.json").read_text())
    out = {}
    for r in db["factors"]:
        # server-produced records carry the server's absolute code_path —
        # remap onto the local shared researcher package by basename.
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        out[r["id"]] = path.read_text()
    return out


def load_zoo():
    pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
    return {m["factor_id"]: m["code"] for m in pb["members"]}


def make_estimator(model, n_factors):
    from quant_fund_agent.modeling.catalog import _wrap, build_estimator
    if model in ("lgbm_l2n", "lgbm_l2"):
        from lightgbm import LGBMRegressor
        lam = float(n_factors) * (1.0 if model == "lgbm_l2n" else 10.0)
        # identical to the catalog's lightgbm defaults + the L2 leaf penalty
        return _wrap(LGBMRegressor(
            n_estimators=200, num_leaves=15, max_depth=-1, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=lam,
            random_state=42, n_jobs=-1, verbose=-1))
    return build_estimator(model, None)


def combined_prediction(signals, close, is_mask, model):
    """Mirror of harness._combined_prediction with an injectable estimator."""
    import numpy as np
    import pandas as pd
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.research_eval.harness import _label_available_mask

    index, cols = close.index, close.columns
    n_rows, n_cols = len(index), len(cols)
    is_idx = index[is_mask]
    feats = []
    for sig in signals:
        s = sig.reindex(index=index, columns=cols).replace([np.inf, -np.inf], np.nan)
        z = per_underlying_zscore(s, is_idx)
        feats.append(np.nan_to_num(z.to_numpy(dtype=float).ravel(),
                                   nan=0.0, posinf=0.0, neginf=0.0))
    X = np.column_stack(feats)
    del feats
    y = forward_returns(close, horizon=H).to_numpy(dtype=float).ravel()
    fit_mask = np.asarray(is_mask) & _label_available_mask(is_mask, H)
    is_rows = np.repeat(fit_mask, n_cols)
    train = np.flatnonzero(is_rows & np.isfinite(y))
    est = make_estimator(model, len(signals))
    est.fit(X[train], y[train])
    pred = np.asarray(est.predict(X), dtype=float).reshape(n_rows, n_cols)
    return pd.DataFrame(pred, index=index, columns=cols), est


def model_diag(est):
    """Best-effort complexity diagnostics of a fitted (wrapped) estimator."""
    import numpy as np
    from quant_fund_agent.modeling.catalog import unwrap_estimator
    try:
        inner = unwrap_estimator(est)
        d = {"inner": type(inner).__name__}
        if hasattr(inner, "alpha_"):
            d["alpha"] = float(inner.alpha_)
        if hasattr(inner, "coef_"):
            c = np.asarray(inner.coef_).ravel()
            d["n_nonzero_coef"] = int((np.abs(c) > 1e-10).sum())
            d["n_coef"] = int(c.size)
        return d
    except Exception:  # noqa: BLE001
        return {}


def main():
    import numpy as np
    import pandas as pd

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.research_eval.harness import _pooled_ic

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"combiner_{ARGS.panel}.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["combo"], r["model"]))
        log.info("resume: %d results already present", len(done))

    fields = sorted(usable_fields())
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    log.info("panel %s: %d bars x %d tickers %s -> %s", ARGS.panel, len(idx),
             close.shape[1], idx[0].date(), idx[-1].date())

    blocks = []  # (label, mask) extra scoring windows
    if ARGS.panel == "forward":
        research = np.asarray(idx < pd.Timestamp("2024-07-28"))
        n_res = int(research.sum())
        cut = int(n_res * 0.8)
        fit_mask = np.zeros(len(idx), bool)
        fit_mask[:cut] = True
        oos = {"test": np.zeros(len(idx), bool), "forward": np.asarray(~research)}
        oos["test"][cut:n_res] = True
        books = {"terra": load_book("L4_terra_s0"), "opus": load_book("L2_opus5_s0"),
                 "zoo": load_zoo()}
        combos = {"terra": ["terra"], "opus": ["opus"], "zoo": ["zoo"],
                  "terra_plus_opus": ["terra", "opus"],
                  "terra_plus_opus_zoo": ["terra", "opus", "zoo"]}
    else:
        wf_start = pd.Timestamp("2021-07-20")
        fit_mask = np.asarray(idx < wf_start)
        oos = {"wf": np.asarray(idx >= wf_start)}
        books = {"l2wf": load_book("L2WF_terra_s0"), "l4wf": load_book("L4WF_terra_s0"),
                 "zoo": load_zoo()}
        combos = {"l2wf": ["l2wf"], "l4wf": ["l4wf"], "zoo": ["zoo"],
                  "l2wf_plus_l4wf": ["l2wf", "l4wf"]}
        # phase-2 block windows from the prequential record (same for both arms)
        for line in (WS / "L4WF_terra_s0/evolution/prequential.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["generation"] >= 11:
                m = np.asarray((idx >= pd.Timestamp(r["start"])) & (idx < pd.Timestamp(r["end"])))
                blocks.append((f"block_g{r['generation']}", m,
                               r["start"][:10], r["end"][:10]))

    log.info("fit window: %s -> %s (%d bars)", idx[fit_mask][0].date(),
             idx[fit_mask][-1].date(), int(fit_mask.sum()))

    # ── signals ─────────────────────────────────────────────────────────────
    discover_factors()
    signals = {}
    for label, codes in books.items():
        ok = 0
        for fid, code in codes.items():
            try:
                cls = get_factor_class(fid) or compile_factor(code, fid)
                sig = compute_signal(cls, panel).reindex(
                    index=idx, columns=close.columns).astype("float32")
                if float(np.isfinite(sig.to_numpy()).mean()) < 0.01:
                    raise ValueError("degenerate coverage")
                signals[(label, fid)] = sig
                ok += 1
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s failed: %s", label, fid, e)
        log.info("[%s] %d/%d signals", label, ok, len(codes))

    # ── per-factor pooled ICs (fit + oos windows) ───────────────────────────
    pf_path = OUT / f"per_factor_{ARGS.panel}.csv"
    if not pf_path.exists():
        rows = []
        for (label, fid), sig in signals.items():
            sigf = sig.astype(float)
            rec = {"book": label, "factor_id": fid}
            rec["ic_fit"] = _pooled_ic(sigf, close, H, row_mask=fit_mask,
                                       available_mask=fit_mask)[0]
            for w, m in oos.items():
                rec[f"ic_{w}"] = _pooled_ic(sigf, close, H, row_mask=m,
                                            available_mask=m)[0]
            rows.append(rec)
        pd.DataFrame(rows).to_csv(pf_path, index=False)
        log.info("wrote %s", pf_path.name)

    # ── combined fits ───────────────────────────────────────────────────────
    def sigs_for(labels):
        return [s.astype(float) for (l, f), s in signals.items() if l in labels]

    for combo, labels in combos.items():
        sigs = sigs_for(labels)
        for model in MODELS:
            if (combo, model) in done:
                continue
            log.info("fit %s / %s (%d signals)...", combo, model, len(sigs))
            try:
                pred, est = combined_prediction(sigs, close, fit_mask, model)
            except Exception as e:  # noqa: BLE001
                log.warning("fit %s/%s FAILED: %s", combo, model, e)
                continue
            rec = {"panel": ARGS.panel, "combo": combo, "model": model,
                   "n_factors": len(sigs), "diag": model_diag(est)}
            rec["ic_fit"] = _pooled_ic(pred, close, H, row_mask=fit_mask,
                                       available_mask=fit_mask)[0]
            for w, m in oos.items():
                rec[f"ic_{w}"] = _pooled_ic(pred, close, H, row_mask=m,
                                            available_mask=m)[0]
            for b in blocks:
                name, m = b[0], b[1]
                rec.setdefault("blocks", {})[name] = {
                    "start": b[2], "end": b[3],
                    "ic": _pooled_ic(pred, close, H, row_mask=m, available_mask=m)[0]}
            with open(res_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            log.info("  %s/%s: fit=%.4f %s", combo, model, rec["ic_fit"] or float("nan"),
                     {w: round(rec[f"ic_{w}"], 4) if rec[f"ic_{w}"] else None for w in oos})
            del pred, est
    log.info("DONE %s -> %s", ARGS.panel, res_path)


if __name__ == "__main__":
    main()
