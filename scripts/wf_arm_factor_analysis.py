"""Detailed per-arm factor analysis on the WF panel.

For one prerun (evolution WF arm, oneshot book, or the 101-alpha zoo):

  * per-factor pooled per-underlying IC on the fit window (< 2021-07-20),
    on each of the 10 phase-2 walk-forward blocks, block mean/std/hit-rate,
    plus a like-for-like in-sample block metric (trailing 126-bar chunks of
    the fit window) and signal coverage;
  * book diversity: mean |pairwise corr|, effective N (participation ratio);
  * combined-book models (ridge / lasso / lightgbm), fit ONCE on the pre-2021
    window, scored per block (static-fit reference — the PIT refit study is
    scripts/wf_pit_combiner_study.py);
  * for evolution arms: the run's own prequential record side-by-side.

Outputs under <out-root>/<arm>/:
  per_factor_blocks.csv, combined_static.csv, diversity.json,
  prequential_record.csv (evolution arms), REPORT.md

Resume-safe: existing output files are skipped (delete to recompute).

Usage:  python scripts/wf_arm_factor_analysis.py --arm L5WF_terra_s0
        python scripts/wf_arm_factor_analysis.py --arm zoo
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
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("arm_analysis")

H = 6
BLOCK = 126
WF_START = "2021-07-20"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
STATIC_MODELS = ("ridge", "lasso", "lightgbm")
REF_PREQUENTIAL = WS / "L4WF_terra_s0/evolution/prequential.jsonl"


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


def block_windows(arm: str, idx):
    """(generation, mask, start, end) for the 10 phase-2 prequential blocks."""
    import numpy as np
    import pandas as pd

    src = WS / arm / "evolution/prequential.jsonl"
    if not src.exists():
        src = REF_PREQUENTIAL
    blocks = []
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"]))
                           & (idx < pd.Timestamp(r["end"])))
            blocks.append((r["generation"], m, r["start"][:10], r["end"][:10]))
    return blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--out-root", default=str(REPO / "data/comparisons/wf_arm_analysis"))
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import _pooled_ic

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "combiner_lib", REPO / "scripts/analyze_combiner_models.py")
    saved = sys.argv
    sys.argv = ["x", "--panel", "wf"]
    combiner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(combiner)
    sys.argv = saved

    # version-tolerant estimators: extend the RidgeCV grid past its 1e4 edge
    # (known saturation) and survive newer sklearn dropping LassoCV(n_alphas=)
    _orig_make = combiner.make_estimator

    def _make(model, n_factors):
        from quant_fund_agent.modeling.catalog import _wrap
        if model == "ridge":
            from sklearn.linear_model import RidgeCV
            return _wrap(RidgeCV(alphas=np.logspace(-2, 6, 17)))
        if model == "lasso":
            try:
                return _orig_make(model, n_factors)
            except TypeError:
                from sklearn.linear_model import LassoCV
                return _wrap(LassoCV(cv=3, random_state=42))
        return _orig_make(model, n_factors)

    combiner.make_estimator = _make

    out_dir = Path(args.out_root) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    fit_mask = np.asarray(idx < pd.Timestamp(WF_START))
    fit_pos = np.flatnonzero(fit_mask)
    log.info("[%s] panel %d bars x %d tickers", args.arm, len(idx), close.shape[1])

    # trailing in-sample 126-bar blocks (like-for-like reference)
    is_blocks = []
    end = len(fit_pos)
    while end - BLOCK >= 0:
        m = np.zeros(len(idx), bool)
        m[fit_pos[end - BLOCK:end]] = True
        is_blocks.append(m)
        end -= BLOCK
    is_blocks.reverse()

    oos_blocks = block_windows(args.arm, idx)
    log.info("[%s] %d IS blocks, %d OOS blocks", args.arm,
             len(is_blocks), len(oos_blocks))

    # ── signals (shared parquet store — computed at most once ever) ─────────
    discover_factors()
    sys.path.insert(0, str(REPO / "scripts"))
    from wf_common import load_or_compute_signal
    book = load_book(args.arm)
    signals, failed = {}, []
    for fid, code in book.items():
        try:
            sig = load_or_compute_signal(fid, code, panel, idx, close.columns)
            cov = float(np.isfinite(sig.to_numpy()).mean())
            signals[fid] = (sig, cov)
        except Exception as e:  # noqa: BLE001
            failed.append({"factor_id": fid, "error": str(e)})
            log.warning("[%s] %s failed: %s", args.arm, fid, e)
    log.info("[%s] %d/%d signals computed", args.arm, len(signals), len(book))

    # ── per-factor blocks ───────────────────────────────────────────────────
    pf_path = out_dir / "per_factor_blocks.csv"
    if not pf_path.exists():
        rows = []
        for fid, (sig, cov) in signals.items():
            sigf = sig.astype(float)
            rec = {"factor_id": fid, "coverage": round(cov, 4),
                   "ic_fit": _pooled_ic(sigf, close, H, row_mask=fit_mask,
                                        available_mask=fit_mask)[0]}
            is_ics = [ic for m in is_blocks
                      if (ic := _pooled_ic(sigf, close, H, row_mask=m,
                                           available_mask=m)[0]) is not None]
            rec["ic_is_blockmean"] = float(np.mean(is_ics)) if is_ics else None
            rec["ic_is_blockstd"] = (float(np.std(is_ics))
                                     if len(is_ics) > 1 else None)
            bl = []
            for g, m, *_ in oos_blocks:
                ic = _pooled_ic(sigf, close, H, row_mask=m, available_mask=m)[0]
                rec[f"ic_g{g}"] = ic
                if ic is not None:
                    bl.append(ic)
            rec["ic_wf_blockmean"] = float(np.mean(bl)) if bl else None
            rec["ic_wf_blockstd"] = float(np.std(bl)) if len(bl) > 1 else None
            rec["wf_hit_rate"] = (float(np.mean([i > 0 for i in bl]))
                                  if bl else None)
            # sign-consistent retention: OOS mean IC aligned with fit sign
            if rec["ic_fit"] and rec["ic_wf_blockmean"] is not None:
                s = np.sign(rec["ic_fit"])
                rec["ic_wf_signed"] = s * rec["ic_wf_blockmean"]
                if rec["ic_is_blockmean"]:
                    denom = abs(rec["ic_is_blockmean"])
                    rec["retention"] = (rec["ic_wf_signed"] / denom
                                        if denom > 1e-6 else None)
            rows.append(rec)
        pd.DataFrame(rows).to_csv(pf_path, index=False)
        log.info("[%s] wrote per_factor_blocks.csv (%d rows)",
                 args.arm, len(rows))

    # ── diversity ───────────────────────────────────────────────────────────
    div_path = out_dir / "diversity.json"
    if not div_path.exists() and len(signals) >= 2:
        stride = max(1, int(fit_mask.sum()) // 400)
        rows_sel = fit_pos[::stride]
        mat = np.column_stack([
            np.nan_to_num(sig.astype(float).to_numpy()[rows_sel].ravel())
            for sig, _ in signals.values()])
        c = np.corrcoef(mat, rowvar=False)
        n = c.shape[0]
        off = c[np.triu_indices(n, 1)]
        off = off[np.isfinite(off)]
        eig = np.linalg.eigvalsh(np.nan_to_num(c))
        eig = np.clip(eig, 0, None)
        pr = float((eig.sum() ** 2) / (eig ** 2).sum()) if eig.sum() > 0 else None
        div = {"n_factors": n,
               "mean_abs_corr": float(np.mean(np.abs(off))) if off.size else None,
               "max_abs_corr": float(np.max(np.abs(off))) if off.size else None,
               "effective_n_participation_ratio": pr,
               "n_failed": len(failed), "failed": failed}
        div_path.write_text(json.dumps(div, indent=2))
        log.info("[%s] diversity: mean|rho|=%.3f effN=%.1f", args.arm,
                 div["mean_abs_corr"] or float("nan"), pr or float("nan"))

    # ── prequential record (evolution arms) ─────────────────────────────────
    prq_src = WS / args.arm / "evolution/prequential.jsonl"
    prq_path = out_dir / "prequential_record.csv"
    if prq_src.exists() and not prq_path.exists():
        rows = [json.loads(l) for l in prq_src.read_text().splitlines()]
        pd.DataFrame(rows).to_csv(prq_path, index=False)

    # ── combined static fits ────────────────────────────────────────────────
    comb_path = out_dir / "combined_static.csv"
    if not comb_path.exists():
        sigs = [s.astype(float) for s, _ in signals.values()]
        out = []
        for model in STATIC_MODELS:
            log.info("[%s] static fit %s (%d signals)...",
                     args.arm, model, len(sigs))
            try:
                pred, est = combiner.combined_prediction(
                    sigs, close, fit_mask, model)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s FAILED: %s", args.arm, model, e)
                continue
            rec = {"arm": args.arm, "model": model, "n_factors": len(sigs),
                   "diag": json.dumps(combiner.model_diag(est))}
            is_ics = [ic for m in is_blocks
                      if (ic := _pooled_ic(pred, close, H, row_mask=m,
                                           available_mask=m)[0]) is not None]
            rec["ic_is_blockmean"] = float(np.mean(is_ics)) if is_ics else None
            bl = []
            for g, m, *_ in oos_blocks:
                ic = _pooled_ic(pred, close, H, row_mask=m, available_mask=m)[0]
                rec[f"ic_g{g}"] = ic
                if ic is not None:
                    bl.append(ic)
            rec["ic_wf_blockmean"] = float(np.mean(bl)) if bl else None
            rec["ic_wf_blockstd"] = float(np.std(bl)) if len(bl) > 1 else None
            rec["wf_hit_rate"] = (float(np.mean([i > 0 for i in bl]))
                                  if bl else None)
            out.append(rec)
            del pred, est
        pd.DataFrame(out).to_csv(comb_path, index=False)
        log.info("[%s] wrote combined_static.csv", args.arm)

    # ── report ──────────────────────────────────────────────────────────────
    pf = pd.read_csv(pf_path)
    lines = [f"# {args.arm} — WF factor analysis", "",
             f"Factors: {len(pf)} computed / {len(book)} in book "
             f"({len(failed)} failed). Fit window < {WF_START}; OOS = 10 "
             "prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are "
             "pooled per-underlying; the WF statistic is the MEAN of the "
             "per-block ICs.", ""]
    ok = pf.dropna(subset=["ic_wf_blockmean"])
    if len(ok):
        lines += [
            f"- per-factor mean |IC| fit(blocks): "
            f"{ok.ic_is_blockmean.abs().mean():.4f} -> WF: "
            f"{ok.ic_wf_blockmean.abs().mean():.4f}",
            f"- sign-consistent retention (median): "
            f"{ok.retention.median():.2f}" if "retention" in ok else "",
            f"- factors with WF hit-rate >= 0.6: "
            f"{(ok.wf_hit_rate >= 0.6).sum()}/{len(ok)}",
            f"- best WF factor: {ok.loc[ok.ic_wf_blockmean.abs().idxmax(), 'factor_id']} "
            f"(blockmean {ok.ic_wf_blockmean.abs().max():.4f})", ""]
    if comb_path.exists():
        cb = pd.read_csv(comb_path)
        lines.append("| model | n | IS blockmean | WF blockmean | WF std | hit |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in cb.iterrows():
            lines.append(
                f"| {r.model} | {r.n_factors} | {r.ic_is_blockmean:.4f} | "
                f"{r.ic_wf_blockmean:.4f} | {r.ic_wf_blockstd:.4f} | "
                f"{r.wf_hit_rate:.0%} |")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    log.info("[%s] DONE -> %s", args.arm, out_dir)


if __name__ == "__main__":
    main()
