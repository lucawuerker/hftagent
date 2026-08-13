"""WF panel, block-metric edition: EVERYTHING as mean IC per 126-bar block.

The walk-forward OOS statistic is the mean of the 10 per-block ICs (the runs'
own prequential convention). For a like-for-like in-sample reference the FIT
window (2010 -> 2021-07-20) is chunked into trailing 126-bar blocks and the
combined model's (and each factor's) in-sample statistic is the mean per-block
IC there. Models are refit exactly as in analyze_combiner_models (fit on the
whole pre-2021 window; only the SCORING is per block).

Writes combined_wf_blocks.csv and per_factor_blocks_wf.csv (extended).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["QF_CONFIG_FILE"] = "quant.config.nasdaq100_2010_wf.yaml"
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("wf_blocks")

H = 6
BLOCK = 126
OUT = REPO / "data/comparisons/combiner_models"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
MODELS = ("ridge", "lasso", "lightgbm", "lgbm_l2n", "lgbm_l2")


def load_book(prerun):
    db = json.loads((WS / prerun / "factors/factor_db.json").read_text())
    out = {}
    for r in db["factors"]:
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        out[r["id"]] = path.read_text()
    return out


def load_zoo():
    pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
    return {m["factor_id"]: m["code"] for m in pb["members"]}


def main():
    import numpy as np
    import pandas as pd

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.research_eval.harness import _pooled_ic

    # load analyze_combiner_models as a module (it parses argv at import)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "combiner_lib", REPO / "scripts/analyze_combiner_models.py")
    # NOTE: that module parses argv at import; emulate it
    saved_argv = sys.argv
    sys.argv = ["x", "--panel", "wf"]
    combiner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(combiner)
    sys.argv = saved_argv

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()), n_tickers=None)
    close = panel["close"]
    idx = close.index

    wf_start = pd.Timestamp("2021-07-20")
    fit_mask = np.asarray(idx < wf_start)
    fit_pos = np.flatnonzero(fit_mask)

    # trailing 126-bar in-sample blocks (drop the oldest remainder)
    is_blocks = []
    end = len(fit_pos)
    while end - BLOCK >= 0:
        m = np.zeros(len(idx), bool)
        m[fit_pos[end - BLOCK:end]] = True
        is_blocks.append(m)
        end -= BLOCK
    is_blocks.reverse()
    log.info("%d in-sample blocks of %d bars", len(is_blocks), BLOCK)

    oos_blocks = []
    for line in (WS / "L4WF_terra_s0/evolution/prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["generation"] >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"])) & (idx < pd.Timestamp(r["end"])))
            oos_blocks.append((r["generation"], m))

    def block_stats(pred, blocks):
        ics = []
        for b in blocks:
            m = b[1] if isinstance(b, tuple) else b
            ic = _pooled_ic(pred, close, H, row_mask=m, available_mask=m)[0]
            if ic is not None:
                ics.append(ic)
        if not ics:
            return None, None, None
        return (float(np.mean(ics)), float(np.std(ics)),
                float(np.mean([i > 0 for i in ics])))

    discover_factors()
    books = {"l2wf": load_book("L2WF_terra_s0"), "l4wf": load_book("L4WF_terra_s0"),
             "zoo": load_zoo()}
    signals = {}
    for label, codes in books.items():
        for fid, code in codes.items():
            try:
                cls = get_factor_class(fid) or compile_factor(code, fid)
                sig = compute_signal(cls, panel).reindex(
                    index=idx, columns=close.columns).astype("float32")
                if float(np.isfinite(sig.to_numpy()).mean()) < 0.01:
                    raise ValueError("degenerate")
                signals[(label, fid)] = sig
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s failed: %s", label, fid, e)
        log.info("[%s] signals ready", label)

    # per-factor block metrics
    rows = []
    for (label, fid), sig in signals.items():
        sigf = sig.astype(float)
        m_is = block_stats(sigf, is_blocks)
        m_oos = block_stats(sigf, oos_blocks)
        rows.append({"book": label, "factor_id": fid,
                     "ic_fit_blockmean": m_is[0], "ic_fit_blockstd": m_is[1],
                     "ic_wf_blockmean": m_oos[0], "ic_wf_blockstd": m_oos[1]})
    pd.DataFrame(rows).to_csv(OUT / "per_factor_blockmetric_wf.csv", index=False)
    log.info("per-factor block metric written")

    combos = {"l2wf": ["l2wf"], "l4wf": ["l4wf"], "zoo": ["zoo"],
              "l2wf_plus_l4wf": ["l2wf", "l4wf"]}
    out = []
    for combo, labels in combos.items():
        sigs = [s.astype(float) for (l, f), s in signals.items() if l in labels]
        for model in MODELS:
            log.info("fit %s/%s ...", combo, model)
            pred, est = combiner.combined_prediction(sigs, close, fit_mask, model)
            is_m, is_s, is_h = block_stats(pred, is_blocks)
            oos_m, oos_s, oos_h = block_stats(pred, oos_blocks)
            per_block = {}
            for g, m in oos_blocks:
                per_block[f"g{g}"] = _pooled_ic(pred, close, H, row_mask=m,
                                                available_mask=m)[0]
            out.append({"combo": combo, "model": model, "n_factors": len(sigs),
                        "ic_fit_blockmean": is_m, "ic_fit_blockstd": is_s,
                        "ic_wf_blockmean": oos_m, "ic_wf_blockstd": oos_s,
                        "wf_hit_rate": oos_h, "diag": combiner.model_diag(est),
                        **per_block})
            log.info("  %s/%s: IS-blockmean=%.4f WF-blockmean=%.4f hit=%.0f%%",
                     combo, model, is_m, oos_m, 100 * oos_h)
            del pred, est
    pd.DataFrame(out).to_csv(OUT / "combined_wf_blocks.csv", index=False)
    log.info("DONE -> combined_wf_blocks.csv")


if __name__ == "__main__":
    main()
