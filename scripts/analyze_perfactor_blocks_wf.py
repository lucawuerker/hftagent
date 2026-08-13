"""Per-factor IC per walk-forward block (WF panel) -> per_factor_blocks_wf.csv.

For each factor of L2WF / L4WF / the 101-alpha zoo and each of the 10 phase-2
prequential blocks (dates from L4WF's prequential.jsonl), the pooled
per-underlying IC inside that block; plus the fit-window IC. The report's WF
OOS statistic is the MEAN of the per-block ICs (the user's convention:
one IC per walk-forward step, then average).
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
log = logging.getLogger("pf_blocks")

H = 6
OUT = REPO / "data/comparisons/combiner_models"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def main():
    import numpy as np
    import pandas as pd

    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.research_eval.harness import _pooled_ic

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

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()), n_tickers=None)
    close = panel["close"]
    idx = close.index

    fit_mask = np.asarray(idx < pd.Timestamp("2021-07-20"))
    blocks = []
    for line in (WS / "L4WF_terra_s0/evolution/prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["generation"] >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"])) & (idx < pd.Timestamp(r["end"])))
            blocks.append((r["generation"], m))
    log.info("%d blocks", len(blocks))

    discover_factors()
    books = {"l2wf": load_book("L2WF_terra_s0"), "l4wf": load_book("L4WF_terra_s0"),
             "zoo": load_zoo()}
    rows = []
    for label, codes in books.items():
        for fid, code in codes.items():
            try:
                cls = get_factor_class(fid) or compile_factor(code, fid)
                sig = compute_signal(cls, panel).reindex(
                    index=idx, columns=close.columns).astype(float)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s failed: %s", label, fid, e)
                continue
            rec = {"book": label, "factor_id": fid,
                   "ic_fit": _pooled_ic(sig, close, H, row_mask=fit_mask,
                                        available_mask=fit_mask)[0]}
            bl = []
            for g, m in blocks:
                ic = _pooled_ic(sig, close, H, row_mask=m, available_mask=m)[0]
                rec[f"ic_g{g}"] = ic
                if ic is not None:
                    bl.append(ic)
            rec["ic_block_mean"] = float(np.mean(bl)) if bl else None
            rec["ic_block_std"] = float(np.std(bl)) if len(bl) > 1 else None
            rows.append(rec)
        log.info("[%s] done", label)
    pd.DataFrame(rows).to_csv(OUT / "per_factor_blocks_wf.csv", index=False)
    log.info("DONE -> per_factor_blocks_wf.csv (%d rows)", len(rows))


if __name__ == "__main__":
    main()
