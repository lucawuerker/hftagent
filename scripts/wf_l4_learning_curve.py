"""L4WF learning curve: the archive AS OF each generation, combined with
ic-weights fit strictly before a fixed evaluation window (the last three
prequential blocks, 2024-12→2026-06), scored on that window.

Only book COMPOSITION varies across generations — fit window, weights recipe
and evaluation window are identical — so the curve isolates what the
evolution's selection actually improved.  Snapshots at generation >= 18 saw
part of the evaluation window during selection (reveal schedule) and are
flagged.

Output: data/comparisons/wf_book_analysis/derived/l4_learning_curve.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("curve")

import numpy as np
import pandas as pd

RAW = REPO / "data/comparisons/wf_book_analysis/raw/L4WF_terra_s0/evolution"
OUT = REPO / "data/comparisons/wf_book_analysis/derived/l4_learning_curve.csv"
H = 6
EVAL_FROM_GEN = 18   # eval window = blocks 18..20


def main() -> None:
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc

    st = json.load((RAW / "state.json").open())
    code = {}
    for e in st["kept_pool"] + st["archive"]:
        g = e["genome"]
        code[g["genome_id"]] = (g["programs"][0]["factor_id"],
                                g["programs"][0]["code"])
    members = defaultdict(set)
    for line in (RAW / "lineage.jsonl").open():
        r = json.loads(line)
        if r.get("event") == "rescore":
            members[r["generation"]].add(r["genome_id"])

    preq = [json.loads(l) for l in (RAW / "prequential.jsonl").open()]
    eval_start = pd.Timestamp(next(r["start"] for r in preq
                                   if r["generation"] == EVAL_FROM_GEN))
    eval_end = pd.Timestamp(next(r["end"] for r in preq
                                 if r["generation"] == 20))
    log.info("fixed eval window %s -> %s", eval_start.date(), eval_end.date())

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    fit_mask = np.asarray(idx < eval_start)
    eval_mask = np.asarray((idx >= eval_start) & (idx < eval_end))
    fwd = (close.shift(-H) / close - 1).to_numpy()

    def pooled_ic(sig_z: np.ndarray, mask: np.ndarray) -> float:
        num_w = 0.0
        num = 0.0
        for j in range(sig_z.shape[1]):
            x, y = sig_z[mask, j], fwd[mask, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
                continue
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            if np.isfinite(r):
                num += r * ok.sum()
                num_w += ok.sum()
        return num / num_w if num_w else np.nan

    needed = sorted({gid for s in members.values() for gid in s})
    z_by_gid: dict[str, np.ndarray] = {}
    w_by_gid: dict[str, float] = {}
    for i, gid in enumerate(needed):
        fid, src = code[gid]
        try:
            cls = compile_factor(src, fid)
            sig = compute_signal(cls, panel).reindex(index=idx,
                                                     columns=close.columns)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed (%s) — skipped", fid, e)
            continue
        v = sig.to_numpy(dtype=float)
        mu = np.nanmean(np.where(fit_mask[:, None], v, np.nan), axis=0)
        sd = np.nanstd(np.where(fit_mask[:, None], v, np.nan), axis=0)
        sd[sd == 0] = np.nan
        z = (v - mu) / sd
        z_by_gid[gid] = z
        w_by_gid[gid] = pooled_ic(z, fit_mask)
        if (i + 1) % 25 == 0:
            log.info("signals %d/%d", i + 1, len(needed))

    rows = []
    for g in sorted(members):
        gids = [x for x in members[g] if x in z_by_gid
                and np.isfinite(w_by_gid.get(x, np.nan))]
        if not gids:
            continue
        P = np.zeros_like(fwd)
        W = np.zeros_like(fwd)
        for gid in gids:
            z = z_by_gid[gid]
            w = w_by_gid[gid]
            ok = np.isfinite(z)
            P[ok] += w * z[ok]
            W[ok] += abs(w)
        with np.errstate(invalid="ignore"):
            P = np.where(W > 0, P / W, np.nan)
        rows.append({"generation": g, "n_members": len(gids),
                     "ic_fixed_eval": pooled_ic(P, eval_mask),
                     "eval_seen_by_selection": g >= EVAL_FROM_GEN})
        log.info("gen %d: n=%d ic=%.4f", g, len(gids), rows[-1]["ic_fixed_eval"])

    pd.DataFrame(rows).to_csv(OUT, index=False)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
