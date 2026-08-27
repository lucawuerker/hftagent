"""Cross-sectional IC robustness check for the ablation chapter.

The thesis scores every combined forecast with the *pooled per-underlying*
information coefficient of Definition 1.0.3 (``_pooled_ic``): a Pearson
correlation computed WITHIN each asset over time, then averaged across assets
with observation-count weights.  That is a time-series ("timing") IC.

The field standard for cross-sectional equity factors is a different statistic:
the cross-sectional IC computed ACROSS assets within each date and then averaged
over dates (Grinold & Kahn; Qlib; Alphalens), usually also as a Spearman RankIC.

This script rescores the *already fitted* point-in-time combiner predictions --
cached per block as ``pred_<method>.parquet`` by ``wf_pit_combiner_study.py`` --
under all three metrics on exactly the same ten walk-forward blocks, the same
rows and the same forward-return labels.  Nothing is refitted, so any difference
is attributable to the metric alone.

Outputs
    data/comparisons/thesis_ablation/tables/xs_ic_robustness.csv   (per label/method)
    data/comparisons/thesis_ablation/tables/xs_ic_blocks.csv       (per block)
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("xs_ic")

H = 6
MIN_ASSETS = 5
LOCAL = REPO / "data/comparisons/wf_arm_analysis_local/pit_combiners"
RAW = REPO / "data/comparisons/wf_book_analysis/raw"
OUT = REPO / "data/comparisons/thesis_ablation/tables"

# thesis arm -> PIT label carrying the arm's published BOOK race.
# (Arms 3/4 use the *_s0b curated reruns, which is what Table 4.2 reports.)
ARM_LABELS: dict[str, str] = {
    "1": "LDU8CUR_terra_s0",
    "2": "LDP8CUR_terra_s0",
    "3": "LDGCUR_terra_s0b",
    "3b": "LDGCLNC_4omini_s0",
    "4": "L1HCUR_terra_s0b",
    "4b": "L1HB4OMINICUR_s0",
    "5": "L2WFP_terra_s0",
    "6": "L4WF_terra_s0",
    "7": "L5WF_terra_s0",
    "8": "L1HBDCUR_terra_s0",
    "9": "L0WF_gp_s0",
    "9c": "L0WFCLEAN_gp_s0",
    "Z": "zoo",
}


WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def artifact_root(label: str) -> Path | None:
    """Directory holding the g11..g20 subdirs of cached per-block fits.

    Two vintages exist: the newer runs persisted ``pred_<method>.parquet``, the
    older evolution-arm runs only ``weights_<method>.json``.  Both are usable —
    see :func:`reconstruct_pred`.
    """
    cands = [LOCAL / "artifacts" / label,
             RAW / label / "pit/artifacts" / label,
             RAW / label / "pit/artifacts"]
    for c in cands:
        if c.is_dir() and any(c.glob("g*/pred_*.parquet")):
            return c
    for c in cands:
        if c.is_dir() and any(c.glob("g*/weights_*.json")):
            return c
    return None


def load_codes(label: str) -> dict[str, str]:
    """factor_id -> source, from the run's factor DB plus its kept pool."""
    import json

    run = label
    prerun = WS / run
    if not prerun.is_dir():
        raise FileNotFoundError(f"no prerun for {label}")
    out: dict[str, str] = {}
    db = json.loads((prerun / "factors/factor_db.json").read_text())
    for r in db["factors"]:
        p = Path(r["code_path"])
        if not p.exists():
            p = REPO / "quant_fund_agent/factors/researcher" / p.name
        if p.exists():
            out[r["id"]] = p.read_text()
    # the kept pool (superset of the published book) lives in the evolution
    # state, which for the older evolution arms sits under the analysis raw dir
    for state in (prerun / "evolution/state.json",
                  RAW / run / "evolution/state.json"):
        if not state.exists():
            continue
        st = json.loads(state.read_text())
        for eg in st.get("kept_pool", []) + st.get("archive", []):
            for prog in eg["genome"]["programs"]:
                out.setdefault(prog["factor_id"], prog["code"])
    return out


def signal_from_store(fid: str, idx, cols):
    """Cached signal panel for ``fid``, recovered by factor id alone.

    The store is keyed ``<safe_fid>__<codehash>.parquet``.  A factor id that
    resolves to several hashes is ambiguous and refused rather than guessed.
    """
    import pandas as pd

    sys.path.insert(0, str(REPO / "scripts"))
    from wf_common import SIGNAL_STORE

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in fid)[:80]
    hits = sorted(SIGNAL_STORE.glob(f"{safe}__*.parquet"))
    if len(hits) != 1:
        if len(hits) > 1:
            log.warning("signal store: %s is ambiguous (%d variants)", fid, len(hits))
        return None
    return pd.read_parquet(hits[0]).reindex(index=idx, columns=cols).astype("float32")


def reconstruct_pred(gdir: Path, method: str, label: str, panel, close, idx,
                     codes: dict[str, str]):
    """Rebuild a linear combiner's prediction from its persisted weights.

    ``wf_pit_combiner_study`` fits ``pred = X @ w`` where X holds the
    per-underlying z-scored signals (fit-window statistics, NaN -> 0) of the
    block's point-in-time factor list.  The intercept is irrelevant here: an
    additive constant cancels in every correlation we compute.
    """
    import json

    import numpy as np
    import pandas as pd

    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    sys.path.insert(0, str(REPO / "scripts"))
    from wf_common import load_or_compute_signal

    meta = json.loads((gdir / "factors.json").read_text())
    wj = json.loads((gdir / f"weights_{method}.json").read_text())
    fids, w = wj["fids"], np.asarray(wj["weights"], dtype=float)
    start = pd.Timestamp(meta["start"])
    end = pd.Timestamp(meta["end"])
    fit_idx = idx[idx < start]

    acc = np.zeros((len(idx), close.shape[1]), dtype=float)
    used = 0
    for f, wt in zip(fids, w):
        if wt == 0.0:
            used += 1
            continue
        code = codes.get(f)
        if code is not None:
            sig = load_or_compute_signal(f, code, panel, idx, close.columns)
        else:
            # code not in this prerun's DB (older run, program only lived in a
            # snapshot) — the signal store keeps the computed panel keyed by
            # fid+codehash, so recover it by factor id alone.
            sig = signal_from_store(f, idx, close.columns)
            if sig is None:
                log.warning("[%s] %s: no code and no cached signal for %s",
                            label, gdir.name, f)
                continue
        z = per_underlying_zscore(sig.astype(float), fit_idx).to_numpy(dtype=float)
        acc += wt * np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        used += 1
    if used < len(fids):
        log.warning("[%s] %s %s: %d/%d factors reconstructed",
                    label, gdir.name, method, used, len(fids))
    block = (idx >= start) & (idx < end)
    return pd.DataFrame(acc, index=idx, columns=close.columns)[block]


def xs_ic_per_date(pred, fwd, *, method: str = "pearson"):
    """Cross-sectional IC per date -> pd.Series indexed by date.

    Demeans (or ranks) across assets within each row, exactly the convention of
    ``backtesting.engine._ic_series``.  Rows with fewer than ``MIN_ASSETS``
    finite pairs are dropped.
    """
    import numpy as np

    valid = pred.notna() & fwd.notna()
    n_valid = valid.sum(axis=1)
    x = pred.where(valid)
    y = fwd.where(valid)
    if method == "spearman":
        x = x.rank(axis=1)
        y = y.rank(axis=1)
    x = x.sub(x.mean(axis=1), axis=0)
    y = y.sub(y.mean(axis=1), axis=0)
    cov = (x * y).sum(axis=1)
    denom = (x.pow(2).sum(axis=1).pow(0.5) * y.pow(2).sum(axis=1).pow(0.5))
    ic = cov / denom.replace(0, np.nan)
    return ic[n_valid >= MIN_ASSETS].dropna()


def main() -> None:
    import numpy as np
    import pandas as pd

    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.ic import _weighted_asset_pearson
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import _label_available_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARM_LABELS))
    ap.add_argument("--methods", default="lasso,lightgbm,rf,ridge,ic,equal")
    args = ap.parse_args()
    want_methods = set(args.methods.split(","))

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    fwd_full = forward_returns(close, horizon=H)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    log.info("panel %s -> %s, %d bars x %d tickers",
             idx[0].date(), idx[-1].date(), len(idx), close.shape[1])

    block_rows, summary_rows = [], []
    for arm in args.arms.split(","):
        label = ARM_LABELS.get(arm, arm)
        root = artifact_root(label)
        if root is None:
            log.warning("arm %-3s (%s): no cached predictions -- skipped", arm, label)
            continue
        by_method: dict[str, list[dict]] = {}
        has_pred = any(root.glob("g*/pred_*.parquet"))
        codes = {} if has_pred else load_codes(label)
        for gdir in sorted(root.glob("g*"), key=lambda p: int(p.name[1:])):
            g = int(gdir.name[1:])
            if has_pred:
                sources = [(pq.stem[len("pred_"):], pq) for pq
                           in sorted(gdir.glob("pred_*.parquet"))]
            else:
                sources = [(wj.stem[len("weights_"):], wj) for wj
                           in sorted(gdir.glob("weights_*.json"))]
            for method, src in sources:
                if method not in want_methods:
                    continue
                if has_pred:
                    pred = pd.read_parquet(src)
                else:
                    pred = reconstruct_pred(gdir, method, label, panel, close,
                                            idx, codes)
                rows = pred.index.intersection(idx)
                if len(rows) == 0:
                    continue
                # rebuild the block mask on the panel grid from the cached rows
                block_mask = np.zeros(len(idx), dtype=bool)
                block_mask[pos.loc[rows].to_numpy()] = True
                scored = block_mask & _label_available_mask(block_mask, H)

                # cached predictions are float32; squaring them in the
                # cross-sectional sums overflows, so widen before scoring
                p_full = pred.reindex(index=idx, columns=close.columns).astype(float)
                p = p_full[scored]
                f = fwd_full[scored].astype(float)

                ts_ic, n_obs = _weighted_asset_pearson(
                    p.to_numpy(dtype=float), f.to_numpy(dtype=float))
                xs = xs_ic_per_date(p, f, method="pearson")
                rk = xs_ic_per_date(p, f, method="spearman")
                rec = {
                    "arm": arm, "label": label, "method": method, "block": g,
                    "n_dates": int(len(xs)), "n_obs": int(n_obs),
                    "pooled_ts_ic": ts_ic,
                    "xs_ic": float(xs.mean()) if len(xs) else np.nan,
                    "xs_rank_ic": float(rk.mean()) if len(rk) else np.nan,
                }
                block_rows.append(rec)
                by_method.setdefault(method, []).append(rec)

        for method, recs in sorted(by_method.items()):
            row = {"arm": arm, "label": label, "method": method,
                   "n_blocks": len(recs)}
            for key in ("pooled_ts_ic", "xs_ic", "xs_rank_ic"):
                v = np.array([r[key] for r in recs], dtype=float)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                row[f"{key}_mean"] = float(v.mean())
                row[f"{key}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
                row[f"{key}_hit"] = float((v > 0).mean())
            summary_rows.append(row)
            if method == "lasso":
                log.info("arm %-3s %-22s pooled=%+.4f  xs=%+.4f  rankxs=%+.4f",
                         arm, label, row.get("pooled_ts_ic_mean", np.nan),
                         row.get("xs_ic_mean", np.nan), row.get("xs_rank_ic_mean", np.nan))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(block_rows).to_csv(OUT / "xs_ic_blocks.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "xs_ic_robustness.csv", index=False)
    log.info("wrote %s and %s", OUT / "xs_ic_robustness.csv", OUT / "xs_ic_blocks.csv")


if __name__ == "__main__":
    main()
