"""Track 1 — single-factor Information Coefficient.

Recompute every researched factor's cross-sectional rank-IC on the *shared*
current panel at several horizons and compare the per-model distributions.  This
mirrors the user's ``prelim_files/tests.ipynb`` (the step-by-step rank-IC and the
``ic_by_horizon`` overview); we reuse the canonical engine routine so the numbers
are identical to the research-time IC backtest.

Recomputing here (rather than reading each prerun's stored IC) guarantees an
apples-to-apples comparison: every factor is scored on the same panel/universe,
so differences reflect the factor — not the data a given prerun happened to use.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("comparison.ic")


def evaluate_prerun_ic(
    prerun: str,
    factor_ids: list[str],
    panel: dict[str, Any],
    horizons: tuple[int, ...] = (1, 6, 60),
    names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-factor IC rows for one prerun (one row per usable factor)."""
    from quant_fund_agent.backtesting.engine import backtest_factor
    from quant_fund_agent.modeling.service import _factor_signal

    names = names or {}
    rows: list[dict[str, Any]] = []
    n = len(factor_ids)
    log.info("IC: prerun '%s' — %d factors at horizons %s", prerun, n, list(horizons))
    for i, fid in enumerate(factor_ids):
        if i % 10 == 0:
            log.info("IC: prerun '%s' [%d/%d] ...", prerun, i + 1, n)
        try:
            # Reuse the signal already cached by usable_factor_ids — avoids
            # recomputing factor.calc() for every factor a second time.
            sig = _factor_signal(fid, panel)
            metrics = backtest_factor(None, panel, horizons=horizons, signal=sig)
        except Exception as e:  # noqa: BLE001 — one bad factor must not abort
            log.warning("IC backtest failed for %s/%s: %s", prerun, fid, e)
            continue
        ibh = metrics.ic_by_horizon or {}
        row: dict[str, Any] = {
            "prerun": prerun,
            "factor_id": fid,
            "name": names.get(fid, fid),
        }
        n_ts = 0
        for h in horizons:
            block = ibh.get(str(h)) or {}
            row[f"ic_{h}"] = block.get("ic")
            row[f"icir_{h}"] = block.get("ic_ir")
            row[f"ic_hit_{h}"] = block.get("ic_hit_rate")
            n_ts = max(n_ts, int(block.get("n_timestamps") or 0))
        row["n_timestamps"] = n_ts
        rows.append(row)
    return rows


def summarise_ic(
    rows: list[dict[str, Any]], horizons: tuple[int, ...] = (1, 6, 60),
) -> list[dict[str, Any]]:
    """Aggregate per-factor IC rows into one summary row per prerun."""
    import numpy as np
    import pandas as pd

    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for prerun, g in df.groupby("prerun"):
        s: dict[str, Any] = {"prerun": prerun, "n_factors": int(len(g))}
        for h in horizons:
            ic = pd.to_numeric(g.get(f"ic_{h}"), errors="coerce").abs()
            icir = pd.to_numeric(g.get(f"icir_{h}"), errors="coerce").abs()
            s[f"mean_abs_ic_{h}"] = _f(ic.mean())
            s[f"median_abs_ic_{h}"] = _f(ic.median())
            s[f"mean_abs_icir_{h}"] = _f(icir.mean())
            s[f"pct_abs_ic_gt_002_{h}"] = _f((ic > 0.02).mean())
        # headline: the best factor by |IC| at the default (~1m) horizon
        gi = g.dropna(subset=["ic_6"]) if "ic_6" in g else g.iloc[0:0]
        if len(gi):
            best = gi.loc[pd.to_numeric(gi["ic_6"], errors="coerce").abs().idxmax()]
            s["best_factor_h6"] = best["factor_id"]
            s["best_abs_ic_h6"] = _f(abs(float(best["ic_6"])))
        out.append(s)
    return out


def _f(v: Any) -> float | None:
    import math

    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None
