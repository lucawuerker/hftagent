"""Track 1 — single-factor Information Coefficient (per-underlying, time-series).

For each researched factor we measure how well its *value* predicts the
underlying's *own forward return* — **not** cross-sectionally.  Concretely, the
factor's value vector and the matching forward-return vector are taken per
underlying, the factor is standardised per underlying over time so the magnitudes
are comparable, the pairs are **concatenated across underlyings into one vector**,
and the IC is the Pearson correlation of that pooled vector.  With a single
ticker this reduces to exactly "one vector of factor values, one of forward
returns, compute the IC"; with many tickers it pools them (treats the universe as
one underlying), so there is no cross-section anywhere.

This is selected by ``cfg.fit_standardize == "per_underlying"`` (the default).
``cross_sectional`` keeps the legacy cross-sectional IC (``engine.backtest_factor``)
for users who still want a cross-sectional comparison.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("comparison.ic")


# ── pooled per-underlying time-series IC primitives ──────────────────────────

def _pearson(x, y) -> tuple[float | None, int]:
    """Pearson correlation of two 1-D arrays over their finite-paired rows."""
    import numpy as np

    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return None, n
    xr = np.asarray(x[mask], dtype=float)
    yr = np.asarray(y[mask], dtype=float)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = math.sqrt(float((xr ** 2).sum()) * float((yr ** 2).sum()))
    if denom == 0.0:
        return None, n
    return float((xr * yr).sum() / denom), n


def _block_ir(x, y, n_blocks: int = 20) -> float | None:
    """IC information ratio = mean/std of the IC across contiguous time blocks.

    The pooled vectors are timestamp-major, so a contiguous slice is a contiguous
    time window (pooling whatever underlyings are present in it).  A best-effort
    stability proxy — ``None`` if too few blocks carry a finite IC.
    """
    import numpy as np

    n = len(x)
    if n < n_blocks * 3:
        return None
    ics = []
    for xb, yb in zip(np.array_split(x, n_blocks), np.array_split(y, n_blocks)):
        ic, _ = _pearson(xb, yb)
        if ic is not None:
            ics.append(ic)
    if len(ics) < 3:
        return None
    sd = float(np.std(ics))
    return float(np.mean(ics) / sd) if sd > 0 else None


def _hit_rate(x, y) -> float | None:
    """Fraction of paired bars where the signal and forward return share a sign."""
    import numpy as np

    mask = np.isfinite(x) & np.isfinite(y) & (x != 0.0) & (y != 0.0)
    n = int(mask.sum())
    if n == 0:
        return None
    return float((np.sign(x[mask]) == np.sign(y[mask])).mean())


def _per_underlying_ic_row(
    prerun: str, fid: str, sig, panel, horizons, names: dict[str, str],
    factor_horizon: int | None = None,
) -> dict[str, Any]:
    """One IC row for a factor: pooled per-underlying time-series IC per horizon.

    When ``factor_horizon`` is given, the factor's IC at its *own* horizon is also
    emitted as ``ic_own`` / ``icir_own`` / ``ic_hit_own`` / ``horizon_own`` — the
    apples-to-apples headline alongside the shared ``ic_{h}`` grid.
    """
    import numpy as np

    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.standardize import per_underlying_zscore

    close = panel["close"]
    sig = sig.reindex(index=close.index, columns=close.columns)
    # Per-underlying time-series z-score makes the columns comparable so they can be
    # pooled into one Pearson correlation across names.
    x = per_underlying_zscore(sig).to_numpy(dtype=float).ravel()

    row: dict[str, Any] = {"prerun": prerun, "factor_id": fid, "name": names.get(fid, fid)}
    n_obs = 0

    def _ic_at(h: int) -> tuple[float | None, float | None, float | None, int]:
        y = forward_returns(close, horizon=h).to_numpy(dtype=float).ravel()
        ic, n = _pearson(x, y)
        return ic, _block_ir(x, y), _hit_rate(x, y), n

    for h in horizons:
        ic, icir, hit, n = _ic_at(h)
        row[f"ic_{h}"] = ic
        row[f"icir_{h}"] = icir
        row[f"ic_hit_{h}"] = hit
        row[f"ic_t_{h}"] = (
            float(ic * math.sqrt((n - 2) / (1.0 - ic ** 2)))
            if ic is not None and n > 2 and abs(ic) < 1.0 else None
        )
        n_obs = max(n_obs, n)

    if factor_horizon is not None:
        row["horizon_own"] = int(factor_horizon)
        if f"ic_{factor_horizon}" in row:  # already measured on the grid — reuse
            row["ic_own"] = row[f"ic_{factor_horizon}"]
            row["icir_own"] = row[f"icir_{factor_horizon}"]
            row["ic_hit_own"] = row[f"ic_hit_{factor_horizon}"]
        else:
            ic, icir, hit, n = _ic_at(int(factor_horizon))
            row["ic_own"], row["icir_own"], row["ic_hit_own"] = ic, icir, hit
            n_obs = max(n_obs, n)

    row["n_timestamps"] = n_obs
    return row


def evaluate_prerun_ic(
    prerun: str,
    factor_ids: list[str],
    panel: dict[str, Any],
    horizons: tuple[int, ...] = (1, 6, 60),
    names: dict[str, str] | None = None,
    *,
    cfg: Any | None = None,
    horizons_by_factor: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-factor IC rows for one prerun (one row per usable factor).

    ``cfg.fit_standardize`` selects the IC definition: ``per_underlying`` (the
    default, also used when ``cfg`` is None) → pooled per-underlying time-series
    Pearson IC; ``cross_sectional`` → legacy cross-sectional IC.

    ``horizons_by_factor`` maps factor id → its own prediction horizon; when
    given, each row also carries the factor's IC *at its own horizon*
    (``ic_own`` …) so factors with different horizons compare like-for-like.
    """
    from quant_fund_agent.modeling.service import _factor_signal

    names = names or {}
    by_factor = horizons_by_factor or {}
    mode = getattr(cfg, "fit_standardize", "per_underlying")
    rows: list[dict[str, Any]] = []
    n = len(factor_ids)
    log.info("IC: prerun '%s' — %d factors at horizons %s (mode=%s)",
             prerun, n, list(horizons), mode)
    for i, fid in enumerate(factor_ids):
        if i % 10 == 0:
            log.info("IC: prerun '%s' [%d/%d] ...", prerun, i + 1, n)
        try:
            # Reuse the signal already cached by usable_factor_ids — avoids
            # recomputing factor.calc() for every factor a second time.
            sig = _factor_signal(fid, panel)
            fh = by_factor.get(fid)
            if mode == "cross_sectional":
                rows.append(_cross_sectional_ic_row(
                    prerun, fid, sig, panel, horizons, names, fh))
            else:
                rows.append(_per_underlying_ic_row(
                    prerun, fid, sig, panel, horizons, names, fh))
        except Exception as e:  # noqa: BLE001 — one bad factor must not abort
            log.warning("IC computation failed for %s/%s: %s", prerun, fid, e)
            continue
    return rows


def _cross_sectional_ic_row(
    prerun: str, fid: str, sig, panel, horizons, names: dict[str, str],
    factor_horizon: int | None = None,
) -> dict[str, Any]:
    """Legacy cross-sectional Pearson IC row (engine.backtest_factor)."""
    from quant_fund_agent.backtesting.engine import backtest_factor

    metrics = backtest_factor(
        None, panel, horizons=horizons, signal=sig, factor_horizon=factor_horizon)
    ibh = metrics.ic_by_horizon or {}
    row: dict[str, Any] = {"prerun": prerun, "factor_id": fid, "name": names.get(fid, fid)}
    n_ts = 0
    for h in horizons:
        block = ibh.get(str(h)) or {}
        row[f"ic_{h}"] = block.get("ic")
        row[f"icir_{h}"] = block.get("ic_ir")
        row[f"ic_hit_{h}"] = block.get("ic_hit_rate")
        n_ts = max(n_ts, int(block.get("n_timestamps") or 0))
    if factor_horizon is not None:
        own = ibh.get(str(int(factor_horizon))) or {}
        row["horizon_own"] = int(factor_horizon)
        row["ic_own"] = own.get("ic")
        row["icir_own"] = own.get("ic_ir")
        row["ic_hit_own"] = own.get("ic_hit_rate")
    row["n_timestamps"] = n_ts
    return row


def summarise_ic(
    rows: list[dict[str, Any]], horizons: tuple[int, ...] = (1, 6, 60),
) -> list[dict[str, Any]]:
    """Aggregate per-factor IC rows into one summary row per prerun."""
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
        # Apples-to-apples headline: each factor at its OWN prediction horizon.
        if "ic_own" in g:
            ic_own = pd.to_numeric(g["ic_own"], errors="coerce").abs()
            icir_own = pd.to_numeric(g.get("icir_own"), errors="coerce").abs()
            s["mean_abs_ic_own"] = _f(ic_own.mean())
            s["median_abs_ic_own"] = _f(ic_own.median())
            s["mean_abs_icir_own"] = _f(icir_own.mean())
            s["pct_abs_ic_own_gt_002"] = _f((ic_own > 0.02).mean())
        # headline: the best factor by |IC| at the default (~1m) horizon
        gi = g.dropna(subset=["ic_6"]) if "ic_6" in g else g.iloc[0:0]
        if len(gi):
            best = gi.loc[pd.to_numeric(gi["ic_6"], errors="coerce").abs().idxmax()]
            s["best_factor_h6"] = best["factor_id"]
            s["best_abs_ic_h6"] = _f(abs(float(best["ic_6"])))
        out.append(s)
    return out


def _f(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None
