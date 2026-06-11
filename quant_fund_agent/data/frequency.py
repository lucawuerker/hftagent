"""Frequency-aware annualisation.

The system was built on 10-second LOBSTER bars and hardcoded the implied
annualisation everywhere: ``2340`` bars per trading day (6.5h × 360 bars/h) and
``252`` trading days per year.  On daily data (yfinance / FMP / …) that would
annualise per-*bar* statistics as if there were 2340 bars per day — inflating
Sharpe and returns by ~√2340.

These helpers infer the bar spacing from a ``DatetimeIndex`` and derive the right
annualisation.  **By construction they reproduce the legacy numbers exactly for
10-second equity data** (real LOBSTER *and* the synthetic ``freq="10s"`` test
panels), so nothing changes on the existing path — only genuinely different
frequencies (e.g. daily → 252/yr) get a different, correct factor.

Inference uses the *median bar spacing* combined with a session length, rather
than counting bars per calendar day, so it is robust to overnight gaps and to
synthetic continuous-time panels alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── legacy defaults (10-second LOBSTER equity bars) ────────────────────────
# Single source of truth, and the fallback when an index is too short to infer.
TRADING_DAYS_PER_YEAR: int = 252
SESSION_SECONDS_EQUITY: int = int(6.5 * 3600)          # 23_400
DEFAULT_BARS_PER_DAY: int = SESSION_SECONDS_EQUITY // 10  # 2340
DEFAULT_BARS_PER_YEAR: int = DEFAULT_BARS_PER_DAY * TRADING_DAYS_PER_YEAR

# A median bar spacing at/above this is treated as "daily or coarser" → 1 bar/day.
_DAILY_THRESHOLD_SECONDS: int = 12 * 3600


def trading_days_per_year(asset_class: str = "equity") -> int:
    """Trading days per year for an asset class (crypto trades every day)."""
    return 365 if str(asset_class).lower() == "crypto" else TRADING_DAYS_PER_YEAR


def _is_continuous_calendar(index) -> bool:
    """Infer whether the data trades 7 days/week (crypto) from the index itself.

    Equities, FX and LOBSTER are weekday-stamped (no Sat/Sun bars); crypto trades
    continuously, so a meaningful fraction of its bars fall on weekends.  We need a
    few distinct days to judge — a window too short to span a weekend can't be
    classified, so we conservatively return ``False`` (the equity calendar), which
    is also exactly what keeps the legacy/short-window paths unchanged.

    This keeps annualisation *data-driven* (the Phase-1 philosophy) instead of
    threading ``asset_class`` through every metric function: crypto's defining
    trait is literally present in the ``DatetimeIndex``.
    """
    idx = pd.DatetimeIndex(index)
    days = pd.DatetimeIndex(idx.normalize().unique())
    if len(days) < 7:                       # too short to see a weekend → assume 5-day
        return False
    weekend_fraction = float((days.dayofweek >= 5).mean())  # Sat=5, Sun=6
    # Equities ≈ 0 weekend days; crypto ≈ 2/7 ≈ 0.29. A 0.1 threshold separates
    # them with wide margin and tolerates the odd missing weekend bar.
    return weekend_fraction > 0.1


def trading_days_per_year_from_index(index) -> int:
    """Trading days/year inferred from the index: 365 if it trades weekends, else 252.

    Lets every existing call site (which already passes the return index) get the
    right crypto annualisation with no ``asset_class`` plumbing.
    """
    return 365 if _is_continuous_calendar(index) else TRADING_DAYS_PER_YEAR


def _median_bar_seconds(index) -> float | None:
    """Median spacing between consecutive timestamps, in seconds (``None`` if <3).

    Uses ``Timedelta.total_seconds`` so it is correct regardless of the index's
    datetime resolution (pandas 2.x panels may be ``datetime64[us]`` etc.).
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return None
    secs = idx.to_series().diff().dropna().dt.total_seconds().to_numpy()
    secs = secs[secs > 0]
    if secs.size == 0:
        return None
    return float(np.median(secs))


def bars_per_day_from_index(index, asset_class: str | None = None) -> int:
    """Infer bars-per-trading-day from a ``DatetimeIndex``.

    10-second equity bars → 2340; 1-minute → 390; daily (any spacing ≥ 12h) → 1.
    For intraday data the session length (24h crypto vs 6.5h equity) is taken from
    an explicit ``asset_class`` when given, else inferred from the index (weekend
    bars ⇒ continuous 24h market).  Falls back to :data:`DEFAULT_BARS_PER_DAY`
    when the index is too short to infer (degenerate series produce meaningless
    metrics anyway).
    """
    sec = _median_bar_seconds(index)
    if sec is None or sec <= 0:
        return DEFAULT_BARS_PER_DAY
    if sec >= _DAILY_THRESHOLD_SECONDS:
        return 1
    if asset_class is not None:
        continuous = str(asset_class).lower() == "crypto"
    else:
        continuous = _is_continuous_calendar(index)
    session = 24 * 3600 if continuous else SESSION_SECONDS_EQUITY
    return max(1, int(round(session / sec)))


def periods_per_year_from_index(index, asset_class: str | None = None) -> int:
    """Annualisation factor (periods/year) inferred from a ``DatetimeIndex``.

    With an explicit ``asset_class`` the calendar is taken from it; otherwise both
    the session length and the trading-days/year are inferred from the index, so
    weekend-trading (crypto) data annualises at 365 with no extra plumbing.
    """
    bpd = bars_per_day_from_index(index, asset_class)
    tdays = (trading_days_per_year(asset_class) if asset_class is not None
             else trading_days_per_year_from_index(index))
    return bpd * tdays
