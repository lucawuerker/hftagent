"""Point-in-time alignment of non-OHLCV (fundamental/estimate/event) data.

Fundamentals are reported with a **lag** and only become knowable on a filing
date weeks after the fiscal period ends.  Aligning a value to its fiscal-period
date would leak the future into a backtest.  This module enforces the
look-ahead discipline *in the data layer* so no factor can get it wrong:

1. **Availability stamping** — every record is stamped at its availability date:
   the vendor's filing / ``reportedDate`` when present, else
   ``fiscalDateEnding + reporting_lag`` (a conservative fixed lag).
2. **Forward-fill onto the daily panel index** — a value holds from its
   availability date until the next report, with a **staleness cap** (a value
   older than the cap → ``NaN`` → the factor is gated out for that name/date).

Because the result rides the **same daily ``DatetimeIndex`` as prices**, the
existing ``modeling/service._truncate_as_of`` slice (``index < as_of``)
automatically hides any value that wasn't yet available at the cutoff — PIT for
free, no per-factor effort.

Restatement caveat: free vendor tiers expose only the *latest* value per fiscal
period (no as-first-reported vintages), so a restated figure is what you get.
We still stamp **availability** conservatively, which prevents the dominant
look-ahead error (seeing a quarter before it was filed).
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from quant_fund_agent.data.fields import CATEGORICAL_FIELDS

# Defaults (overridable via DataSettings / env in config.py).
DEFAULT_REPORTING_LAG_DAYS = 60
DEFAULT_STALENESS_CAP_DAYS = 400  # ~5 quarters; older → treated as missing


def parse_date(value: Any) -> pd.Timestamp | None:
    """Best-effort parse of a vendor date string/number to a normalized Timestamp."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def availability_date(
    filing_date: Any,
    period_end: Any,
    *,
    reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
) -> pd.Timestamp | None:
    """When a record first becomes knowable.

    Uses the vendor's ``filing_date`` (``reportedDate`` / filing date) when it is
    present and parseable; otherwise ``period_end + reporting_lag_days``.  Returns
    ``None`` only when neither date is usable (the record is then dropped).
    """
    filed = parse_date(filing_date)
    if filed is not None:
        return filed
    end = parse_date(period_end)
    if end is not None:
        return (end + pd.Timedelta(days=int(reporting_lag_days))).normalize()
    return None


def build_record_frame(rows: list[Mapping[str, Any]]) -> pd.DataFrame | None:
    """Tidy one symbol's records into an availability-indexed frame.

    ``rows`` is a list of dicts, each carrying an ``"availability"`` Timestamp and
    a subset of canonical fields (different endpoints fill different columns).
    Returns a DataFrame indexed by availability date (ascending, de-duplicated by
    keeping the latest non-null per column), or ``None`` if empty.

    Columns keep their natural dtype — categorical labels stay objects, numerics
    stay floats — and each field forward-fills **independently** downstream, so a
    quarter's ``revenue`` and a later analyst ``epsEstimate`` become visible on
    their own availability dates.
    """
    clean = [r for r in rows if r.get("availability") is not None]
    if not clean:
        return None
    df = pd.DataFrame(clean)
    df["availability"] = pd.to_datetime(df["availability"])
    df = df.set_index("availability").sort_index()
    if df.empty:
        return None
    # Merge exact-duplicate availability dates: keep the latest non-null per column.
    if df.index.duplicated().any():
        df = df.groupby(level=0).last()
    return df.sort_index()


def _align_series(
    s: pd.Series, index: pd.DatetimeIndex, cap: pd.Timedelta | None
) -> pd.Series:
    """Forward-fill one field's observations onto ``index`` with a staleness cap.

    A value observed at availability date ``D`` is visible on every panel date
    ``>= D`` until superseded — and blanked once it is older than ``cap``.  Dates
    before the first observation stay ``NaN`` (the PIT guarantee).
    """
    is_categorical = s.dtype == object
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if s.empty:
        return pd.Series(index=index, dtype=object if is_categorical else float)

    union = s.index.union(index)
    filled = s.reindex(union).ffill().reindex(index)

    # Carry the *date of the last real observation* forward in lockstep so we can
    # measure each filled value's age and blank stale ones.
    obs_dates = pd.Series(s.index, index=s.index)
    obs_filled = obs_dates.reindex(union).ffill().reindex(index)

    out = filled
    if cap is not None:
        as_of = pd.Series(index, index=index)
        age = as_of - obs_filled  # NaT before the first observation
        out = filled.where(age.notna() & (age <= cap))
    return out


def align_to_index(
    records: Mapping[str, pd.DataFrame | None],
    index: pd.DatetimeIndex,
    *,
    staleness_cap_days: int | None = DEFAULT_STALENESS_CAP_DAYS,
) -> dict[str, pd.DataFrame]:
    """Project per-symbol availability frames onto the daily panel ``index``.

    ``records`` maps symbol → availability-indexed frame (from
    :func:`build_record_frame`).  Returns the wide panel contract
    ``{field: DataFrame(index × symbols)}`` — forward-filled from each value's
    availability date with the staleness cap applied.  Categorical fields
    (``sector``/``industry``) are object dtype; everything else is float.
    """
    index = pd.DatetimeIndex(index)
    cap = pd.Timedelta(days=int(staleness_cap_days)) if staleness_cap_days else None

    per_field: dict[str, dict[str, pd.Series]] = {}
    for sym, df in records.items():
        if df is None or df.empty:
            continue
        for col in df.columns:
            # Categorical labels (sector/industry) are near-constant and stamped
            # at a far-past sentinel — they never "go stale", so the cap is N/A.
            col_cap = None if col in CATEGORICAL_FIELDS else cap
            aligned = _align_series(df[col], index, col_cap)
            if aligned.notna().any():
                per_field.setdefault(col, {})[sym] = aligned

    out: dict[str, pd.DataFrame] = {}
    for field, series_by_sym in per_field.items():
        wide = pd.DataFrame(series_by_sym, index=index)
        # Preserve object dtype for categorical labels; coerce numerics to float.
        if field not in CATEGORICAL_FIELDS:
            wide = wide.astype(float)
        out[field] = wide
    return out
