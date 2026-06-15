"""Shared operator primitives for factor computation.

These mirror the formula language used in WorldQuant-style alpha
expressions.  Every factor file can import the operators it needs::

    from quant_fund_agent.factors.ops import rank, delta, ts_rank, correlation

All operators work on ``pd.DataFrame`` (index=dates, columns=tickers)
and return a ``pd.DataFrame`` of the same shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Rolling-window helper
# ---------------------------------------------------------------------------

def _rolling_windows(col: np.ndarray, n: int) -> np.ndarray:
    """All length-``n`` rolling windows of a 1-D array → shape ``(T-n+1, n)``.

    A view (no copy); callers materialise only the reductions they need.
    """
    return sliding_window_view(col, n)


def _apply_window_reduce(df: pd.DataFrame, n: int, reduce_fn) -> pd.DataFrame:
    """Apply a vectorised window reduction per column, ``min_periods=n``.

    ``reduce_fn(windows)`` receives the ``(T-n+1, n)`` window matrix for one
    column and returns a length ``T-n+1`` array.  Matches
    ``rolling(n, min_periods=n)``: the first ``n-1`` rows are NaN, and — because
    ``min_periods`` counts *non-NaN* observations — any window containing a NaN
    yields NaN (the reduction's own value there is discarded).  ``n > T`` →
    all-NaN.
    """
    values = df.to_numpy(dtype=float)
    T = values.shape[0]
    out = np.full(values.shape, np.nan, dtype=float)
    if n >= 1 and n <= T:
        for j in range(values.shape[1]):
            windows = _rolling_windows(values[:, j], n)  # (T-n+1, n)
            red = np.asarray(reduce_fn(windows), dtype=float)
            red[np.isnan(windows).any(axis=1)] = np.nan  # min_periods=n
            out[n - 1:, j] = red
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return ``data["returns"]`` or compute from close prices."""
    if "returns" in data:
        return data["returns"]
    return data["close"].pct_change(fill_method=None)


def vwap(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return ``data["vwap"]`` or approximate from HLC."""
    if "vwap" in data:
        return data["vwap"]
    if all(k in data for k in ("high", "low", "close")):
        return (data["high"] + data["low"] + data["close"]) / 3.0
    return data["close"]


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per row (timestamp)."""
    return df.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------

def delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.diff(n)


def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.shift(n)


def ts_sum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).sum()


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def stddev(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).std()


def ts_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).min()


def ts_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).max()


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position of the max value within a rolling window (1 = oldest).

    Vectorised equivalent of ``rolling(n).apply(lambda x: np.argmax(x) + 1)``
    (``np.argmax`` on the same windows, so the tie/NaN behaviour is identical).
    """
    return _apply_window_reduce(df, n, lambda w: np.argmax(w, axis=1) + 1.0)


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position of the min value within a rolling window (1 = oldest).

    Vectorised equivalent of ``rolling(n).apply(lambda x: np.argmin(x) + 1)``.
    """
    return _apply_window_reduce(df, n, lambda w: np.argmin(w, axis=1) + 1.0)


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile rank of the latest value within a rolling window.

    Vectorised equivalent of
    ``rolling(n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])``:
    the average-method percentile rank of the window's last value, dividing by
    the number of non-NaN observations (a NaN last value yields NaN).
    """
    def _rank_last(w: np.ndarray) -> np.ndarray:
        last = w[:, -1]
        valid = ~np.isnan(w)
        count = valid.sum(axis=1)                       # non-NaN per window
        less = np.nansum(w < last[:, None], axis=1)     # strictly smaller
        equal = np.nansum(w == last[:, None], axis=1)   # equal (incl. itself)
        avg_rank = less + (equal + 1.0) / 2.0           # average 1-based rank
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = avg_rank / count
        # NaN last value (or empty window) → NaN, matching pandas.
        pct = np.where(np.isnan(last) | (count == 0), np.nan, pct)
        return pct

    return _apply_window_reduce(df, n, _rank_last)


# ---------------------------------------------------------------------------
# Pairwise / math operators
# ---------------------------------------------------------------------------

def correlation(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).cov(y)


def signed_power(df: pd.DataFrame, a: float) -> pd.DataFrame:
    return np.sign(df) * (np.abs(df) ** a)


def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.replace(0, np.nan))


def abs_(df: pd.DataFrame) -> pd.DataFrame:
    return np.abs(df)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def adv(volume: pd.DataFrame, n: int) -> pd.DataFrame:
    """Average daily volume over *n* days."""
    return ts_sum(volume, n) / n


def product(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling product over *n* periods.

    Vectorised equivalent of ``rolling(n).apply(np.prod)`` (``np.prod`` over the
    same windows, so NaN propagation is identical).
    """
    return _apply_window_reduce(df, n, lambda w: np.prod(w, axis=1))


def scale(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row so that sum(abs(values)) == 1."""
    row_sum = df.abs().sum(axis=1).replace(0, np.nan)
    return df.div(row_sum, axis=0)


def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly-weighted moving average: weight[i] = i+1 for i in 0..n-1.

    Most recent observation gets the highest weight (n).
    """
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    # Vectorised equivalent of ``rolling(n).apply(lambda x: np.dot(x, weights))``:
    # the window matrix is oldest→newest, matching the weight ordering, so the
    # dot product (and its NaN propagation) is identical per window.
    return _apply_window_reduce(df, n, lambda w: w @ weights)


def power(df: pd.DataFrame, a: float) -> pd.DataFrame:
    """Element-wise exponentiation: df ** a."""
    return df ** a


def indneutralize(
    df: pd.DataFrame,
    groups: "pd.Series | pd.DataFrame",
) -> pd.DataFrame:
    """Industry-neutralize by subtracting the group mean per row.

    Args:
        df: signal DataFrame (index=dates, columns=tickers).
        groups: either a Series mapping ticker → group label (e.g. GICS
                sector), **or** the wide panel field ``data["sector"]``
                (dates × tickers of labels) the data layer now supplies.  A
                wide frame is collapsed to the latest known label per ticker
                (sector is near-static), so callers can pass ``data["sector"]``
                directly.  Tickers with no label are left un-neutralized.
    """
    if isinstance(groups, pd.DataFrame):
        # Collapse the wide (dates × tickers) label frame to a ticker→label
        # Series using each ticker's most recent non-null label.
        groups = groups.ffill().bfill().iloc[-1]
    groups = groups.dropna()

    result = df.copy()
    for _label, tickers in groups.groupby(groups).groups.items():
        cols = [c for c in tickers if c in df.columns]
        if cols:
            row_mean = df[cols].mean(axis=1)
            result[cols] = df[cols].sub(row_mean, axis=0)
    return result
