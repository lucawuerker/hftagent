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
    """Position of the max value within a rolling window (1 = oldest)."""
    return df.rolling(n, min_periods=n).apply(
        lambda x: float(np.argmax(x) + 1), raw=True,
    )


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position of the min value within a rolling window (1 = oldest)."""
    return df.rolling(n, min_periods=n).apply(
        lambda x: float(np.argmin(x) + 1), raw=True,
    )


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile rank of the latest value within a rolling window."""
    return df.rolling(n, min_periods=n).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]),
        raw=False,
    )


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
    """Rolling product over *n* periods."""
    return df.rolling(n, min_periods=n).apply(np.prod, raw=True)


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

    return df.rolling(n, min_periods=n).apply(
        lambda x: np.dot(x, weights), raw=True,
    )


def power(df: pd.DataFrame, a: float) -> pd.DataFrame:
    """Element-wise exponentiation: df ** a."""
    return df ** a


def indneutralize(
    df: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    """Industry-neutralize by subtracting the group mean per row.

    Args:
        df: signal DataFrame (index=dates, columns=tickers).
        groups: Series mapping ticker → group label (e.g. GICS
                sub-industry).  Tickers not in ``groups`` are left
                un-neutralized.
    """
    result = df.copy()
    for _label, tickers in groups.groupby(groups).groups.items():
        cols = [c for c in tickers if c in df.columns]
        if cols:
            row_mean = df[cols].mean(axis=1)
            result[cols] = df[cols].sub(row_mean, axis=0)
    return result
