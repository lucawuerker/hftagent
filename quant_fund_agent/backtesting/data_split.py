"""Chronological in-sample / out-of-sample split helpers.

The architect only ever sees the in-sample slice; the out-of-sample slice
is reserved for the statistician so we have a genuinely held-out test set.

We slice on row index (assuming all frames share the same DatetimeIndex),
which works correctly whether or not there are gaps in the data.
"""

from __future__ import annotations

import pandas as pd


def _cutoff_index(n_rows: int, oos_ratio: float) -> int:
    if not 0.0 < oos_ratio < 1.0:
        raise ValueError(f"oos_ratio must be in (0, 1), got {oos_ratio}")
    return int(n_rows * (1.0 - oos_ratio))


def split_panel(
    data: dict[str, pd.DataFrame],
    oos_ratio: float = 0.2,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split an OHLCV-like panel chronologically.

    Returns ``(in_sample, out_of_sample)`` with the same keys as ``data``.
    """
    first = next(iter(data.values()))
    cutoff = _cutoff_index(len(first), oos_ratio)
    in_sample = {k: df.iloc[:cutoff] for k, df in data.items()}
    out_of_sample = {k: df.iloc[cutoff:] for k, df in data.items()}
    return in_sample, out_of_sample


def split_signals(
    signals: dict[str, pd.DataFrame],
    oos_ratio: float = 0.2,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split a ``{factor_id: signal_df}`` mapping chronologically.

    Factors are computed on the *full* panel (so warm-up bars at the
    boundary do not leak into OOS), and then sliced here.
    """
    first = next(iter(signals.values()))
    cutoff = _cutoff_index(len(first), oos_ratio)
    return (
        {k: df.iloc[:cutoff] for k, df in signals.items()},
        {k: df.iloc[cutoff:] for k, df in signals.items()},
    )
