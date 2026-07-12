"""Causal state-frame builders for execution programs.

Executors condition on market/strategy context, but they must never compute it
themselves — the harness builds every state frame **causally** (rolling /
expanding, past-only) so input causality is guaranteed by construction (DESIGN
locked decision 7).  The executor's *own* book (positions + unrealised P&L +
running drawdown) is the one state that cannot be precomputed; it lives in
``execution.base.BookState`` and is threaded through the stepwise loop.

Fields built here (each a (T × N) frame aligned to the signal grid):

* ``vol``        — trailing volatility: rolling std of 1-bar close returns.
* ``adv``        — liquidity proxy: rolling mean dollar volume (close·volume)
                   when the panel carries ``volume``, else rolling mean |return|.
* ``spread``     — passed through from the panel when the feed carries it.
* ``drawdown``   — per-name price drawdown from the running max (≤ 0, causal).
* ``signal_age`` — bars since the composite signal last changed per name (the
                   decay-aware-holding input).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_VOL_WINDOW = 20
DEFAULT_ADV_WINDOW = 20

STATE_FIELDS = ("vol", "adv", "spread", "drawdown", "signal_age")


def _signal_age(signal: pd.DataFrame) -> pd.DataFrame:
    """Bars since each name's signal last changed (0 on the change bar, causal)."""
    arr = signal.to_numpy(dtype=float)
    changed = np.ones_like(arr, dtype=bool)
    if len(arr) > 1:
        prev, cur = arr[:-1], arr[1:]
        same = (prev == cur) | (np.isnan(prev) & np.isnan(cur))
        changed[1:] = ~same
    age = np.zeros_like(arr, dtype=float)
    for t in range(1, len(arr)):
        age[t] = np.where(changed[t], 0.0, age[t - 1] + 1.0)
    return pd.DataFrame(age, index=signal.index, columns=signal.columns)


def build_state_frames(
    panel: dict[str, Any],
    signal: pd.DataFrame,
    *,
    vol_window: int = DEFAULT_VOL_WINDOW,
    adv_window: int = DEFAULT_ADV_WINDOW,
) -> dict[str, pd.DataFrame]:
    """Build every causal state frame an executor may condition on.

    All computations are strictly trailing (rolling with past-only windows or
    running max), so a state value at bar ``t`` never reads a price after ``t``
    — the truncation-replay causality probe would catch a violation anyway,
    but the builders are causal by construction.
    """
    close = panel["close"].reindex(columns=signal.columns).reindex(index=signal.index)
    ret1 = close.pct_change()

    out: dict[str, pd.DataFrame] = {}
    out["vol"] = ret1.rolling(vol_window, min_periods=max(2, vol_window // 4)).std()

    if "volume" in panel:
        vol_f = panel["volume"].reindex(index=signal.index, columns=signal.columns)
        dollar = (close * vol_f).replace([np.inf, -np.inf], np.nan)
        out["adv"] = dollar.rolling(adv_window, min_periods=max(2, adv_window // 4)).mean()
    else:
        out["adv"] = ret1.abs().rolling(
            adv_window, min_periods=max(2, adv_window // 4)).mean()

    if "spread" in panel:
        out["spread"] = panel["spread"].reindex(index=signal.index,
                                                columns=signal.columns)

    running_max = close.cummax()
    out["drawdown"] = (close / running_max - 1.0).replace([np.inf, -np.inf], np.nan)

    out["signal_age"] = _signal_age(signal)
    return out
