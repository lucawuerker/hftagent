"""Panel + factor-signal access for the backtest, sliced per trading window.

The heavy panel and per-factor signals are loaded/computed **once on the full
history** and sliced per window.  We deliberately reuse the *same* module-level
caches the agents use (``quant_fund_agent.modeling.service``) so that:

1. the universe is identical to what the Architect/Statistician designed on
   (same ``ARCHITECT_N_TICKERS`` cap, same loader); and
2. factor signals are computed on the whole history and only *then* sliced — so
   warm-up bars at the start of a window are correct and nothing leaks across
   the cutoff (factors are causal, so values inside a window already reflect the
   prior history).
"""

from __future__ import annotations

import pandas as pd


class SignalCache:
    """Lazily computes factor signals on the full panel and slices windows."""

    def __init__(self) -> None:
        from quant_fund_agent.factors import discover_factors
        from quant_fund_agent.modeling.service import _load_panel

        discover_factors()
        self._panel: dict[str, pd.DataFrame] = _load_panel()  # full, cached

    @property
    def panel(self) -> dict[str, pd.DataFrame]:
        return self._panel

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self._panel.values())).index

    def signal(self, factor_id: str) -> pd.DataFrame:
        """Full-history signal for one factor (cached)."""
        from quant_fund_agent.modeling.service import _factor_signal

        return _factor_signal(factor_id, self._panel)

    # ── window slicing ──────────────────────────────────────────────────

    @staticmethod
    def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        mask = (df.index >= start) & (df.index < end)
        return df.loc[mask]

    def panel_window(
        self, start: pd.Timestamp, end: pd.Timestamp,
        fields: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        keys = fields if fields is not None else list(self._panel.keys())
        return {
            k: self._slice(self._panel[k], start, end)
            for k in keys if k in self._panel
        }

    def signals_window(
        self, factor_ids: list[str], start: pd.Timestamp, end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        return {
            fid: self._slice(self.signal(fid), start, end)
            for fid in factor_ids
        }
