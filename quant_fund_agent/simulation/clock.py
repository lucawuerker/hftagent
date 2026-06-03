"""Trading clock + meeting schedule for the walk-forward backtest.

Given the data panel's ``DatetimeIndex`` and a :class:`BacktestConfig`, the
:class:`TradingClock` produces a deterministic sequence of :class:`Period`s.
Each period carries:

- ``cutoff`` — the as-of timestamp.  Agents may only see data **strictly before**
  it (no look-ahead); the fund **trades** the window at/after it.
- ``live_start`` / ``live_end`` — the (unseen) trading window ``[cutoff, next)``.
- ``research_due`` / ``strategy_due`` / ``pm_due`` — which meetings fire.
- ``is_first_strategy_meeting`` — to size the bootstrap (``initial_strategies``).

The first cutoff is the first grid point with at least ``warmup`` of history, so
the agents always have enough data to research / fit on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_fund_agent.simulation.config import (
    BacktestConfig,
    cadence_steps,
    offset_to_days,
)


@dataclass
class Period:
    """One grid step of the walk-forward loop."""

    index: int                 # 0-based, counted from the first cutoff
    cutoff: pd.Timestamp       # as-of: agents see data strictly before this
    live_start: pd.Timestamp   # first tradeable timestamp (== cutoff)
    live_end: pd.Timestamp     # exclusive end of the trading window
    research_due: bool
    strategy_due: bool
    pm_due: bool
    is_first_strategy_meeting: bool

    @property
    def cutoff_date(self) -> date:
        return self.cutoff.date()

    @property
    def week_tag(self) -> str:
        """Stable session id for the researcher (e.g. ``'2019-W03'``)."""
        iso = self.cutoff.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"


class TradingClock:
    """Builds the weekly meeting/rebalance schedule over the data span."""

    def __init__(self, panel_index: pd.DatetimeIndex, config: BacktestConfig) -> None:
        self.config = config
        self._index = pd.DatetimeIndex(panel_index).sort_values()
        self._periods = self._build_periods()

    # ── schedule construction ─────────────────────────────────────────

    def _build_periods(self) -> list[Period]:
        cfg = self.config
        if len(self._index) == 0:
            return []

        start = pd.Timestamp(cfg.start_date)
        # Clamp the end to what the data actually covers.
        end = min(pd.Timestamp(cfg.end_date), self._index.max().normalize() + pd.Timedelta(days=1))

        step_days = offset_to_days(cfg.grid_freq)
        grid = pd.date_range(start=start, end=end, freq=f"{step_days}D")
        if len(grid) == 0:
            return []

        warmup_days = offset_to_days(cfg.warmup)
        first_cut = start + pd.Timedelta(days=warmup_days)
        cutoffs = [g for g in grid if g >= first_cut]
        if not cutoffs:
            return []

        research_k = cadence_steps(cfg.research_every, cfg.grid_freq)
        strategy_k = cadence_steps(cfg.strategy_every, cfg.grid_freq)
        pm_k = cadence_steps(cfg.pm_rebalance_every, cfg.grid_freq)

        periods: list[Period] = []
        seen_strategy_meeting = False
        for i, cut in enumerate(cutoffs):
            live_start = cut
            live_end = cutoffs[i + 1] if i + 1 < len(cutoffs) else end
            # Skip windows that contain no bars (e.g. trailing past data end).
            if not self._has_bars(live_start, live_end):
                continue
            # The agents also need data *before* the cutoff to work with.
            if not self._has_bars(self._index.min(), cut):
                continue

            strategy_due = (i % strategy_k == 0) and (
                self._has_bars(self._index.min(), cut)
            )
            is_first_strategy = strategy_due and not seen_strategy_meeting
            if strategy_due:
                seen_strategy_meeting = True

            periods.append(Period(
                index=i,
                cutoff=cut,
                live_start=live_start,
                live_end=live_end,
                research_due=cfg.run_research and (i % research_k == 0),
                strategy_due=strategy_due,
                pm_due=(i % pm_k == 0),
                is_first_strategy_meeting=is_first_strategy,
            ))
        return periods

    def _has_bars(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """True if at least one bar falls in ``[start, end)``."""
        lo = self._index.searchsorted(start, side="left")
        hi = self._index.searchsorted(end, side="left")
        return bool(hi > lo)

    # ── public API ─────────────────────────────────────────────────────

    def periods(self) -> list[Period]:
        return list(self._periods)

    def __len__(self) -> int:
        return len(self._periods)

    def __iter__(self):
        return iter(self._periods)
