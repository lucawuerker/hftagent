"""Backtest results: fund equity curve, metrics, attribution, persistence.

A :class:`BacktestResults` accumulates per-window execution output and the log
of every meeting, then computes fund-level statistics and writes a reproducible
run folder under ``<output_root>/<run_id>/``::

    config.json              # the exact BacktestConfig used
    equity.csv               # per-bar fund return, cumulative, drawdown
    fund_metrics.json        # Sharpe / Sortino / Calmar / maxDD / costs / turnover
    attribution.csv          # per-strategy cumulative net contribution
    meetings.jsonl           # one row per meeting (cutoff, approvals, allocations)
    portfolio_snapshots.json # the PM allocation at each rebalance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.strategy_backtester import (
    _drawdown_series,
    _max_drawdown_duration,
)
from quant_fund_agent.data.frequency import (
    bars_per_day_from_index,
    trading_days_per_year_from_index,
)
from quant_fund_agent.simulation.config import BacktestConfig
from quant_fund_agent.simulation.execution import ExecutionResult

log = logging.getLogger("simulation.results")


def fund_metrics(returns: pd.Series, bars_per_day: int | None = None) -> dict[str, Any]:
    """Return-based fund statistics from a per-bar fund return series.

    ``bars_per_day=None`` infers the annualisation from the return index
    (2340 for 10-sec bars, 1 for daily), so fund Sharpe/return are correct at
    any data frequency.
    """
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {"n_bars": n}

    if bars_per_day is None:
        bars_per_day = bars_per_day_from_index(r.index)
    # Trading-days/year is inferred from the index (365 if it trades weekends, e.g.
    # crypto; 252 otherwise), so annualisation is correct across asset classes.
    bars_per_year = bars_per_day * trading_days_per_year_from_index(r.index)
    mean_bar, std_bar = float(r.mean()), float(r.std())
    ann_ret = mean_bar * bars_per_year
    ann_vol = std_bar * np.sqrt(bars_per_year) if std_bar > 0 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    ann_down = dstd * np.sqrt(bars_per_year)
    sortino = ann_ret / ann_down if ann_down > 0 else 0.0

    cum = r.cumsum()
    dd = _drawdown_series(cum)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    wins = r[r > 0]
    return {
        "n_bars": n,
        "annualised_return": round(ann_ret, 6),
        "annualised_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration_bars": _max_drawdown_duration(dd),
        "hit_rate": round(float(len(wins) / n), 4),
        "total_return": round(float(cum.iloc[-1]), 6),
    }


@dataclass
class BacktestResults:
    """Accumulates and reports the output of a walk-forward backtest."""

    config: BacktestConfig
    _return_parts: list[pd.Series] = field(default_factory=list)
    _gross_parts: list[pd.Series] = field(default_factory=list)
    _cost_parts: list[pd.Series] = field(default_factory=list)
    _turnover_parts: list[pd.Series] = field(default_factory=list)
    _netting_parts: list[pd.Series] = field(default_factory=list)
    _attribution: dict[str, list[pd.Series]] = field(default_factory=dict)
    meetings: list[dict[str, Any]] = field(default_factory=list)
    portfolio_snapshots: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # ── accumulation ────────────────────────────────────────────────────

    def add_window(self, result: ExecutionResult) -> None:
        if result.fund_returns is None or len(result.fund_returns) == 0:
            return
        self._return_parts.append(result.fund_returns)
        self._gross_parts.append(result.gross_returns)
        self._cost_parts.append(result.cost)
        self._turnover_parts.append(result.turnover)
        if result.netting_benefit is not None:
            self._netting_parts.append(result.netting_benefit)
        for sid, contrib in result.attribution.items():
            self._attribution.setdefault(sid, []).append(contrib)

    def log_meeting(self, entry: dict[str, Any]) -> None:
        self.meetings.append(entry)

    def log_portfolio(self, snapshot: dict[str, Any]) -> None:
        self.portfolio_snapshots.append(snapshot)

    # ── derived series ──────────────────────────────────────────────────

    @staticmethod
    def _concat(parts: list[pd.Series]) -> pd.Series:
        if not parts:
            return pd.Series(dtype=float)
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s

    @property
    def fund_returns(self) -> pd.Series:
        s = self._concat(self._return_parts)
        s.name = "fund_return"
        return s

    @property
    def equity_curve(self) -> pd.Series:
        return self.fund_returns.cumsum()

    @property
    def attribution(self) -> pd.DataFrame:
        cols = {sid: self._concat(parts) for sid, parts in self._attribution.items()}
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()

    # ── finalisation + persistence ──────────────────────────────────────

    def finalize(self) -> dict[str, Any]:
        r = self.fund_returns
        m = fund_metrics(r)
        bars_per_day = bars_per_day_from_index(r.index)
        cost = self._concat(self._cost_parts)
        turn = self._concat(self._turnover_parts)
        m["total_cost"] = round(float(cost.sum()), 6) if len(cost) else 0.0
        m["avg_daily_turnover"] = (
            round(float(turn.mean()) * bars_per_day, 4) if len(turn) else 0.0
        )
        m["n_meetings"] = len(self.meetings)
        m["n_strategies_deployed"] = len(self._attribution)
        m["execution_model"] = self.config.execution_model.value
        if self._netting_parts:
            nb = self._concat(self._netting_parts)
            m["netting_benefit_total"] = round(float(nb.sum()), 6)
        self.metrics = m
        return m

    def save(self) -> "Path":
        from pathlib import Path

        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2, default=str))

        r = self.fund_returns
        equity = pd.DataFrame({
            "fund_return": r,
            "cumulative_return": r.cumsum(),
            "drawdown": _drawdown_series(r.cumsum()),
        })
        equity.to_csv(out / "equity.csv", index_label="timestamp")

        (out / "fund_metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))

        attr = self.attribution
        if not attr.empty:
            attr.cumsum().to_csv(out / "attribution.csv", index_label="timestamp")

        with (out / "meetings.jsonl").open("w") as fh:
            for entry in self.meetings:
                fh.write(json.dumps(entry, default=str) + "\n")

        (out / "portfolio_snapshots.json").write_text(
            json.dumps(self.portfolio_snapshots, indent=2, default=str)
        )
        log.info("Backtest results written to %s", out)
        return out

    def summary(self) -> str:
        m = self.metrics or self.finalize()
        lines = [
            f"Backtest {self.config.run_id}  ({self.config.execution_model.value} book)",
            f"  span            : {self.config.start} → {self.config.end}",
            f"  meetings        : {m.get('n_meetings')}  |  strategies deployed: {m.get('n_strategies_deployed')}",
            f"  bars traded     : {m.get('n_bars')}",
            f"  total return    : {m.get('total_return')}",
            f"  Sharpe / Sortino: {m.get('sharpe_ratio')} / {m.get('sortino_ratio')}",
            f"  Calmar / MaxDD  : {m.get('calmar_ratio')} / {m.get('max_drawdown')}",
            f"  ann. ret / vol  : {m.get('annualised_return')} / {m.get('annualised_volatility')}",
            f"  total cost (ret): {m.get('total_cost')}  |  avg daily turnover: {m.get('avg_daily_turnover')}",
        ]
        if "netting_benefit_total" in m:
            lines.append(f"  netting benefit : {m.get('netting_benefit_total')}")
        return "\n".join(lines)
