"""Walk-forward backtest harness for the quant fund.

This package drives the *agent* pipeline (`quant_fund_agent.pipeline`) through
time: weekly research / strategy / PM "meetings", then trades the resulting
book on genuinely unseen data with realistic (spread-aware) execution.

Design rule — **the backtest is separate from the agents.**  The agents never
import this package; this package talks to the agents only through the
`pipeline.py` seam.  The same agent code path therefore runs identically in a
backtest and in live trading (the long-term open-source goal).

Public API
----------
- :class:`~quant_fund_agent.simulation.config.BacktestConfig` — all knobs.
- :func:`~quant_fund_agent.simulation.simulator.run_backtest` — the entry point.
- :class:`~quant_fund_agent.simulation.results.BacktestResults` — outputs.
"""

from __future__ import annotations

from quant_fund_agent.simulation.config import BacktestConfig, ExecutionModel
from quant_fund_agent.simulation.results import BacktestResults

__all__ = ["BacktestConfig", "ExecutionModel", "BacktestResults", "run_backtest"]


def run_backtest(config: "BacktestConfig") -> "BacktestResults":
    """Run a walk-forward backtest (thin re-export to avoid import cycles)."""
    from quant_fund_agent.simulation.simulator import run_backtest as _run

    return _run(config)
