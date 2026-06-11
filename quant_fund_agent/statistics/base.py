"""Base class for statistical tests.

Every test is a small self-contained class that takes a ``StatTestContext``
(a bundle of everything the statistician knows about the candidate strategy)
and returns a ``StatTestResult``.

The tests are "block-style" — the statistician agent picks tests by ID,
instantiates them, and collects their outputs.  Later on, researcher
agents can add new test classes without the statistician code changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_fund_agent.agents.architect.state import StrategySpec, TrialRecord
from quant_fund_agent.data.frequency import (
    DEFAULT_BARS_PER_DAY,
    TRADING_DAYS_PER_YEAR,
)
from quant_fund_agent.statistics.schemas import StatTestResult


@dataclass
class StatTestContext:
    """Everything a statistical test might need.

    Not all tests use every field — each test picks what it needs.
    """

    # ── strategy under test ──
    strategy_spec: StrategySpec
    hypothesis: str = ""
    initial_reasoning: str = ""
    final_reasoning: str = ""

    # ── in-sample backtest result (already run by the architect) ──
    is_portfolio_returns: pd.Series | None = None
    is_equity_curve: pd.Series | None = None
    is_metrics: dict[str, Any] = field(default_factory=dict)

    # ── held-out data for re-running the strategy OOS ──
    data_full: dict[str, pd.DataFrame] | None = None
    factor_signals_full: dict[str, pd.DataFrame] | None = None
    oos_split_ratio: float = 0.2

    # ── trial history (for multiple-testing deflation) ──
    trial_history: list[TrialRecord] = field(default_factory=list)

    # ── bars / annualisation config ──
    # Both inferred from the panel by the context builder; these are the fallbacks.
    bars_per_day: int = DEFAULT_BARS_PER_DAY
    # Trading days/year: 252 for equities, 365 for 24/7 markets (crypto).
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR


class BaseStatTest(ABC):
    """Abstract base class for a statistical test.

    Subclasses set the class-level metadata and implement ``run()``.
    """

    test_id: str = ""
    name: str = ""
    description: str = ""
    is_mandatory: bool = False  # must be run for every strategy

    @abstractmethod
    def run(self, ctx: StatTestContext) -> StatTestResult:
        """Execute the test and return a standardised result."""
        ...
