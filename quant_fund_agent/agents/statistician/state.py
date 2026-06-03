"""State for the Statistician Agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quant_fund_agent.agents.architect.state import StrategySpec, TrialRecord
from quant_fund_agent.statistics.schemas import StatTestResult


class StatisticianState(BaseModel):
    """State flowing through the statistician agent graph."""

    # ── inputs from the architect ──
    hypothesis: str = ""
    strategy_spec: StrategySpec = Field(default_factory=StrategySpec)
    trial_history: list[TrialRecord] = Field(default_factory=list)
    backtest_metrics: dict[str, Any] = Field(default_factory=dict)

    # Original (first-iteration) and final (approved-iteration) architect rationale
    initial_reasoning: str = ""
    final_reasoning: str = ""

    # OOS split the architect used (statistician reuses the same split)
    oos_split_ratio: float = 0.2

    # Walk-forward cutoff (backtest only): ISO date string limiting visible
    # data so the OOS test never sees post-cutoff bars.  None → full history.
    as_of: str | None = None

    # ── config ──
    bars_per_day: int = 2340
    # Test IDs that must always be run.  If empty the statistician uses
    # whichever tests are registered as ``is_mandatory=True``.
    mandatory_test_ids: list[str] = Field(default_factory=list)
    # Test IDs to force-run on top of mandatory (e.g. supplied by caller).
    forced_optional_test_ids: list[str] = Field(default_factory=list)

    # ── outputs ──
    test_results: list[StatTestResult] = Field(default_factory=list)
    risk_assessment: str = ""
    selected_optional_test_ids: list[str] = Field(default_factory=list)
    final_decision: str = ""          # "accept" or "reject"
    final_reasoning_statistician: str = ""

    class Config:
        arbitrary_types_allowed = True
