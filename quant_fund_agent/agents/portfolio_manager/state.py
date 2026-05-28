"""State for the Portfolio Manager Agent.

This is the *full* state struct the PM agent reads & writes.  It is
deliberately wider than the trivial PortfolioManagerState that lived in
the top-level ``state.py`` (which we keep as a thin alias for backwards
compatibility).

The PM is configured at the top of the state:

* ``mode``           – ``SELECTOR`` (passive: PM picks strategies, the
                       construction method is fixed in advance) or
                       ``ACTIVE`` (PM also chooses the construction
                       method and may override weights through the LLM).
* ``personality``    – which pre-baked risk profile to use.  ``CUSTOM``
                       means the caller supplies its own ``profile``.
* ``profile``        – the ``PersonalityProfile`` actually in effect.
                       Resolved from ``personality`` if not given.
* ``pm_name``        – distinguishes side-by-side PMs in the audit log.

Outputs land in:

* ``selected_strategy_ids``  – strategies that survived screening
* ``flagged_strategy_ids``   – strategies sent back to the research team
* ``retired_strategy_ids``   – strategies killed
* ``allocations``            – the actual portfolio (list[PortfolioAllocation])
* ``expected_metrics``       – ex-ante portfolio Sharpe / vol / etc.
* ``pm_rationale``           – natural-language summary
* ``portfolio_record``       – the persisted PortfolioRecord
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
from quant_fund_agent.portfolio.personalities import PersonalityProfile
from quant_fund_agent.schemas import (
    ConstructionMethod,
    PMMode,
    PMPersonality,
    PortfolioAllocation,
    PortfolioRecord,
)


class PortfolioManagerState(BaseModel):
    """State flowing through the PM agent's LangGraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── config ──────────────────────────────────────────────────────
    pm_name: str = "default_pm"
    mode: PMMode = PMMode.SELECTOR
    personality: PMPersonality = PMPersonality.BALANCED
    profile: PersonalityProfile | None = None  # resolved if None

    # If set, this overrides ``profile.default_construction_method`` in
    # SELECTOR mode.  In ACTIVE mode the LLM picks this itself.
    construction_method_override: ConstructionMethod | None = None

    # How many strategies the PM wants to deploy.  Defaults to the
    # personality's ``target_n_strategies`` if not given.
    target_n_strategies: int | None = None

    # Annualisation factor for ex-ante metrics (bars_per_year for
    # intraday strategies).  Default 1.0 leaves everything in raw units.
    annualisation_factor: float = 1.0

    # When True the PM produces a PortfolioRecord *without* mutating
    # ``strategy_db`` (pm_status updates) or ``portfolio_db`` (record
    # persistence).  Used by the committee so that several PMs can
    # propose in parallel and the final mutation only happens once,
    # against the consensus.  Default False — single-PM behaviour is
    # unchanged.
    propose_only: bool = False

    # ── databases (passed in; not persisted by the agent) ───────────
    strategy_db: StrategyDatabase = Field(default_factory=StrategyDatabase)
    portfolio_db: PortfolioDatabase = Field(default_factory=PortfolioDatabase)

    # ── working memory built up by the nodes ────────────────────────
    universe_summary: list[dict[str, Any]] = Field(default_factory=list)
    correlation_matrix: pd.DataFrame = Field(default_factory=pd.DataFrame)
    covariance_matrix: pd.DataFrame = Field(default_factory=pd.DataFrame)

    monitor_actions: list[dict[str, Any]] = Field(default_factory=list)

    selected_strategy_ids: list[str] = Field(default_factory=list)
    flagged_strategy_ids: list[str] = Field(default_factory=list)
    retired_strategy_ids: list[str] = Field(default_factory=list)
    screen_rationale: str = ""

    chosen_method: ConstructionMethod = ConstructionMethod.EQUAL_WEIGHT
    raw_weights: dict[str, float] = Field(default_factory=dict)

    # ── outputs ─────────────────────────────────────────────────────
    allocations: list[PortfolioAllocation] = Field(default_factory=list)
    expected_metrics: dict[str, float] = Field(default_factory=dict)
    pm_rationale: str = ""
    portfolio_record: PortfolioRecord | None = None
