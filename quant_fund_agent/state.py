"""LangGraph state definitions shared across all agent subgraphs."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState
from pydantic import Field

from quant_fund_agent.agents.portfolio_manager.state import (
    PortfolioManagerState,
)
from quant_fund_agent.databases import (
    FactorDatabase,
    PaperDatabase,
    PortfolioDatabase,
    StrategyDatabase,
)

__all__ = [
    "FundState",
    "FactorResearchState",
    "StrategyState",
    "PortfolioManagerState",
]


# ---------------------------------------------------------------------------
# Shared top-level state that flows through the orchestrator graph.
# Each agent subgraph receives and returns a slice of this state.
# ---------------------------------------------------------------------------

class FundState(MessagesState):
    """Root state for the entire quant-fund orchestrator graph.

    Extends LangGraph's `MessagesState` so the built-in message list is
    available for LLM interactions, while the databases provide persistent
    structured storage across agent invocations.
    """

    factor_db: FactorDatabase = Field(default_factory=FactorDatabase)
    paper_db: PaperDatabase = Field(default_factory=PaperDatabase)
    strategy_db: StrategyDatabase = Field(default_factory=StrategyDatabase)
    portfolio_db: PortfolioDatabase = Field(default_factory=PortfolioDatabase)

    current_task: str = ""
    current_agent: str = ""

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Per-agent sub-states (used inside agent subgraphs)
# ---------------------------------------------------------------------------

class FactorResearchState(MessagesState):
    """State for the factor research agent subgraph."""
    factor_db: FactorDatabase = Field(default_factory=FactorDatabase)
    paper_db: PaperDatabase = Field(default_factory=PaperDatabase)
    hypothesis: str = ""
    candidate_factor: dict[str, Any] = Field(default_factory=dict)
    backtest_result: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class StrategyState(MessagesState):
    """State for the strategy agent subgraph."""
    factor_db: FactorDatabase = Field(default_factory=FactorDatabase)
    strategy_db: StrategyDatabase = Field(default_factory=StrategyDatabase)
    candidate_strategy: dict[str, Any] = Field(default_factory=dict)
    backtest_result: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# ``PortfolioManagerState`` is defined in
# ``quant_fund_agent.agents.portfolio_manager.state`` (re-exported above) so
# the orchestrator and the PM subgraph share the same definition.
