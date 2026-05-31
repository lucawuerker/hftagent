"""LangGraph state definition for the top-level orchestrator graph.

Only ``FundState`` lives here — the per-agent sub-states are defined
inside each agent package (e.g. ``agents/architect/state.py``,
``agents/portfolio_manager/state.py``) next to the graph that uses them.
``PortfolioManagerState`` is re-exported for backwards compatibility.
"""

from __future__ import annotations

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
    "PortfolioManagerState",
]


class FundState(MessagesState):
    """Root state for the entire quant-fund orchestrator graph.

    Extends LangGraph's ``MessagesState`` so the built-in message list is
    available for LLM interactions, while the databases provide persistent
    structured storage that flows between agent invocations.
    """

    factor_db: FactorDatabase = Field(default_factory=FactorDatabase)
    paper_db: PaperDatabase = Field(default_factory=PaperDatabase)
    strategy_db: StrategyDatabase = Field(default_factory=StrategyDatabase)
    portfolio_db: PortfolioDatabase = Field(default_factory=PortfolioDatabase)

    current_task: str = ""
    current_agent: str = ""

    class Config:
        arbitrary_types_allowed = True
