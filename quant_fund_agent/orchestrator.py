"""Top-level orchestrator graph.

Routes work to the correct agent subgraph based on the current task:
  - "factor_research"    → Factor Research Agent
  - "strategy"           → Strategy Agent
  - "portfolio_manager"  → Portfolio Manager Agent

The router can be replaced with an LLM-based dispatcher that reads the
conversation and autonomously decides which agent to invoke next.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.factor_research.graph import factor_research_graph
from quant_fund_agent.agents.portfolio_manager.graph import portfolio_manager_graph
from quant_fund_agent.agents.strategy.graph import strategy_graph
from quant_fund_agent.state import FundState


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_task(state: FundState) -> str:
    """Decide which agent subgraph to invoke.

    TODO: replace with LLM-based routing that examines messages and
    database state to decide autonomously.
    """
    task = state.current_task.lower()
    if "factor" in task:
        return "factor_research"
    if "strategy" in task:
        return "strategy"
    if "portfolio" in task:
        return "portfolio_manager"
    return "end"


# ---------------------------------------------------------------------------
# Wrapper nodes that invoke the compiled subgraphs
# ---------------------------------------------------------------------------

def run_factor_research(state: FundState) -> dict:
    """Invoke the factor-research subgraph."""
    result = factor_research_graph.invoke(
        {
            "messages": state.messages,
            "factor_db": state.factor_db,
            "paper_db": state.paper_db,
        }
    )
    return {"messages": result["messages"]}


def run_strategy(state: FundState) -> dict:
    """Invoke the strategy subgraph."""
    result = strategy_graph.invoke(
        {
            "messages": state.messages,
            "factor_db": state.factor_db,
            "strategy_db": state.strategy_db,
        }
    )
    return {"messages": result["messages"]}


def run_portfolio_manager(state: FundState) -> dict:
    """Invoke the portfolio-manager subgraph.

    The PM agent reads from the strategy DB (universe + per-strategy
    metrics + correlations) and writes a ``PortfolioRecord`` into the
    portfolio DB.  Both databases live on the root ``FundState``.
    """
    result = portfolio_manager_graph.invoke(
        {
            "strategy_db": state.strategy_db,
            "portfolio_db": state.portfolio_db,
        }
    )
    # We don't currently surface portfolio details back into messages —
    # downstream agents (or callers) inspect ``state.portfolio_db.latest()``.
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_orchestrator() -> StateGraph:
    graph = StateGraph(FundState)

    graph.add_node("factor_research", run_factor_research)
    graph.add_node("strategy", run_strategy)
    graph.add_node("portfolio_manager", run_portfolio_manager)

    graph.add_conditional_edges(
        START,
        route_task,
        {
            "factor_research": "factor_research",
            "strategy": "strategy",
            "portfolio_manager": "portfolio_manager",
            "end": END,
        },
    )

    graph.add_edge("factor_research", END)
    graph.add_edge("strategy", END)
    graph.add_edge("portfolio_manager", END)

    return graph


orchestrator_graph = build_orchestrator().compile()
