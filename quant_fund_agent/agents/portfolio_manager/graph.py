"""Portfolio Manager Agent – LangGraph subgraph.

Workflow:
  1. review_strategies    – survey all approved / live strategies
  2. assess_correlations  – compute pairwise strategy correlations
  3. allocate_portfolio   – decide which strategies to run and at what weight
  4. flag_underperformers – identify strategies that need rework
     └─ output list of strategy IDs to send back to the strategy agent

Implement the body of each node function; the graph wiring is ready.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quant_fund_agent.state import PortfolioManagerState


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def review_strategies(state: PortfolioManagerState) -> dict:
    """Pull all strategies and their recent performance."""
    # TODO: read from strategy_db, gather live PnL / risk metrics
    return {}


def assess_correlations(state: PortfolioManagerState) -> dict:
    """Compute or retrieve pairwise correlations between live strategies."""
    # TODO: build correlation matrix of strategy returns
    return {}


def allocate_portfolio(state: PortfolioManagerState) -> dict:
    """Decide which strategies to enable and their capital allocation."""
    # TODO: optimisation (e.g. risk-parity, mean-variance) or LLM reasoning
    return {"portfolio_allocation": {}, "live_strategy_ids": []}


def flag_underperformers(state: PortfolioManagerState) -> dict:
    """Identify strategies whose performance has degraded and may need
    retraining / rework by the strategy agent."""
    # TODO: compare recent metrics to thresholds, build review notes
    return {"review_notes": ""}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_portfolio_manager_graph() -> StateGraph:
    graph = StateGraph(PortfolioManagerState)

    graph.add_node("review_strategies", review_strategies)
    graph.add_node("assess_correlations", assess_correlations)
    graph.add_node("allocate_portfolio", allocate_portfolio)
    graph.add_node("flag_underperformers", flag_underperformers)

    graph.add_edge(START, "review_strategies")
    graph.add_edge("review_strategies", "assess_correlations")
    graph.add_edge("assess_correlations", "allocate_portfolio")
    graph.add_edge("allocate_portfolio", "flag_underperformers")
    graph.add_edge("flag_underperformers", END)

    return graph


portfolio_manager_graph = build_portfolio_manager_graph().compile()
