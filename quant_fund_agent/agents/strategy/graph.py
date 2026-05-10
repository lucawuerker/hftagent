"""Strategy Agent – LangGraph subgraph.

Workflow:
  1. select_factors        – pick promising factors from the factor DB
  2. design_strategy       – combine factors (ML, regression, rule-based, …)
  3. backtest_strategy     – run multi-factor strategy backtest
  4. evaluate_strategy     – check if strategy meets quality / correlation bar
     ├─ (pass) → add_to_database
     └─ (fail) → refine_strategy  (loops back to step 2)

Implement the body of each node function; the graph wiring is ready.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quant_fund_agent.state import StrategyState


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def select_factors(state: StrategyState) -> dict:
    """Browse factor DB and choose a set of factors to combine."""
    # TODO: query factor_db, use LLM or heuristic to pick factors
    return {}


def design_strategy(state: StrategyState) -> dict:
    """Combine selected factors into a strategy definition."""
    # TODO: LLM-driven or algorithmic combination (regression, ML, etc.)
    return {"candidate_strategy": {}}


def backtest_strategy(state: StrategyState) -> dict:
    """Run a full strategy backtest."""
    # TODO: call backtesting.engine.backtest_strategy
    return {"backtest_result": {}}


def evaluate_strategy(state: StrategyState) -> str:
    """Accept or reject the strategy based on metrics and correlation
    with existing strategies.

    Returns the next node name.
    """
    # TODO: check Sharpe, correlation with live book, drawdown, etc.
    return "add_to_database"


def add_to_database(state: StrategyState) -> dict:
    """Persist the approved strategy into the strategy database."""
    # TODO: build a Strategy, call strategy_db.add_strategy(...)
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_strategy_graph() -> StateGraph:
    graph = StateGraph(StrategyState)

    graph.add_node("select_factors", select_factors)
    graph.add_node("design_strategy", design_strategy)
    graph.add_node("backtest_strategy", backtest_strategy)
    graph.add_node("evaluate_strategy", evaluate_strategy)
    graph.add_node("add_to_database", add_to_database)

    graph.add_edge(START, "select_factors")
    graph.add_edge("select_factors", "design_strategy")
    graph.add_edge("design_strategy", "backtest_strategy")
    graph.add_conditional_edges(
        "backtest_strategy",
        evaluate_strategy,
        {
            "add_to_database": "add_to_database",
            "design_strategy": "design_strategy",
        },
    )
    graph.add_edge("add_to_database", END)

    return graph


strategy_graph = build_strategy_graph().compile()
