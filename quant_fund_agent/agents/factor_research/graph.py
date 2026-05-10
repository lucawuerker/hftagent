"""Factor Research Agent – LangGraph subgraph.

Workflow:
  1. generate_hypothesis  – read papers / brainstorm new trading ideas
  2. design_factor        – translate hypothesis into a computable factor
  3. backtest_factor      – run single-factor backtest & compute metrics
  4. evaluate_factor      – decide whether the factor meets the quality bar
     ├─ (pass) → add_to_database
     └─ (fail) → refine_hypothesis  (loops back to step 1 / 2)

Implement the body of each node function; the graph wiring is ready.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quant_fund_agent.state import FactorResearchState


# ---------------------------------------------------------------------------
# Node functions – each receives and returns FactorResearchState
# ---------------------------------------------------------------------------

def generate_hypothesis(state: FactorResearchState) -> dict:
    """Read papers, survey existing ideas, and propose a new hypothesis."""
    # TODO: use LLM to read from paper_db, propose a hypothesis
    return {"hypothesis": ""}


def design_factor(state: FactorResearchState) -> dict:
    """Translate the hypothesis into a concrete, computable factor."""
    # TODO: use LLM to produce a factor definition (formula, params, etc.)
    return {"candidate_factor": {}}


def backtest_factor(state: FactorResearchState) -> dict:
    """Run the single-factor backtest and store results."""
    # TODO: call backtesting.engine.backtest_factor, populate metrics
    return {"backtest_result": {}}


def evaluate_factor(state: FactorResearchState) -> str:
    """Decide whether the factor passes quality thresholds.

    Returns the name of the next node: 'add_to_database' or
    'generate_hypothesis' (to refine).
    """
    # TODO: check IC, Sharpe, drawdown, etc. against thresholds
    return "add_to_database"


def add_to_database(state: FactorResearchState) -> dict:
    """Persist the approved factor into the factor database."""
    # TODO: build a Factor from candidate_factor & backtest_result,
    # call factor_db.add_factor(...)
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_factor_research_graph() -> StateGraph:
    graph = StateGraph(FactorResearchState)

    graph.add_node("generate_hypothesis", generate_hypothesis)
    graph.add_node("design_factor", design_factor)
    graph.add_node("backtest_factor", backtest_factor)
    graph.add_node("evaluate_factor", evaluate_factor)
    graph.add_node("add_to_database", add_to_database)

    graph.add_edge(START, "generate_hypothesis")
    graph.add_edge("generate_hypothesis", "design_factor")
    graph.add_edge("design_factor", "backtest_factor")
    graph.add_conditional_edges(
        "backtest_factor",
        evaluate_factor,
        {
            "add_to_database": "add_to_database",
            "generate_hypothesis": "generate_hypothesis",
        },
    )
    graph.add_edge("add_to_database", END)

    return graph


factor_research_graph = build_factor_research_graph().compile()
