"""Top-level orchestrator graph.

Routes a single task to the correct *real* agent pipeline:

  - ``"factor_research"``    → Factor Researcher session
  - ``"strategy"``           → Selector → Architect → Statistician → persist
  - ``"portfolio_manager"``  → Portfolio Manager rebalance

Each node delegates to :mod:`quant_fund_agent.pipeline`, so the
orchestrator, the ``run_fund.py`` script, the notebook, and the future
backtest all drive the agents through the same code path.

The ``"strategy"`` route used to invoke a stub subgraph (every node was a
``# TODO`` no-op).  It now runs the genuine Selector → Architect →
Statistician pipeline and — crucially — persists an approved strategy
into ``state.strategy_db`` so the Portfolio Manager has something to
allocate.  For a full sequential run of every stage, use ``run_fund.py``.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from quant_fund_agent import pipeline
from quant_fund_agent.state import FundState

log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_task(state: FundState) -> str:
    """Decide which agent pipeline to invoke from ``state.current_task``.

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
# Nodes — each delegates to the reusable pipeline functions
# ---------------------------------------------------------------------------

def run_factor_research(state: FundState) -> dict:
    """Invoke one Factor Researcher session (persists kept factors to disk)."""
    result = pipeline.run_research_session()
    kept = result.get("kept_factor_ids", [])
    return {"messages": [AIMessage(
        content=f"Factor research session kept {len(kept)} new factor(s): {kept}"
    )]}


def run_strategy(state: FundState) -> dict:
    """Run the real Selector → Architect → Statistician pipeline.

    An approved strategy is registered into ``state.strategy_db`` (record
    + per-bar returns) so the Portfolio Manager can deploy it.
    """
    res = pipeline.run_strategy_pipeline()
    record = pipeline.persist_approved_strategy(state.strategy_db, res)
    if record is None:
        msg = f"Strategy pipeline produced no approved strategy (rejected at {res.reject_stage})."
    else:
        sharpe = getattr(record.backtest_metrics, "sharpe_ratio", None)
        msg = (f"Approved + persisted strategy '{record.name}' "
               f"({record.id}, IS Sharpe={sharpe}).")
    log.info(msg)
    return {"messages": [AIMessage(content=msg)]}


def run_portfolio_manager(state: FundState) -> dict:
    """Run the Portfolio Manager over the current strategy book.

    Reads ``state.strategy_db`` and writes a ``PortfolioRecord`` into
    ``state.portfolio_db``.  The decision is surfaced back into the
    message stream (it used to be silently dropped).
    """
    record = pipeline.run_pm_rebalance(state.strategy_db, state.portfolio_db)
    if record is None:
        msg = "Portfolio Manager: strategy book is empty — no allocation produced."
    else:
        live = [a for a in record.allocations if a.enabled]
        msg = (f"Portfolio Manager '{record.pm_name}' allocated {len(live)} "
               f"strategies via {record.construction_method.value}.")
    log.info(msg)
    return {"messages": [AIMessage(content=msg)]}


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
