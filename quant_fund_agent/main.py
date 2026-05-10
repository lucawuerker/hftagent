"""Entry point – run the quant-fund orchestrator interactively."""

from __future__ import annotations

from quant_fund_agent.databases import FactorDatabase, PaperDatabase, StrategyDatabase
from quant_fund_agent.orchestrator import orchestrator_graph


def run(task: str = "factor_research") -> dict:
    """Invoke the orchestrator with a given task type.

    Args:
        task: one of "factor_research", "strategy", "portfolio_manager".
    """
    initial_state = {
        "messages": [],
        "factor_db": FactorDatabase(),
        "paper_db": PaperDatabase(),
        "strategy_db": StrategyDatabase(),
        "current_task": task,
        "current_agent": "",
    }
    result = orchestrator_graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    result = run("factor_research")
    print(result)
