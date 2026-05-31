"""Entry point – dispatch a single task through the orchestrator graph.

For a full sequential run of every stage (research → strategy → PM),
use ``run_fund.py`` instead; this module exercises the orchestrator's
single-task router.
"""

from __future__ import annotations

from dotenv import load_dotenv

from quant_fund_agent.databases import (
    FactorDatabase,
    PaperDatabase,
    PortfolioDatabase,
    StrategyDatabase,
)
from quant_fund_agent.orchestrator import orchestrator_graph
from quant_fund_agent.pipeline import STRATEGY_DB_PATH

load_dotenv()


def run(task: str = "portfolio_manager") -> dict:
    """Invoke the orchestrator with a given task type.

    Args:
        task: one of "factor_research", "strategy", "portfolio_manager".
    """
    strategy_db = StrategyDatabase()
    strategy_db.load_from_json(STRATEGY_DB_PATH)

    initial_state = {
        "messages": [],
        "factor_db": FactorDatabase(),
        "paper_db": PaperDatabase(),
        "strategy_db": strategy_db,
        "portfolio_db": PortfolioDatabase(),
        "current_task": task,
        "current_agent": "",
    }
    return orchestrator_graph.invoke(initial_state)


if __name__ == "__main__":
    result = run("portfolio_manager")
    for m in result.get("messages", []):
        print(m.content)
