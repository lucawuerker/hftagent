"""The joint objective J and (J2) the bandit context vector.

``J`` = the deflated net-of-cost VAL Sharpe of the full pipeline the SOTA state
represents: curated book → IS-fit combined model → SOTA executor (baseline
until one exists) → cost layer.  Computed by the deterministic MCP service
(:func:`quant_fund_agent.mcp.research_service.score_joint_state`), deflated at
the ledger's current joint look count.  Scored at every block boundary so
consecutive values are like-for-like; ΔJ across a block is the scheduler's
reward — and every scoring call itself bills the ledger's look count.
"""

from __future__ import annotations

import logging
from typing import Any

from quant_fund_agent.joint_evolution.ledger import TrialsLedger
from quant_fund_agent.joint_evolution.state import SOTAState

log = logging.getLogger("joint_evolution.objective")


def score_joint(sota: SOTAState, ledger: TrialsLedger, *,
                target_horizon: int = 6,
                is_frac: float = 0.6, val_frac: float = 0.2,
                cutoff_date: str | None = None,
                data_dir: str = "ticker_data", n_tickers: int | None = 15,
                fields: list[str] | None = None,
                model: str = "ridge", cost_rate: float = 5e-4,
                ) -> dict[str, Any]:
    """Score the current SOTA state → {"ok", "J", diagnostics}.  Bills one look.

    Before the first factor block there is no book — returns
    ``{"ok": False, "J": None}`` without billing (nothing was looked at).
    """
    from quant_fund_agent.mcp import research_client

    if not sota.book:
        return {"ok": False, "J": None, "reason": "no book yet"}
    out = research_client.score_joint_state(
        sota.book, sota.sota_executor,
        n_joint_looks=max(1, ledger.joint_count() + 1),  # count this look too
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        cutoff_date=cutoff_date, data_dir=data_dir, n_tickers=n_tickers,
        fields=fields, model=model, cost_rate=cost_rate)
    ledger.bill_look(1, source="joint_objective")
    if not out.get("ok"):
        log.warning("score_joint_state failed: %s", out.get("error"))
    return out
