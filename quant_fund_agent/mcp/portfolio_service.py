"""Deterministic portfolio analytics shared by the quant-portfolio MCP server
and its in-process fallback.

The Portfolio Manager's LLM steps (monitor, screen, construction-method choice)
and its database orchestration (reading the strategy universe, writing PM
statuses, persisting the portfolio record) stay in the agent graph — those
mutate the live, in-memory ``StrategyDatabase`` / ``PortfolioDatabase`` handles
that flow through the graph state and the committee, which cannot cross a
stateless MCP boundary.

What *does* move here is the PM's deterministic quantitative toolbox, computed
purely from JSON inputs:

* ``screen_strategies``         – personality hard-filters + greedy
  correlation-aware roster pick.
* ``construct_portfolio``       – the construction-method dispatcher
  (equal-weight, min-var, max-Sharpe, risk-parity, HRP …) with an
  equal-weight safety net.
* ``expected_portfolio_metrics`` – ex-ante volatility / return / Sharpe.

Covariance / correlation matrices cross the boundary as ``to_dict('split')``
payloads (``{"index", "columns", "data"}``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("portfolio.service")


# ---------------------------------------------------------------------------
# DataFrame (de)serialisation across the MCP boundary
# ---------------------------------------------------------------------------

def df_to_payload(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {"index": [], "columns": [], "data": []}
    return {
        "index": [str(i) for i in df.index],
        "columns": [str(c) for c in df.columns],
        "data": [[float(v) for v in row] for row in df.values],
    }


def df_from_payload(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not payload or not payload.get("index"):
        return pd.DataFrame()
    return pd.DataFrame(
        data=payload["data"],
        index=payload["index"],
        columns=payload["columns"],
    )


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def _passes_personality_filters(row: dict[str, Any], profile: dict[str, Any]) -> bool:
    sharpe = row.get("is_sharpe")
    if sharpe is None or sharpe < profile["min_sharpe"]:
        return False
    dd = row.get("is_max_dd")
    if dd is not None and dd < profile["max_drawdown"]:
        return False
    hit = row.get("is_hit_rate")
    min_hit = profile.get("min_hit_rate")
    if min_hit is not None and hit is not None and hit < min_hit:
        return False
    return True


def _greedy_diversified_pick(
    candidates: list[str],
    sharpes: dict[str, float],
    corr: pd.DataFrame,
    max_pairwise_corr: float,
    target_n: int,
) -> list[str]:
    ranked = sorted(candidates, key=lambda s: sharpes.get(s, -np.inf), reverse=True)
    chosen: list[str] = []
    for sid in ranked:
        if len(chosen) >= target_n:
            break
        too_correlated = False
        if not corr.empty and sid in corr.index:
            for picked in chosen:
                if picked not in corr.columns:
                    continue
                rho = corr.loc[sid, picked]
                if pd.notna(rho) and abs(float(rho)) > max_pairwise_corr:
                    too_correlated = True
                    break
        if too_correlated:
            continue
        chosen.append(sid)
    return chosen


def screen_strategies(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    target_n: int,
    correlation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply personality hard-filters then greedily pick a diversified roster.

    Returns ``{"selected_strategy_ids": [...], "eligible_strategy_ids": [...],
    "n_eligible": int}``.  ``selected`` is the greedy pick; ``eligible`` is the
    post-personality-filter universe (the caller's LLM screen reasons over it).
    """
    corr = df_from_payload(correlation)
    eligible = [r for r in rows if _passes_personality_filters(r, profile)]
    eligible_ids = [r["strategy_id"] for r in eligible]
    selected = _greedy_diversified_pick(
        eligible_ids,
        {r["strategy_id"]: (r.get("is_sharpe") or 0.0) for r in eligible},
        corr,
        profile["max_pairwise_correlation"],
        target_n,
    )
    return {
        "selected_strategy_ids": selected,
        "eligible_strategy_ids": eligible_ids,
        "n_eligible": len(eligible),
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def construct_portfolio(
    method: str,
    strategy_ids: list[str],
    cov: dict[str, Any] | None = None,
    expected_returns: dict[str, float] | None = None,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> dict[str, Any]:
    """Run the chosen construction method; fall back to equal weight on failure.

    Returns ``{"weights": {sid: w}, "method": str}`` where ``method`` is the
    method actually used (may differ from the request if the solver failed).
    """
    from quant_fund_agent.portfolio import construct_portfolio as _construct
    from quant_fund_agent.schemas import ConstructionMethod

    if not strategy_ids:
        return {"weights": {}, "method": method}

    cov_df = df_from_payload(cov)
    method_enum = ConstructionMethod(method)
    try:
        weights = _construct(
            method=method_enum,
            strategy_ids=strategy_ids,
            cov=cov_df if not cov_df.empty else None,
            expected_returns=expected_returns,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            max_weight=max_weight,
        )
        used = method_enum
    except Exception as e:  # noqa: BLE001 — always return a valid portfolio
        log.warning("%s construction failed (%s); falling back to equal weight.",
                    method, e)
        weights = _construct(
            method=ConstructionMethod.EQUAL_WEIGHT, strategy_ids=strategy_ids,
        )
        used = ConstructionMethod.EQUAL_WEIGHT
    return {"weights": weights, "method": used.value}


def expected_portfolio_metrics(
    weights: dict[str, float],
    expected_returns: dict[str, float] | None,
    cov: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Ex-ante portfolio statistics from weights + covariance (JSON in/out)."""
    from quant_fund_agent.portfolio import expected_portfolio_metrics as _metrics

    cov_df = df_from_payload(cov)
    if cov_df.empty:
        return {}
    return _metrics(weights, expected_returns=expected_returns, cov=cov_df)
