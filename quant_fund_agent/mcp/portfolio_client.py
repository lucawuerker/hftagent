"""Synchronous client facade for the quant-portfolio MCP server.

Covariance / correlation matrices are passed as ``pandas`` DataFrames here and
serialised to JSON payloads for the MCP boundary.

Set ``PORTFOLIO_USE_MCP=0`` (or the global ``QF_USE_MCP=0``) to call the
:mod:`quant_fund_agent.mcp.portfolio_service` functions in-process instead.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_fund_agent.mcp._bridge import MCPBridge, use_mcp
from quant_fund_agent.mcp.portfolio_service import df_to_payload

_SERVER_MODULE = "quant_fund_agent.mcp.portfolio_server"


def _use_mcp() -> bool:
    return use_mcp("PORTFOLIO_USE_MCP")


def _bridge() -> MCPBridge:
    return MCPBridge.instance(_SERVER_MODULE)


def screen_strategies(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    target_n: int,
    correlation: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Personality filter + greedy diversified pick."""
    payload = df_to_payload(correlation)
    if not _use_mcp():
        from quant_fund_agent.mcp import portfolio_service as svc
        return svc.screen_strategies(rows, profile, target_n, payload)
    return _bridge().call("screen_strategies", {
        "rows": rows, "profile": profile, "target_n": target_n,
        "correlation": payload,
    })


def construct_portfolio(
    method: str,
    strategy_ids: list[str],
    cov: pd.DataFrame | None = None,
    expected_returns: dict[str, float] | None = None,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> dict[str, Any]:
    """Run a construction method; returns ``{"weights", "method"}``."""
    args: dict[str, Any] = dict(
        method=method, strategy_ids=strategy_ids, cov=df_to_payload(cov),
        expected_returns=expected_returns, risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate, allow_short=allow_short, max_weight=max_weight,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import portfolio_service as svc
        return svc.construct_portfolio(**args)
    return _bridge().call("construct_portfolio",
                          {k: v for k, v in args.items() if v is not None})


def expected_portfolio_metrics(
    weights: dict[str, float],
    expected_returns: dict[str, float] | None,
    cov: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Ex-ante portfolio metrics from weights + covariance."""
    payload = df_to_payload(cov)
    if not _use_mcp():
        from quant_fund_agent.mcp import portfolio_service as svc
        return svc.expected_portfolio_metrics(weights, expected_returns, payload)
    args = {"weights": weights, "cov": payload}
    if expected_returns is not None:
        args["expected_returns"] = expected_returns
    return _bridge().call("expected_portfolio_metrics", args)
