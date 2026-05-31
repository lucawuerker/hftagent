"""MCP server exposing the Portfolio Manager's deterministic analytics.

Run as a stdio server::

    python -m quant_fund_agent.mcp.portfolio_server

Covers the PM's pure quantitative toolbox — strategy screening, portfolio
construction, and ex-ante metrics.  Database reads/writes stay in the agent
graph (they mutate live in-memory DB handles shared with the committee).

All real work is delegated to :mod:`quant_fund_agent.mcp.portfolio_service`, so
the in-process fallback in ``portfolio_client.py`` computes identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.mcp import portfolio_service as svc

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.portfolio_server")

mcp = FastMCP("quant-portfolio")


@mcp.tool()
def screen_strategies(
    rows: list[dict],
    profile: dict,
    target_n: int,
    correlation: dict | None = None,
) -> str:
    """Personality hard-filters + greedy correlation-aware roster pick (JSON)."""
    return json.dumps(svc.screen_strategies(rows, profile, target_n, correlation))


@mcp.tool()
def construct_portfolio(
    method: str,
    strategy_ids: list[str],
    cov: dict | None = None,
    expected_returns: dict | None = None,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> str:
    """Run the chosen construction method; returns JSON ``{weights, method}``."""
    return json.dumps(svc.construct_portfolio(
        method=method, strategy_ids=strategy_ids, cov=cov,
        expected_returns=expected_returns, risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate, allow_short=allow_short, max_weight=max_weight,
    ))


@mcp.tool()
def expected_portfolio_metrics(
    weights: dict,
    expected_returns: dict | None = None,
    cov: dict | None = None,
) -> str:
    """Ex-ante portfolio volatility / return / Sharpe (JSON)."""
    return json.dumps(svc.expected_portfolio_metrics(weights, expected_returns, cov))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
