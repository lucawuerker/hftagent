"""Synchronous client facade for the quant-statistics MCP server.

Set ``STATISTICS_USE_MCP=0`` (or the global ``QF_USE_MCP=0``) to call the
:mod:`quant_fund_agent.mcp.statistics_service` functions in-process instead.
"""

from __future__ import annotations

from typing import Any

from quant_fund_agent.mcp._bridge import MCPBridge, use_mcp

_SERVER_MODULE = "quant_fund_agent.mcp.statistics_server"


def _use_mcp() -> bool:
    return use_mcp("STATISTICS_USE_MCP")


def _bridge() -> MCPBridge:
    return MCPBridge.instance(_SERVER_MODULE)


def list_tests() -> dict[str, Any]:
    """Return registered tests grouped by mandatory/optional."""
    if not _use_mcp():
        from quant_fund_agent.mcp import statistics_service as svc
        return svc.list_tests()
    return _bridge().call("list_tests", {})


def run_tests(
    test_ids: list[str],
    spec: dict[str, Any],
    is_metrics: dict[str, Any] | None = None,
    trial_history: list[dict[str, Any]] | None = None,
    hypothesis: str = "",
    oos_split_ratio: float = 0.2,
    bars_per_day: int = 2340,
) -> list[dict[str, Any]]:
    """Run the requested statistical tests; returns StatTestResult dumps."""
    args: dict[str, Any] = dict(
        test_ids=test_ids,
        spec=spec,
        is_metrics=is_metrics,
        trial_history=trial_history,
        hypothesis=hypothesis,
        oos_split_ratio=oos_split_ratio,
        bars_per_day=bars_per_day,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import statistics_service as svc
        return svc.run_tests(**args)
    return _bridge().call("run_tests", {k: v for k, v in args.items() if v is not None})
