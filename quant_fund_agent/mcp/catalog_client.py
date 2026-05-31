"""Synchronous client facade for the quant-catalog MCP server.

Set ``CATALOG_USE_MCP=0`` (or the global ``QF_USE_MCP=0``) to call the
:mod:`quant_fund_agent.mcp.catalog_service` functions in-process instead.
"""

from __future__ import annotations

from typing import Any

from quant_fund_agent.mcp._bridge import MCPBridge, use_mcp

_SERVER_MODULE = "quant_fund_agent.mcp.catalog_server"


def _use_mcp() -> bool:
    return use_mcp("CATALOG_USE_MCP")


def load_factor_catalog() -> list[dict[str, Any]]:
    """Return the factor catalog (via MCP, or in-process if disabled)."""
    if not _use_mcp():
        from quant_fund_agent.mcp.catalog_service import load_factor_catalog as _impl
        return _impl()
    return MCPBridge.instance(_SERVER_MODULE).call("load_factor_catalog", {})
