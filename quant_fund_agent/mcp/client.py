"""Synchronous client facade for the quant-modeling MCP server.

The Architect runs inside synchronous LangGraph nodes, but the MCP SDK is
async-only and the stdio session must stay open across many calls (so the server
loads the data panel just once).  The :mod:`quant_fund_agent.mcp._bridge` module
handles that; this module is the thin domain facade that exposes plain
synchronous ``list_models`` / ``fit_and_backtest`` functions.

Set ``MODELING_USE_MCP=0`` (or the global ``QF_USE_MCP=0``) to bypass MCP
entirely and call the same :mod:`quant_fund_agent.modeling.service` functions
in-process — used by the test suite and as an escape hatch if the stdio
transport misbehaves.
"""

from __future__ import annotations

import logging
from typing import Any

from quant_fund_agent.mcp._bridge import MCPBridge, use_mcp

log = logging.getLogger("mcp.client")

_SERVER_MODULE = "quant_fund_agent.mcp.modeling_server"


def _use_mcp() -> bool:
    return use_mcp("MODELING_USE_MCP")


# ---------------------------------------------------------------------------
# In-process path (also the implementation the MCP server wraps)
# ---------------------------------------------------------------------------

def _inproc_list_models() -> list[dict[str, Any]]:
    from quant_fund_agent.modeling.service import model_menu
    return model_menu()


def _inproc_fit_and_backtest(**kwargs: Any) -> dict[str, Any]:
    from quant_fund_agent.modeling.service import run_fit_and_backtest
    return run_fit_and_backtest(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models() -> list[dict[str, Any]]:
    """Return the model catalog (via MCP, or in-process if disabled)."""
    if not _use_mcp():
        return _inproc_list_models()
    return MCPBridge.instance(_SERVER_MODULE).call("list_models", {})


def fit_and_backtest(**kwargs: Any) -> dict[str, Any]:
    """Fit + backtest one candidate (via MCP, or in-process if disabled).

    Accepts the same keyword arguments as
    :func:`quant_fund_agent.modeling.service.run_fit_and_backtest`.
    """
    if not _use_mcp():
        return _inproc_fit_and_backtest(**kwargs)
    # MCP arguments must be JSON-serialisable; drop ``None`` so server defaults apply.
    args = {k: v for k, v in kwargs.items() if v is not None}
    return MCPBridge.instance(_SERVER_MODULE).call("fit_and_backtest", args)
