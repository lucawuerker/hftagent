"""Tests for the quant-research MCP server + client (protocol round-trip + parity).

``existing_factor_ids`` touches only the registry + factor DB so it's fast.
``materialise_factor`` writes a tiny generated factor file and is cleaned up.
"""

from __future__ import annotations

import pytest


def test_existing_factor_ids_mcp_matches_inprocess(monkeypatch):
    from quant_fund_agent.mcp import research_client

    monkeypatch.setenv("RESEARCH_USE_MCP", "0")
    inproc = research_client.existing_factor_ids()

    monkeypatch.setenv("RESEARCH_USE_MCP", "1")
    viamcp = research_client.existing_factor_ids()

    assert inproc == viamcp
    assert isinstance(inproc, list)


_GOOD_FACTOR = '''\
from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class McpParityProbe(BaseFactor):
    factor_id = "mcp_parity_probe"
    name = "MCP Parity Probe"
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return close.pct_change().fillna(0.0)
'''


def test_materialise_factor_over_mcp(monkeypatch):
    monkeypatch.setenv("RESEARCH_USE_MCP", "1")
    from quant_fund_agent.mcp import research_client
    from quant_fund_agent.mcp.research_service import _drop_researcher_factor

    fid = "mcp_parity_probe"
    result = research_client.materialise_factor(fid, _GOOD_FACTOR)
    try:
        assert result["ok"] is True
        assert result["code_path"].endswith(f"{fid}.py")
    finally:
        _drop_researcher_factor(fid, result.get("code_path", ""))


def test_materialise_factor_rejects_bad_code(monkeypatch):
    monkeypatch.setenv("RESEARCH_USE_MCP", "1")
    from quant_fund_agent.mcp import research_client

    result = research_client.materialise_factor("mcp_bad_probe", "this is not python {")
    assert result["ok"] is False
    assert "error" in result
