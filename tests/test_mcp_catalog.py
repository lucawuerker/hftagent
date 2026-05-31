"""Tests for the quant-catalog MCP server + client (protocol round-trip + parity).

Exercises the real MCP stdio boundary; reads the persisted factor DB only, so
it's fast and skipped when the factor DB isn't present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund_agent.mcp.catalog_service import FACTOR_DB_PATH

HAS_DB = Path(FACTOR_DB_PATH).is_file()


def _ids(catalog):
    return sorted(f["factor_id"] for f in catalog)


@pytest.mark.skipif(not HAS_DB, reason="factor_db.json not available")
def test_load_factor_catalog_inprocess(monkeypatch):
    monkeypatch.setenv("CATALOG_USE_MCP", "0")
    from quant_fund_agent.mcp import catalog_client

    catalog = catalog_client.load_factor_catalog()
    assert isinstance(catalog, list)
    assert all("factor_id" in f and "name" in f for f in catalog)


@pytest.mark.skipif(not HAS_DB, reason="factor_db.json not available")
def test_load_factor_catalog_mcp_matches_inprocess(monkeypatch):
    from quant_fund_agent.mcp import catalog_client

    monkeypatch.setenv("CATALOG_USE_MCP", "0")
    inproc = catalog_client.load_factor_catalog()

    monkeypatch.setenv("CATALOG_USE_MCP", "1")
    viamcp = catalog_client.load_factor_catalog()

    assert _ids(inproc) == _ids(viamcp)
    assert inproc == viamcp
