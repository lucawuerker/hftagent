"""Tests for the quant-statistics MCP server + client (protocol round-trip + parity).

``list_tests`` exercises the real MCP stdio boundary without touching data, so
it's fast.  (``run_tests`` needs a full panel + fitted artifact and is covered
indirectly by the statistician integration path.)
"""

from __future__ import annotations


def test_list_tests_mcp_matches_inprocess(monkeypatch):
    from quant_fund_agent.mcp import statistics_client

    monkeypatch.setenv("STATISTICS_USE_MCP", "0")
    inproc = statistics_client.list_tests()

    monkeypatch.setenv("STATISTICS_USE_MCP", "1")
    viamcp = statistics_client.list_tests()

    assert inproc == viamcp
    # The two MVP mandatory tests must be advertised.
    assert "deflated_sharpe_ratio" in inproc["mandatory"]
    assert "out_of_sample_backtest" in inproc["mandatory"]
