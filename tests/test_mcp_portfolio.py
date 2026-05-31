"""Tests for the quant-portfolio MCP server + client (protocol round-trip + parity).

The PM analytics are pure functions of JSON inputs, so these tests run on small
synthetic inputs and need no market data.
"""

from __future__ import annotations

import pandas as pd
import pytest

_IDS = ["a", "b", "c"]
_COV = pd.DataFrame(
    [[0.04, 0.01, 0.00],
     [0.01, 0.09, 0.02],
     [0.00, 0.02, 0.16]],
    index=_IDS, columns=_IDS,
)
_EXPECTED = {"a": 0.10, "b": 0.20, "c": 0.15}
_ROWS = [
    {"strategy_id": "a", "is_sharpe": 1.2, "is_max_dd": -0.10, "is_hit_rate": 0.55},
    {"strategy_id": "b", "is_sharpe": 0.8, "is_max_dd": -0.20, "is_hit_rate": 0.52},
    {"strategy_id": "c", "is_sharpe": 0.3, "is_max_dd": -0.05, "is_hit_rate": 0.50},
]
_PROFILE = {
    "min_sharpe": 0.5,
    "max_drawdown": -0.5,
    "min_hit_rate": None,
    "max_pairwise_correlation": 0.9,
}


def test_screen_strategies_parity(monkeypatch):
    from quant_fund_agent.mcp import portfolio_client

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "0")
    inproc = portfolio_client.screen_strategies(_ROWS, _PROFILE, 2, _COV)

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "1")
    viamcp = portfolio_client.screen_strategies(_ROWS, _PROFILE, 2, _COV)

    assert inproc == viamcp
    # 'c' fails the min_sharpe filter; 'a' and 'b' are eligible.
    assert set(inproc["eligible_strategy_ids"]) == {"a", "b"}


def test_construct_portfolio_parity(monkeypatch):
    from quant_fund_agent.mcp import portfolio_client

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "0")
    inproc = portfolio_client.construct_portfolio(
        "max_sharpe", _IDS, _COV, _EXPECTED, max_weight=0.6)

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "1")
    viamcp = portfolio_client.construct_portfolio(
        "max_sharpe", _IDS, _COV, _EXPECTED, max_weight=0.6)

    assert inproc["method"] == viamcp["method"]
    for sid in _IDS:
        assert inproc["weights"][sid] == pytest.approx(viamcp["weights"][sid], rel=1e-9, abs=1e-9)
    assert sum(abs(w) for w in inproc["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_expected_metrics_parity(monkeypatch):
    from quant_fund_agent.mcp import portfolio_client

    weights = {"a": 0.5, "b": 0.3, "c": 0.2}

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "0")
    inproc = portfolio_client.expected_portfolio_metrics(weights, _EXPECTED, _COV)

    monkeypatch.setenv("PORTFOLIO_USE_MCP", "1")
    viamcp = portfolio_client.expected_portfolio_metrics(weights, _EXPECTED, _COV)

    assert inproc.keys() == viamcp.keys()
    for k in inproc:
        assert inproc[k] == pytest.approx(viamcp[k], rel=1e-9, abs=1e-9)
