"""Tests for the modeling MCP server + client (protocol round-trip + parity).

The ``list_models`` tests exercise the real MCP stdio boundary but don't touch
data, so they're fast.  The ``fit_and_backtest`` parity test loads a tiny slice
of the real panel and is skipped when ``ticker_data/`` isn't present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Cap the universe for any real-panel load the MCP server / in-process service
# does in this module (set before the client ever spawns the server subprocess).
os.environ.setdefault("ARCHITECT_N_TICKERS", "3")

from quant_fund_agent.modeling import list_model_menu  # noqa: E402

HAS_DATA = Path(os.getenv("DATA_DIR", "ticker_data")).is_dir()
ARTIFACT_DIR = Path("data/strategies/models")


def _model_types(menu):
    return sorted(m["model_type"] for m in menu)


def test_list_models_inprocess(monkeypatch):
    monkeypatch.setenv("MODELING_USE_MCP", "0")
    from quant_fund_agent.mcp import client

    assert _model_types(client.list_models()) == _model_types(list_model_menu())


def test_list_models_over_mcp(monkeypatch):
    monkeypatch.setenv("MODELING_USE_MCP", "1")
    from quant_fund_agent.mcp import client

    menu = client.list_models()
    assert _model_types(menu) == _model_types(list_model_menu())
    # The static baseline plus the always-available sklearn models are advertised.
    assert "static_weights" in _model_types(menu)
    assert "random_forest" in _model_types(menu)


@pytest.mark.skipif(not HAS_DATA, reason="ticker_data/ not available")
def test_fit_and_backtest_mcp_matches_inprocess(monkeypatch):
    from quant_fund_agent.mcp import client

    common = dict(
        model_type="ridge",
        factor_ids=None,
        model_params={"alpha": 1.0},
        target_horizon=6,
        holding_period=6,
        max_positions=10,
        oos_split_ratio=0.2,
    )
    # ridge needs features; use whatever seed factors exist in the registry.
    from quant_fund_agent.factors import discover_factors, get_all_factor_classes

    discover_factors()
    factor_ids = sorted(get_all_factor_classes().keys())[:3]
    assert factor_ids, "no factors registered to test with"
    common["factor_ids"] = factor_ids

    created: list[str] = []
    try:
        monkeypatch.setenv("MODELING_USE_MCP", "0")
        inproc = client.fit_and_backtest(strategy_id="parity_inproc", **common)
        created.append(inproc["artifact_path"])

        monkeypatch.setenv("MODELING_USE_MCP", "1")
        viamcp = client.fit_and_backtest(strategy_id="parity_mcp", **common)
        created.append(viamcp["artifact_path"])

        # Ridge is deterministic → MCP and in-process must agree.
        assert inproc["metrics"]["sharpe_ratio"] == pytest.approx(
            viamcp["metrics"]["sharpe_ratio"], rel=1e-6, abs=1e-6
        )
        assert inproc["metrics"]["annualised_return"] == pytest.approx(
            viamcp["metrics"]["annualised_return"], rel=1e-6, abs=1e-6
        )

        # The MCP-produced artifact is reloadable from the main process.
        import joblib

        blob = joblib.load(viamcp["artifact_path"])
        assert blob["factor_ids"] == factor_ids
        assert blob["model_type"] == "ridge"
    finally:
        for p in created:
            if p:
                Path(p).unlink(missing_ok=True)
