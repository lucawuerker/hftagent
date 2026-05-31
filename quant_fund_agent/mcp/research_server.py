"""MCP server exposing the Factor Researcher's deterministic toolbox.

Run as a stdio server::

    python -m quant_fund_agent.mcp.research_server

It owns the (heavy) factor panel server-side and caches it, so materialising a
candidate and IC-backtesting it reuse a single loaded panel.  Only small JSON
crosses the MCP boundary.

All real work is delegated to :mod:`quant_fund_agent.mcp.research_service`, so
the in-process fallback in ``research_client.py`` computes identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.mcp import research_service as svc

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.research_server")

mcp = FastMCP("quant-research")


@mcp.tool()
def load_papers(
    n: int = 2,
    cutoff_date: str | None = None,
    strategy: str = "unread_first",
    max_chars: int = 30_000,
) -> str:
    """Select up to ``n`` papers (published before ``cutoff_date`` if given) and
    extract their text.  Returns a JSON array of snippet objects.
    """
    return json.dumps(svc.load_papers(
        n=n, cutoff_date=cutoff_date, strategy=strategy, max_chars=max_chars,
    ))


@mcp.tool()
def existing_factor_ids() -> str:
    """Return the sorted union of registered + persisted factor IDs (JSON array)."""
    return json.dumps(svc.existing_factor_ids())


@mcp.tool()
def materialise_factor(factor_id: str, code: str) -> str:
    """Validate / write / import / smoke-test generated factor code.

    Returns JSON ``{"ok": bool, "code_path"?: str, "error"?: str}``.
    """
    return json.dumps(svc.materialise_factor(factor_id, code))


@mcp.tool()
def backtest_factors(
    factor_ids: list[str],
    horizon: int = 6,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
) -> str:
    """Run the single-factor IC backtest for each factor (JSON keyed by id)."""
    return json.dumps(svc.backtest_factors(
        factor_ids=factor_ids, horizon=horizon, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers,
    ))


@mcp.tool()
def persist_results(kept_records: list[dict], rejected: list[dict]) -> str:
    """Persist survivor FactorRecords and purge rejects.  Returns JSON
    ``{"kept_factor_ids": [...], "rejected_factor_ids": [...]}``.
    """
    return json.dumps(svc.persist_results(kept_records, rejected))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
