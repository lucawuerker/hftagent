"""MCP server exposing the Statistician's statistical-test toolbox.

Run as a stdio server::

    python -m quant_fund_agent.mcp.statistics_server

It owns the (heavy) data panel server-side and caches factor signals, so the
out-of-sample backtest reuses a single loaded panel.  Only small JSON crosses
the MCP boundary (the strategy spec, IS metrics, trial history → test results).

All real work is delegated to :mod:`quant_fund_agent.mcp.statistics_service`, so
the in-process fallback in ``statistics_client.py`` computes identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.mcp import statistics_service as svc

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.statistics_server")

mcp = FastMCP("quant-statistics")


@mcp.tool()
def list_tests() -> str:
    """List the registered statistical tests (JSON).

    Returns ``{"mandatory": [test_id, ...], "optional": [{"id", "description"}, ...]}``.
    """
    return json.dumps(svc.list_tests())


@mcp.tool()
def run_tests(
    test_ids: list[str],
    spec: dict,
    is_metrics: dict | None = None,
    trial_history: list[dict] | None = None,
    hypothesis: str = "",
    oos_split_ratio: float = 0.2,
    bars_per_day: int = 2340,
    as_of: str | None = None,
) -> str:
    """Run the requested statistical tests on the candidate strategy.

    The full test context (panel, factor signals, deserialised IS returns) is
    rebuilt server-side from ``spec`` / ``is_metrics`` / ``trial_history``.
    ``as_of`` (walk-forward backtest): ISO date string limiting visible data.
    Returns a JSON array of ``StatTestResult`` objects.
    """
    return json.dumps(svc.run_tests(
        test_ids=test_ids,
        spec=spec,
        is_metrics=is_metrics,
        trial_history=trial_history,
        hypothesis=hypothesis,
        oos_split_ratio=oos_split_ratio,
        bars_per_day=bars_per_day,
        as_of=as_of,
    ))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
