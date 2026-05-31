"""MCP server exposing the factor catalog used by the Selector Agent.

Run as a stdio server::

    python -m quant_fund_agent.mcp.catalog_server

All real work is delegated to :mod:`quant_fund_agent.mcp.catalog_service`, so the
in-process fallback in ``catalog_client.py`` computes identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.mcp.catalog_service import load_factor_catalog as _load_factor_catalog

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.catalog_server")

mcp = FastMCP("quant-catalog")


@mcp.tool()
def load_factor_catalog() -> str:
    """Return the persisted factor catalog as a JSON array.

    Each entry has ``factor_id``, ``name``, ``category``, ``description`` and the
    IC / ICIR at the 1, 6 and 60-bar horizons (``ic_1`` … ``icir_60``).
    """
    catalog = _load_factor_catalog()
    log.info("load_factor_catalog -> %d factors", len(catalog))
    return json.dumps(catalog)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
