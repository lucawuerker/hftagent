"""Pluggable market-data layer.

The single seam the rest of the system loads data through.  ``load_panel`` here
replaces direct imports of the LOBSTER CSV loader and dispatches to whichever
provider is configured (``data.provider`` in ``quant.config.yaml`` / env).

See ``docs/data-layer/ARCHITECTURE.md`` for the design.
"""

from __future__ import annotations

from quant_fund_agent.data.panel import get_provider, load_panel

__all__ = ["load_panel", "get_provider"]
