"""Pluggable market-data layer.

The single seam the rest of the system loads data through.  ``load_panel`` here
replaces direct imports of the LOBSTER CSV loader and dispatches to whichever
provider is configured (``data.provider`` in ``quant.config.yaml`` / env).

See ``docs/data-layer/ARCHITECTURE.md`` for the design.
"""

from __future__ import annotations

import os

from quant_fund_agent.data.panel import get_provider, load_panel

__all__ = ["load_panel", "get_provider", "usable_fields"]


def usable_fields(settings=None) -> frozenset[str]:
    """The fields a factor may actually rely on for the configured run.

    Starts from the provider's advertised :meth:`available_fields` (which already
    honors the LOBSTER order-book level) and additionally drops the non-OHLCV
    (fundamental / estimate / event) fields when fundamentals are turned off for
    this run — i.e. when the asset class isn't equity, ``data.fundamentals`` is
    ``False`` or ``QF_FUNDAMENTALS=0``.  This is the set the Factor Researcher's
    data context is built from and the factors it produces are gated against, so
    a researcher never invents factors needing data this run can't supply.
    """
    from quant_fund_agent.config import get_settings
    from quant_fund_agent.data.fields import NON_OHLCV_FIELDS

    settings = settings or get_settings()
    fields = set(get_provider(settings).available_fields())

    fundamentals_on = (
        str(settings.data.asset_class).lower() == "equity"
        and bool(getattr(settings.data, "fundamentals", True))
        and os.getenv("QF_FUNDAMENTALS", "1") != "0"
    )
    if not fundamentals_on:
        fields -= NON_OHLCV_FIELDS
    return frozenset(fields)
