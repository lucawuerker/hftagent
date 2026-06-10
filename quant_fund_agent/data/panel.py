"""Unified panel entry point — the single seam the whole system loads data from.

Every consumer (modeling service, research service, architect graph, simulation)
calls :func:`load_panel` here instead of importing a vendor-specific loader.  The
provider is chosen from :class:`~quant_fund_agent.config.Settings`
(``data.provider``); for the default ``lobster`` provider this delegates to the
original CSV loader with identical arguments, so the legacy path is byte-identical.

Derived-field synthesis (``vwap``/``returns``) and frequency-aware annualization
metadata are layered on here in later phases; for now this is a thin, behaviour-
preserving dispatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_fund_agent.config import Settings, get_settings
from quant_fund_agent.data.providers import get_provider_class

if TYPE_CHECKING:
    from quant_fund_agent.data.providers.base import DataProvider


def get_provider(settings: Settings | None = None) -> "DataProvider":
    """Instantiate the configured provider (cheap; loads no data)."""
    settings = settings or get_settings()
    return get_provider_class(settings.data.provider)(settings)


def load_panel(
    data_dir: str | None = None,
    tickers: list[str] | None = None,
    fields: list[str] | None = None,
    n_tickers: int | None = None,
    dtype: str | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the field panel for the configured provider.

    The positional signature mirrors the original LOBSTER ``load_panel`` so the
    four historical call sites only need to swap their import.  Any explicit
    (non-``None``) argument overrides the ambient config for this call without
    mutating the shared settings object.

    Returns ``dict[field -> DataFrame(index=DatetimeIndex, columns=tickers)]``.
    """
    settings = settings or get_settings()
    settings = settings.with_data_overrides(
        data_dir=data_dir,
        tickers=tickers,
        n_tickers=n_tickers,
        dtype=dtype,
    )
    provider = get_provider(settings)
    return provider.load(fields=fields)
