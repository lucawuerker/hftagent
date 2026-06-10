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


# Fields the panel layer derives from OHLCV when a provider doesn't supply them,
# mapped to the raw fields they need.  ``vwap`` uses the "typical price" proxy.
SYNTH_DEPS: dict[str, tuple[str, ...]] = {
    "vwap": ("high", "low", "close"),
    "returns": ("close",),
}


def get_provider(settings: Settings | None = None) -> "DataProvider":
    """Instantiate the configured provider (cheap; loads no data)."""
    settings = settings or get_settings()
    return get_provider_class(settings.data.provider)(settings)


def _synthesize(
    panel: dict[str, pd.DataFrame], wanted: list[str]
) -> dict[str, pd.DataFrame]:
    """Add any ``wanted`` derived fields the provider didn't already supply.

    A field is only synthesised when all of its OHLCV deps are present; otherwise
    it is left absent so the consumer (factor gating) can drop the factor cleanly.
    """
    for field in wanted:
        if field in panel or field not in SYNTH_DEPS:
            continue
        deps = SYNTH_DEPS[field]
        if not all(d in panel for d in deps):
            continue
        if field == "vwap":
            panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
        elif field == "returns":
            panel["returns"] = panel["close"].pct_change()
    return panel


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

    if fields is None:
        # Full load: materialise everything the provider has, plus every derived
        # field it didn't already supply.
        panel = provider.load(fields=None)
        return _synthesize(panel, list(SYNTH_DEPS))

    # Targeted load: a synthesised field is replaced by its OHLCV deps in the
    # provider request (the provider never sees vwap/returns), then derived and
    # the result trimmed back to exactly what the caller asked for.
    requested = list(fields)
    synth_requested = [f for f in requested if f in SYNTH_DEPS]
    provider_fields: set[str] = {f for f in requested if f not in SYNTH_DEPS}
    for f in synth_requested:
        provider_fields.update(SYNTH_DEPS[f])

    panel = provider.load(fields=sorted(provider_fields) if provider_fields else None)
    panel = _synthesize(panel, synth_requested)
    return {k: panel[k] for k in requested if k in panel}
