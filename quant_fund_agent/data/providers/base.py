"""The data-provider abstraction.

Every market-data vendor is a :class:`DataProvider` that knows how to (a) declare
which fields it can supply and (b) materialise the field panel the rest of the
system consumes.  The panel contract is unchanged from the original LOBSTER
loader::

    load(...) -> dict[field_name -> pd.DataFrame(index=DatetimeIndex, cols=tickers)]

A provider is responsible only for *producing raw fields*.  Capability gating,
derived-field synthesis (``vwap``/``returns``), frequency-aware annualization and
caching are orchestrated above it in :mod:`quant_fund_agent.data.panel`.

See ``docs/data-layer/DATA_PROVIDERS.md`` for a step-by-step guide to adding one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_fund_agent.config import Settings


class DataProvider(ABC):
    """Base class for all market-data providers."""

    #: short identifier used to select the provider from config (``data.provider``)
    name: str = ""
    #: which asset classes this provider serves
    asset_classes: tuple[str, ...] = ("equity",)

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.data = settings.data

    # ── capability ──────────────────────────────────────────────────────

    @abstractmethod
    def available_fields(self) -> frozenset[str]:
        """The set of panel fields this provider can supply.

        Drives factor gating: a factor is admitted iff its ``inputs`` are a
        subset of this (``vwap``/``returns`` are always satisfiable because the
        panel layer synthesizes them).  Only advertise fields you actually
        deliver.
        """
        ...

    # ── data ────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Return the field panel for this provider's configured universe/range.

        ``fields`` is an optional subset to materialise (a memory optimisation);
        ``None`` means "every field this provider supplies".  The returned dict
        maps field name → wide ``DataFrame`` (shared ``DatetimeIndex`` × tickers).
        """
        ...


class ApiProvider(DataProvider):
    """Base for REST/API vendors (yfinance / FMP / AlphaVantage / …).

    Implements the shared ``load()`` flow — resolve the configured universe,
    pull through the parquet cache, assemble the wide panel — so a concrete
    provider only implements :meth:`available_fields` and :meth:`_fetch` (the
    single network call returning ``{symbol: tidy OHLCV DataFrame}``).
    """

    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        from quant_fund_agent.data.cache import cached_fetch
        from quant_fund_agent.data.universe import resolve_universe

        if not (self.data.start and self.data.end):
            raise ValueError(
                f"{self.name} provider needs data.start and data.end "
                "(run `python -m quant_fund_agent.setup` to write quant.config.yaml)."
            )
        symbols = resolve_universe(self.data)
        panel = cached_fetch(
            self.name, symbols, self.data.start, self.data.end,
            self.data.frequency, self.data.asset_class, self.data.cache_dir,
            self._fetch,
        )
        if fields is not None:
            panel = {k: v for k, v in panel.items() if k in fields}
        return panel

    @abstractmethod
    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """The single network call — ``{symbol: tidy DataFrame}`` (index × OHLCV)."""
        ...
