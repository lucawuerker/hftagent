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

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_fund_agent.config import Settings

log = logging.getLogger("data.providers.base")


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
        panel = self._maybe_add_fundamentals(panel, symbols, fields)
        if fields is not None:
            panel = {k: v for k, v in panel.items() if k in fields}
        return panel

    @abstractmethod
    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """The single network call — ``{symbol: tidy DataFrame}`` (index × OHLCV)."""
        ...

    # ── non-OHLCV (fundamentals / estimates / events) ───────────────────────

    def _fetch_fundamentals(
        self, symbols: list[str]
    ) -> dict[str, pd.DataFrame] | None:
        """Per-symbol **availability-stamped** record frames, or ``None``.

        Override in a provider that supplies non-OHLCV fields.  Each frame is
        indexed by availability date (see
        :func:`quant_fund_agent.data.fundamentals.build_record_frame`) with
        canonical field columns; the base class caches them (long TTL) and aligns
        them onto the price index, so the override only does the vendor I/O.
        """
        return None

    def _fundamentals_enabled(self) -> bool:
        """Whether to enrich the panel with non-OHLCV fields for this run.

        Equity-only (crypto/FX have no fundamentals), opt-out via
        ``DataSettings.fundamentals`` or ``QF_FUNDAMENTALS=0``, and only when the
        provider actually advertises non-OHLCV fields.
        """
        from quant_fund_agent.data.fields import NON_OHLCV_FIELDS

        if str(self.data.asset_class).lower() != "equity":
            return False
        if not getattr(self.data, "fundamentals", True):
            return False
        if os.getenv("QF_FUNDAMENTALS", "1") == "0":
            return False
        return bool(self.available_fields() & NON_OHLCV_FIELDS)

    def _maybe_add_fundamentals(
        self,
        panel: dict[str, pd.DataFrame],
        symbols: list[str],
        fields: list[str] | None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch, cache, align and merge non-OHLCV fields into ``panel``.

        Best-effort: a fundamentals failure logs a warning and leaves the OHLCV
        panel intact (the dependent factors then gate out / read NaN) rather than
        aborting the whole load.
        """
        from quant_fund_agent.data.fields import NON_OHLCV_FIELDS

        if not panel or not self._fundamentals_enabled():
            return panel
        # Skip the (costly) fetch when a targeted load asks for no non-OHLCV field.
        if fields is not None and not (set(fields) & NON_OHLCV_FIELDS):
            return panel

        from quant_fund_agent.data.cache import (
            DEFAULT_RECORDS_TTL_DAYS,
            cached_records,
        )
        from quant_fund_agent.data.fundamentals import (
            DEFAULT_STALENESS_CAP_DAYS,
            align_to_index,
        )

        try:
            records = cached_records(
                self.name, symbols, self.data.asset_class, "fundamentals",
                self.data.cache_dir, self._fetch_fundamentals,
                ttl_days=int(getattr(self.data, "fundamentals_ttl_days",
                                     DEFAULT_RECORDS_TTL_DAYS)),
            )
        except Exception as e:  # noqa: BLE001 — enrichment must not kill the load
            log.warning("%s: fundamentals fetch failed, OHLCV-only: %s",
                        self.name, e)
            return panel
        if not records:
            return panel

        price_index = next(iter(panel.values())).index
        aligned = align_to_index(
            records, price_index,
            staleness_cap_days=int(getattr(self.data, "fundamentals_staleness_days",
                                           DEFAULT_STALENESS_CAP_DAYS)),
        )
        panel.update(aligned)
        return panel
