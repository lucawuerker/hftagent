"""LOBSTER provider — the original 10-second microstructure CSV loader.

This wraps the existing :func:`quant_fund_agent.backtesting.data_loader.load_panel`
behind the :class:`~quant_fund_agent.data.providers.base.DataProvider` interface
so the legacy LOBSTER path runs **byte-identically** while every consumer routes
through the unified data layer.  The heavy CSV-parsing implementation continues
to live in ``backtesting/data_loader.py`` (also still importable directly for
``forward_returns`` and friends).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.data.providers.base import DataProvider
from quant_fund_agent.data.tiers import lobster_fields_for_level


class LobsterProvider(DataProvider):
    """Reads local LOBSTER CSVs under ``data.data_dir`` (default ``ticker_data/``)."""

    name = "lobster"
    asset_classes = ("equity",)

    def available_fields(self) -> frozenset[str]:
        # LOBSTER supplies OHLCV + microstructure, but NOT the fundamental
        # classification fields (sector/industry/cap).  Which microstructure
        # fields are advertised depends on the configured order-book level: a
        # Level-2 feed exposes only book-derivable fields, a Level-3 feed adds
        # the trade/message-stream fields (see ``data.lobster_level``).
        level = int(getattr(self.data, "lobster_level", 3) or 3)
        return lobster_fields_for_level(level)

    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        from quant_fund_agent.backtesting.data_loader import load_panel

        # Resolve explicit list > preset > None (auto-discover from disk).
        tickers = self.data.tickers
        if tickers is None and self.data.universe_preset:
            from quant_fund_agent.data.universe import resolve_universe
            tickers = resolve_universe(self.data)

        return load_panel(
            self.data.data_dir,
            tickers=tickers,
            fields=fields,
            n_tickers=self.data.n_tickers,
            dtype=self.data.dtype,
        )
