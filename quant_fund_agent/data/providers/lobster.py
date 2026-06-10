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
from quant_fund_agent.data.tiers import TIERS


class LobsterProvider(DataProvider):
    """Reads local LOBSTER CSVs under ``data.data_dir`` (default ``ticker_data/``)."""

    name = "lobster"
    asset_classes = ("equity",)

    def available_fields(self) -> frozenset[str]:
        # LOBSTER supplies OHLCV + full microstructure, but NOT the fundamental
        # classification fields (sector/industry/cap).
        return TIERS["standard"] | TIERS["microstructure"]

    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        from quant_fund_agent.backtesting.data_loader import load_panel

        return load_panel(
            self.data.data_dir,
            tickers=self.data.tickers,
            fields=fields,
            n_tickers=self.data.n_tickers,
            dtype=self.data.dtype,
        )
