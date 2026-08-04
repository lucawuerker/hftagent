"""LOBSTER provider — the original 10-second microstructure CSV loader.

This wraps the existing :func:`quant_fund_agent.backtesting.data_loader.load_panel`
behind the :class:`~quant_fund_agent.data.providers.base.DataProvider` interface
so the legacy LOBSTER path runs **byte-identically** while every consumer routes
through the unified data layer.  The heavy CSV-parsing implementation continues
to live in ``backtesting/data_loader.py`` (also still importable directly for
``forward_returns`` and friends).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from quant_fund_agent.data.providers.base import DataProvider
from quant_fund_agent.data.tiers import lobster_fields_for_level

_PER_LEVEL_RE = re.compile(r"(?:ask|bid)(?:Price|Depth)\d+$")


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
        fields = lobster_fields_for_level(level)
        # The tier advertises book levels 1-10, but the feed on disk may carry
        # fewer (e.g. a level-5 export).  Advertising a per-level column the
        # CSVs don't have makes the researcher invent factors whose signals
        # are all-NaN — sniff the CSV headers and drop absent per-level
        # columns.  Fail-open: any surprise keeps the static advertisement.
        try:
            discovered = self._discovered_per_level_fields()
            if discovered is not None:
                fields = frozenset(
                    f for f in fields
                    if not _PER_LEVEL_RE.fullmatch(f) or f in discovered)
        except Exception:  # noqa: BLE001 — advertisement must never crash
            pass
        return fields

    def _discovered_per_level_fields(self) -> frozenset[str] | None:
        """Union of per-level book columns present in the on-disk CSV headers.

        Reads one header line per configured ticker directory (cheap).  Returns
        ``None`` when nothing could be sniffed (no dirs / no CSVs) so the
        caller keeps the static tier advertisement.
        """
        root = Path(self.data.data_dir)
        if not root.is_dir():
            return None
        tickers = self.data.tickers
        dirs = ([root / t for t in tickers] if tickers
                else [d for d in sorted(root.iterdir()) if d.is_dir()])
        found: set[str] = set()
        sniffed = False
        for d in dirs:
            if not d.is_dir():
                continue
            csvs = sorted(d.glob("bin*.csv"))
            if not csvs:
                continue
            with open(csvs[0], "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline().strip().split(",")
            sniffed = True
            found.update(c for c in header if _PER_LEVEL_RE.fullmatch(c))
        return frozenset(found) if sniffed else None

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
