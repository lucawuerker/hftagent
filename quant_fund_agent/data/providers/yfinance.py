"""yfinance provider — key-free daily (and intraday) OHLCV from Yahoo Finance.

The first public, no-API-key vendor, so anyone can clone the repo and run the
fund on their own pull.  Supplies the ``standard`` tier (OHLCV); ``vwap`` and
``returns`` are synthesised by the panel layer, and microstructure factors are
gated out (see ``docs/data-layer``).

Prices use ``auto_adjust=True`` — split/dividend adjusted, with ``close`` the
adjusted close.  NOTE: full-history adjustment is not point-in-time, a mild
price-level look-ahead that is standard for daily factor research; combined with
static universe presets this carries the usual survivorship caveats.

Network is touched in exactly one place (:meth:`_fetch`), so universe resolution,
caching and panel assembly are all testable offline by injecting a fake fetch.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_fund_agent.data.providers.base import DataProvider
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.yfinance")

# Map our frequency strings to yfinance interval codes (others pass through).
_YF_INTERVAL = {
    "1d": "1d", "1day": "1d", "daily": "1d",
    "1min": "1m", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "60m", "1h": "60m", "1wk": "1wk", "1mo": "1mo",
}

_OHLCV = ("open", "high", "low", "close", "volume")


def _reshape(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Turn a ``yf.download`` frame into ``{symbol: tidy OHLCV DataFrame}``.

    yfinance returns MultiIndex (ticker, field) columns for many tickers and flat
    columns for a single ticker; handle both.
    """
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out

    multi = isinstance(raw.columns, pd.MultiIndex)

    def _tidy(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.rename(columns=str.lower)
        keep = [c for c in _OHLCV if c in sub.columns]
        sub = sub[keep].dropna(how="all")
        sub.index = pd.to_datetime(sub.index)
        return sub.sort_index()

    if multi:
        level0 = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym in level0:
                tidy = _tidy(raw[sym])
                if not tidy.empty:
                    out[sym] = tidy
    elif len(symbols) == 1:
        tidy = _tidy(raw)
        if not tidy.empty:
            out[symbols[0]] = tidy
    else:
        log.warning("yfinance returned flat columns for %d symbols; cannot "
                    "disambiguate — skipping", len(symbols))
    return out


class YFinanceProvider(DataProvider):
    name = "yfinance"
    asset_classes = ("equity",)

    def available_fields(self) -> frozenset[str]:
        return TIERS["standard"]

    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        from quant_fund_agent.data.cache import cached_fetch
        from quant_fund_agent.data.universe import resolve_universe

        if not (self.data.start and self.data.end):
            raise ValueError(
                "yfinance provider needs data.start and data.end (run "
                "`python -m quant_fund_agent.setup` to write quant.config.yaml)."
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

    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """The ONLY network call — pull OHLCV from Yahoo for ``symbols``."""
        import yfinance as yf

        interval = _YF_INTERVAL.get(self.data.frequency, self.data.frequency)
        raw = yf.download(
            symbols,
            start=self.data.start,
            end=self.data.end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return _reshape(raw, symbols)
