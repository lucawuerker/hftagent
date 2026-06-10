"""AlphaVantage provider — keyed daily OHLCV.

Official REST API; needs ``ALPHAVANTAGE_API_KEY`` in ``.env``.  Supplies the
``standard`` tier (OHLCV); ``vwap``/``returns`` are synthesised by the panel.

CAVEATS (free tier):
  * **UNADJUSTED** — ``TIME_SERIES_DAILY`` returns raw prices (the adjusted
    endpoint is premium), so splits cause level jumps.
  * **~100 bars only** — ``outputsize=full`` is premium, so the free tier returns
    just the most recent ~100 daily bars (``outputsize=compact``).
  * **rate-limited** — ~5 requests/minute, ~25/day.
For longer history or adjusted prices prefer FMP or yfinance.  The shared HTTP
helper throttles to one call every ~13s and the parquet cache makes re-runs free.

Network is touched only in :meth:`_fetch`; the JSON reshape is unit-tested
offline with synthetic payloads.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from quant_fund_agent.data.providers._http import RateLimited, request_json
from quant_fund_agent.data.providers.base import ApiProvider
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.alphavantage")

AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 13.0  # ~5 requests/minute free tier

_COLMAP = {
    "1. open": "open", "2. high": "high", "3. low": "low",
    "4. close": "close", "5. volume": "volume",
}


def _check_limits(payload: dict) -> None:
    """Raise on AlphaVantage's rate-limit / premium / error JSON payloads."""
    if not isinstance(payload, dict):
        return
    if "Note" in payload:
        raise RateLimited(f"alphavantage: {str(payload['Note'])[:160]}")
    if "Information" in payload:  # daily cap / premium endpoint
        raise RateLimited(f"alphavantage: {str(payload['Information'])[:200]}")
    if "Error Message" in payload:
        raise RuntimeError(f"alphavantage: {str(payload['Error Message'])[:200]}")


def _reshape(payload: dict) -> pd.DataFrame | None:
    """AlphaVantage ``TIME_SERIES_DAILY`` JSON → tidy OHLCV (index ascending)."""
    ts = payload.get("Time Series (Daily)") if isinstance(payload, dict) else None
    if not ts:
        return None
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns=_COLMAP)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].astype(float).dropna(how="all")


class AlphaVantageProvider(ApiProvider):
    name = "alphavantage"
    asset_classes = ("equity",)

    def available_fields(self) -> frozenset[str]:
        return TIERS["standard"]

    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set in .env (see .env.example).")

        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            payload = request_json(
                AV_BASE,
                # outputsize=full is premium → use compact (last ~100 bars) on free.
                {"function": "TIME_SERIES_DAILY", "symbol": sym,
                 "outputsize": "compact", "apikey": key},
                provider="alphavantage", min_interval=_MIN_INTERVAL,
            )
            _check_limits(payload)  # raises RateLimited / RuntimeError with a clear msg
            tidy = _reshape(payload)
            if tidy is not None and not tidy.empty:
                out[sym] = tidy
        return out
