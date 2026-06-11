"""AlphaVantage provider — keyed daily OHLCV (multi-asset).

Official REST API; needs ``ALPHAVANTAGE_API_KEY`` in ``.env``.  Supplies the
``standard`` tier (OHLCV); ``vwap``/``returns`` are synthesised by the panel.
By asset class it calls a different free-tier function:

  * equity — ``TIME_SERIES_DAILY`` (UNADJUSTED on the free tier);
  * crypto — ``DIGITAL_CURRENCY_DAILY`` (``symbol``+``market``, e.g. BTC/USD);
  * fx     — ``FX_DAILY`` (``from_symbol``+``to_symbol``) — **no volume** (filled NaN).

CAVEATS (free tier): unadjusted equity prices; ``outputsize=compact`` (~100 most
recent bars; ``full`` is premium); ~5 requests/minute and ~25/day.  For longer or
adjusted history prefer FMP or yfinance.  The shared HTTP helper throttles to one
call every ~13s and the parquet cache makes re-runs free.

Symbols are canonical and translated inside :meth:`_fetch`; results are keyed by
the canonical symbol.  Network is touched only there; the reshapes are unit-tested
offline with synthetic payloads.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from quant_fund_agent.data.providers._http import RateLimited, request_json
from quant_fund_agent.data.providers.base import ApiProvider
from quant_fund_agent.data.symbols import to_alphavantage
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.alphavantage")

AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 13.0  # ~5 requests/minute free tier

_OHLCV = ("open", "high", "low", "close", "volume")
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
    keep = [c for c in _OHLCV if c in df.columns]
    return df[keep].astype(float).dropna(how="all")


def _pick_ohlcv(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Build OHLCV by substring-matching AV's numbered keys.

    Robust to the differing/historic crypto field names (``"1. open"`` vs
    ``"1a. open (USD)"``) and to FX (no volume): for each target field, take the
    first raw column whose name contains that word.
    """
    cols: dict[str, pd.Series] = {}
    for field in _OHLCV:
        match = next((c for c in raw.columns if field in str(c).lower()), None)
        if match is not None:
            cols[field] = pd.to_numeric(raw[match], errors="coerce")
    if "close" not in cols:
        return None
    return pd.DataFrame(cols)


def _reshape_crypto(payload: dict) -> pd.DataFrame | None:
    """AlphaVantage ``DIGITAL_CURRENCY_DAILY`` JSON → tidy OHLCV (index ascending)."""
    ts = (payload.get("Time Series (Digital Currency Daily)")
          if isinstance(payload, dict) else None)
    if not ts:
        return None
    raw = pd.DataFrame.from_dict(ts, orient="index")
    raw.index = pd.to_datetime(raw.index)
    df = _pick_ohlcv(raw.sort_index())
    return df.dropna(how="all") if df is not None else None


def _reshape_fx(payload: dict) -> pd.DataFrame | None:
    """AlphaVantage ``FX_DAILY`` JSON → tidy OHLC (+ NaN volume; FX has none)."""
    ts = payload.get("Time Series FX (Daily)") if isinstance(payload, dict) else None
    if not ts:
        return None
    raw = pd.DataFrame.from_dict(ts, orient="index")
    raw.index = pd.to_datetime(raw.index)
    df = _pick_ohlcv(raw.sort_index())
    if df is None:
        return None
    if "volume" not in df.columns:
        df["volume"] = float("nan")
    return df.dropna(how="all")


class AlphaVantageProvider(ApiProvider):
    name = "alphavantage"
    asset_classes = ("equity", "crypto", "fx")

    def available_fields(self) -> frozenset[str]:
        return TIERS["standard"]

    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set in .env (see .env.example).")

        ac = str(self.data.asset_class).lower()
        out: dict[str, pd.DataFrame] = {}
        for canonical in symbols:
            params, reshape = self._request_spec(ac, canonical, key)
            payload = request_json(
                AV_BASE, params,
                provider="alphavantage", min_interval=_MIN_INTERVAL,
            )
            _check_limits(payload)  # raises RateLimited / RuntimeError with a clear msg
            tidy = reshape(payload)
            if tidy is not None and not tidy.empty:
                out[canonical] = tidy
        return out

    @staticmethod
    def _request_spec(asset_class: str, canonical: str, key: str):
        """(params, reshape_fn) for the AV function matching ``asset_class``."""
        native = to_alphavantage(canonical, asset_class)
        if asset_class == "crypto":
            return ({"function": "DIGITAL_CURRENCY_DAILY", "apikey": key, **native},
                    _reshape_crypto)
        if asset_class == "fx":
            return ({"function": "FX_DAILY", "outputsize": "compact",
                     "apikey": key, **native}, _reshape_fx)
        # equity — outputsize=full is premium → compact (last ~100 bars) on free.
        return ({"function": "TIME_SERIES_DAILY", "outputsize": "compact",
                 "apikey": key, **native}, _reshape)
