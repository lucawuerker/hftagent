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

from quant_fund_agent.data.fields import (
    AV_EARNINGS_ACTUAL_KEYS,
    AV_EARNINGS_ESTIMATE_KEYS,
    AV_NET_INCOME_KEYS,
    AV_PERIOD_END_KEYS,
    AV_PROFILE_MAP,
    AV_REPORTED_DATE_KEYS,
    AV_REVENUE_KEYS,
    coerce_numeric,
    normalize,
    pick,
)
from quant_fund_agent.data.fundamentals import (
    DEFAULT_REPORTING_LAG_DAYS,
    availability_date,
    build_record_frame,
    parse_date,
)
from quant_fund_agent.data.providers._http import RateLimited, request_json
from quant_fund_agent.data.providers.base import ApiProvider
from quant_fund_agent.data.symbols import to_alphavantage
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.alphavantage")

AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 13.0  # ~5 requests/minute free tier

# Static-label availability sentinel (sector/industry are near-constant).
_STATIC_AVAILABILITY = pd.Timestamp("1990-01-01")

# Canonical non-OHLCV fields AV fills with a PIT-safe date (no leaky snapshot
# ratios — those have no historical date on the free tier).
_AV_NON_OHLCV = frozenset(
    {"sector", "industry", "revenue", "netMargin", "eps",
     "epsEstimate", "epsSurprise"}
)

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


# ── fundamentals: per-endpoint JSON → availability-stamped row dicts ──────────

def _av_profile_rows(payload) -> list[dict]:
    """COMPANY_OVERVIEW → one static-label row (sector/industry).

    The overview is an undated *current* snapshot, so only the near-constant
    labels are safe to backfill; its ratios would leak and are skipped.
    """
    if not isinstance(payload, dict):
        return []
    vals = normalize(payload, AV_PROFILE_MAP)
    return [{"availability": _STATIC_AVAILABILITY, **vals}] if vals else []


def _av_income_rows(payload, lag_days: int) -> list[dict]:
    """INCOME_STATEMENT quarterlyReports → revenue + derived netMargin."""
    reports = payload.get("quarterlyReports") if isinstance(payload, dict) else None
    out: list[dict] = []
    for rec in reports or []:
        avail = availability_date(
            None, pick(rec, AV_PERIOD_END_KEYS), reporting_lag_days=lag_days)
        if avail is None:
            continue
        revenue = coerce_numeric(pick(rec, AV_REVENUE_KEYS))
        net_income = coerce_numeric(pick(rec, AV_NET_INCOME_KEYS))
        vals: dict = {}
        if revenue is not None:
            vals["revenue"] = revenue
            if net_income is not None and revenue != 0:
                vals["netMargin"] = net_income / revenue
        if vals:
            out.append({"availability": avail, **vals})
    return out


def _av_earnings_rows(payload) -> list[dict]:
    """EARNINGS quarterlyEarnings → eps + estimate + surprise at the reportedDate."""
    quarterly = payload.get("quarterlyEarnings") if isinstance(payload, dict) else None
    out: list[dict] = []
    for rec in quarterly or []:
        avail = parse_date(pick(rec, AV_REPORTED_DATE_KEYS))
        if avail is None:
            continue
        actual = coerce_numeric(pick(rec, AV_EARNINGS_ACTUAL_KEYS))
        est = coerce_numeric(pick(rec, AV_EARNINGS_ESTIMATE_KEYS))
        vals: dict = {}
        if actual is not None:
            vals["eps"] = actual
        if est is not None:
            vals["epsEstimate"] = est
        if actual is not None and est is not None:
            vals["epsSurprise"] = actual - est
        if vals:
            out.append({"availability": avail, **vals})
    return out


class AlphaVantageProvider(ApiProvider):
    name = "alphavantage"
    asset_classes = ("equity", "crypto", "fx")

    def available_fields(self) -> frozenset[str]:
        # Fundamentals are equity-only; crypto/fx stay OHLCV (`standard`).
        if str(self.data.asset_class).lower() == "equity":
            return TIERS["standard"] | _AV_NON_OHLCV
        return TIERS["standard"]

    def _fetch_fundamentals(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set in .env (see .env.example).")
        lag = int(getattr(self.data, "reporting_lag_days", DEFAULT_REPORTING_LAG_DAYS))

        out: dict[str, pd.DataFrame] = {}
        for canonical in symbols:
            native = to_alphavantage(canonical, "equity")["symbol"]
            rows: list[dict] = []
            rows += _av_profile_rows(self._fund_get("OVERVIEW", native, key))
            rows += _av_income_rows(self._fund_get("INCOME_STATEMENT", native, key), lag)
            rows += _av_earnings_rows(self._fund_get("EARNINGS", native, key))
            frame = build_record_frame(rows)
            if frame is not None and not frame.empty:
                out[canonical] = frame
        return out

    def _fund_get(self, function: str, symbol: str, key: str):
        """One fundamental GET; rate limits propagate, other errors degrade."""
        try:
            payload = request_json(
                AV_BASE, {"function": function, "symbol": symbol, "apikey": key},
                provider="alphavantage", min_interval=_MIN_INTERVAL,
            )
            _check_limits(payload)  # raises RateLimited on Note/Information
            return payload
        except RateLimited:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("alphavantage: %s fetch failed for %s: %s", function, symbol, e)
            return {}

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
