"""Financial Modeling Prep (FMP) provider — keyed daily OHLCV (multi-asset).

Official REST API; needs ``FMP_API_KEY`` in ``.env``.  Supplies the ``standard``
tier (OHLCV); ``vwap``/``returns`` are synthesised by the panel layer.

Uses FMP's current **stable** API (the legacy ``/api/v3`` endpoints return 403
for keys issued after Aug 2025).  By asset class:

  * equity — ``historical-price-eod/dividend-adjusted`` → split/dividend-**adjusted**
    OHLC (``adjOpen`` … ``adjClose``), consistent across corporate actions with no
    manual back-adjustment;
  * crypto / fx — ``historical-price-eod/full`` → **raw** OHLCV (no corporate
    actions for these), with the dash-stripped pair symbol (``BTC-USD`` → ``BTCUSD``).

Symbols are canonical and translated inside :meth:`_fetch`; results are keyed by
the canonical symbol.  Network is touched only there, so both JSON reshapes are
unit-tested offline with synthetic payloads.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from quant_fund_agent.data.fields import (
    FMP_EARNINGS_ACTUAL_KEYS,
    FMP_EARNINGS_ESTIMATE_KEYS,
    FMP_FILING_DATE_KEYS,
    FMP_INCOME_MAP,
    FMP_METRICS_MAP,
    FMP_PERIOD_END_KEYS,
    FMP_PROFILE_MAP,
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
from quant_fund_agent.data.symbols import to_fmp
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.fmp")

FMP_BASE = "https://financialmodelingprep.com/stable"
_MIN_INTERVAL = 0.25  # generous free tier; small spacing is plenty.

# Static-label availability sentinel: profile sector/industry are (near-)constant,
# so they're knowable throughout any backtest window.
_STATIC_AVAILABILITY = pd.Timestamp("1990-01-01")

# Canonical non-OHLCV fields FMP actually fills (drives gating; only advertise
# what we deliver).
_FMP_NON_OHLCV = frozenset(
    {
        "sector", "industry", "marketCap",
        "peRatio", "pbRatio", "psRatio", "roe", "roic",
        "debtToEquity", "currentRatio", "grossMargin", "netMargin",
        "revenue", "eps", "freeCashFlow",
        "epsEstimate", "revenueEstimate", "epsSurprise",
    }
)

_OHLCV = ("open", "high", "low", "close", "volume")
_COLMAP = {
    "adjOpen": "open", "adjHigh": "high", "adjLow": "low",
    "adjClose": "close", "volume": "volume",
}


def _reshape(payload: list | dict) -> pd.DataFrame | None:
    """FMP ``dividend-adjusted`` JSON array → tidy adjusted OHLCV (index ascending)."""
    rows = payload if isinstance(payload, list) else None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "date" not in df or "adjClose" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().rename(columns=_COLMAP)
    keep = [c for c in _OHLCV if c in df.columns]
    return df[keep].astype(float).dropna(how="all")


def _reshape_raw(payload: list | dict) -> pd.DataFrame | None:
    """FMP ``historical-price-eod/full`` array → tidy RAW OHLCV (crypto/fx)."""
    rows = payload if isinstance(payload, list) else None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "date" not in df or "close" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in _OHLCV if c in df.columns]
    return df[keep].astype(float).dropna(how="all")


# ── fundamentals: per-endpoint JSON → availability-stamped row dicts ──────────

def _as_rows(payload) -> list[dict]:
    """FMP fundamental endpoints return a JSON array of records (or an error dict)."""
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(f"fmp: {payload['Error Message']}")
    return payload if isinstance(payload, list) else []


def _profile_rows(payload) -> list[dict]:
    """`profile` → one static-label row (sector/industry), knowable throughout."""
    rows = _as_rows(payload)
    if not rows:
        return []
    vals = normalize(rows[0], FMP_PROFILE_MAP)
    return [{"availability": _STATIC_AVAILABILITY, **vals}] if vals else []


def _metric_rows(payload, lag_days: int) -> list[dict]:
    """`key-metrics`/`ratios` (quarterly) → availability = filing date else end+lag."""
    out: list[dict] = []
    for rec in _as_rows(payload):
        avail = availability_date(
            pick(rec, FMP_FILING_DATE_KEYS), pick(rec, FMP_PERIOD_END_KEYS),
            reporting_lag_days=lag_days,
        )
        vals = normalize(rec, FMP_METRICS_MAP)
        if avail is not None and vals:
            out.append({"availability": avail, **vals})
    return out


def _income_rows(payload, lag_days: int) -> list[dict]:
    """`income-statement` (quarterly) → revenue/eps/netMargin, filing-stamped."""
    out: list[dict] = []
    for rec in _as_rows(payload):
        avail = availability_date(
            pick(rec, FMP_FILING_DATE_KEYS), pick(rec, FMP_PERIOD_END_KEYS),
            reporting_lag_days=lag_days,
        )
        vals = normalize(rec, FMP_INCOME_MAP)
        if avail is not None and vals:
            out.append({"availability": avail, **vals})
    return out


def _earnings_rows(payload) -> list[dict]:
    """`earnings` → estimate + surprise, stamped at the (PIT-safe) report date.

    ``epsSurprise = actual − estimate`` is only knowable at the report, so the
    report ``date`` is the correct availability for both the estimate and the
    surprise (a conservative, non-leaking choice).
    """
    out: list[dict] = []
    for rec in _as_rows(payload):
        avail = parse_date(pick(rec, ("date",)))
        if avail is None:
            continue
        actual = coerce_numeric(pick(rec, FMP_EARNINGS_ACTUAL_KEYS))
        est = coerce_numeric(pick(rec, FMP_EARNINGS_ESTIMATE_KEYS))
        rev_est = coerce_numeric(pick(rec, ("revenueEstimated", "estimatedRevenue")))
        vals: dict = {}
        if est is not None:
            vals["epsEstimate"] = est
        if rev_est is not None:
            vals["revenueEstimate"] = rev_est
        if actual is not None and est is not None:
            vals["epsSurprise"] = actual - est
        if vals:
            out.append({"availability": avail, **vals})
    return out


class FMPProvider(ApiProvider):
    name = "fmp"
    asset_classes = ("equity", "crypto", "fx")

    def available_fields(self) -> frozenset[str]:
        # Fundamentals are equity-only; crypto/fx stay OHLCV (`standard`).
        if str(self.data.asset_class).lower() == "equity":
            return TIERS["standard"] | _FMP_NON_OHLCV
        return TIERS["standard"]

    def _fetch_fundamentals(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise ValueError("FMP_API_KEY not set in .env (see .env.example).")
        lag = int(getattr(self.data, "reporting_lag_days", DEFAULT_REPORTING_LAG_DAYS))

        out: dict[str, pd.DataFrame] = {}
        for canonical in symbols:
            native = to_fmp(canonical, "equity")
            rows: list[dict] = []
            rows += _profile_rows(self._fund_get("profile", native, key))
            rows += _metric_rows(
                self._fund_get("key-metrics", native, key, period="quarter"), lag)
            rows += _metric_rows(
                self._fund_get("ratios", native, key, period="quarter"), lag)
            rows += _income_rows(
                self._fund_get("income-statement", native, key, period="quarter"), lag)
            rows += _earnings_rows(self._fund_get("earnings", native, key))
            frame = build_record_frame(rows)
            if frame is not None and not frame.empty:
                out[canonical] = frame
        return out

    def _fund_get(self, endpoint: str, native: str, key: str, **params):
        """One fundamental GET; a single bad endpoint must not lose the symbol."""
        q = {"symbol": native, "apikey": key, "limit": 40, **params}
        try:
            return request_json(f"{FMP_BASE}/{endpoint}", q,
                                provider="fmp", min_interval=_MIN_INTERVAL)
        except RateLimited:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("fmp: %s fetch failed for %s: %s", endpoint, native, e)
            return []

    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise ValueError("FMP_API_KEY not set in .env (see .env.example).")

        ac = str(self.data.asset_class).lower()
        if ac == "equity":
            endpoint, reshape = "historical-price-eod/dividend-adjusted", _reshape
        else:  # crypto / fx — raw OHLCV, no corporate actions
            endpoint, reshape = "historical-price-eod/full", _reshape_raw

        out: dict[str, pd.DataFrame] = {}
        for canonical in symbols:
            native = to_fmp(canonical, ac)
            try:
                payload = request_json(
                    f"{FMP_BASE}/{endpoint}",
                    {"symbol": native, "from": self.data.start, "to": self.data.end,
                     "apikey": key},
                    provider="fmp", min_interval=_MIN_INTERVAL,
                )
            except RateLimited:
                raise
            except Exception as e:  # noqa: BLE001 — one bad symbol must not abort
                log.warning("fmp: fetch failed for %s: %s", native, e)
                continue
            # FMP signals errors with an "Error Message" dict instead of an array.
            if isinstance(payload, dict) and payload.get("Error Message"):
                raise RuntimeError(f"fmp: {payload['Error Message']}")
            tidy = reshape(payload)
            if tidy is not None and not tidy.empty:
                out[canonical] = tidy
        return out
