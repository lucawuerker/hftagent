"""Financial Modeling Prep (FMP) provider — keyed daily OHLCV.

Official REST API; needs ``FMP_API_KEY`` in ``.env``.  Supplies the ``standard``
tier (OHLCV); ``vwap``/``returns`` are synthesised by the panel layer.

Uses FMP's current **stable** API (the legacy ``/api/v3`` endpoints return 403
for keys issued after Aug 2025).  The ``historical-price-eod/dividend-adjusted``
endpoint returns split/dividend-**adjusted** OHLC directly (``adjOpen`` …
``adjClose``), so prices are consistent across corporate actions with no manual
back-adjustment.

Network is touched only in :meth:`_fetch`, so the JSON reshape is unit-tested
offline with synthetic payloads.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from quant_fund_agent.data.providers._http import RateLimited, request_json
from quant_fund_agent.data.providers.base import ApiProvider
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.fmp")

FMP_BASE = "https://financialmodelingprep.com/stable"
_MIN_INTERVAL = 0.25  # generous free tier; small spacing is plenty.

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
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].astype(float).dropna(how="all")


class FMPProvider(ApiProvider):
    name = "fmp"
    asset_classes = ("equity",)

    def available_fields(self) -> frozenset[str]:
        return TIERS["standard"]

    def _fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise ValueError("FMP_API_KEY not set in .env (see .env.example).")

        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                payload = request_json(
                    f"{FMP_BASE}/historical-price-eod/dividend-adjusted",
                    {"symbol": sym, "from": self.data.start, "to": self.data.end,
                     "apikey": key},
                    provider="fmp", min_interval=_MIN_INTERVAL,
                )
            except RateLimited:
                raise
            except Exception as e:  # noqa: BLE001 — one bad symbol must not abort
                log.warning("fmp: fetch failed for %s: %s", sym, e)
                continue
            # FMP signals errors with an "Error Message" dict instead of an array.
            if isinstance(payload, dict) and payload.get("Error Message"):
                raise RuntimeError(f"fmp: {payload['Error Message']}")
            tidy = _reshape(payload)
            if tidy is not None and not tidy.empty:
                out[sym] = tidy
        return out
