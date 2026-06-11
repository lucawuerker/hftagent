"""Canonical symbol handling for multi-asset universes.

Each vendor names crypto and FX pairs differently (yfinance ``BTC-USD`` /
``EURUSD=X``; FMP ``BTCUSD`` / ``EURUSD``; AlphaVantage splits a pair into
``symbol``+``market`` for crypto and ``from_symbol``+``to_symbol`` for FX).  To
keep the universe presets, the parquet cache and every downstream agent
provider-agnostic, the project uses **one canonical symbol** per instrument and
translates to the vendor's form only inside that provider's ``_fetch``:

  * equity — the plain ticker, e.g. ``AAPL`` (translated as-is, never parsed as a
    pair, so dotted/dashed tickers like ``BRK-B`` are safe);
  * crypto / fx — a ``BASE-QUOTE`` pair, e.g. ``BTC-USD``, ``EUR-USD``.

Providers return their per-symbol frames keyed by the **canonical** symbol, so the
wide panel columns and cache paths are identical regardless of vendor.
"""

from __future__ import annotations


def split_pair(symbol: str) -> tuple[str, str]:
    """``'BTC-USD' -> ('BTC', 'USD')``.

    Raises ``ValueError`` for anything that isn't a ``BASE-QUOTE`` pair — only
    called for crypto/fx symbols, never for plain equity tickers.
    """
    parts = str(symbol).upper().split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Expected a canonical BASE-QUOTE pair like 'BTC-USD', got {symbol!r}."
        )
    return parts[0], parts[1]


def to_yfinance(symbol: str, asset_class: str) -> str:
    """Canonical → Yahoo symbol (``BTC-USD`` stays; ``EUR-USD`` → ``EURUSD=X``)."""
    ac = str(asset_class).lower()
    if ac == "crypto":
        return str(symbol).upper()                 # Yahoo crypto is already BASE-USD
    if ac == "fx":
        base, quote = split_pair(symbol)
        return f"{base}{quote}=X"
    return symbol                                   # equity: passthrough


def to_fmp(symbol: str, asset_class: str) -> str:
    """Canonical → FMP symbol (``BTC-USD`` → ``BTCUSD``; ``EUR-USD`` → ``EURUSD``)."""
    ac = str(asset_class).lower()
    if ac in ("crypto", "fx"):
        base, quote = split_pair(symbol)
        return f"{base}{quote}"
    return symbol                                   # equity: passthrough


def to_alphavantage(symbol: str, asset_class: str) -> dict[str, str]:
    """Canonical → AlphaVantage request params (the shape differs by asset class).

    crypto → ``{"symbol": "BTC", "market": "USD"}`` (DIGITAL_CURRENCY_DAILY);
    fx     → ``{"from_symbol": "EUR", "to_symbol": "USD"}`` (FX_DAILY);
    equity → ``{"symbol": "AAPL"}`` (TIME_SERIES_DAILY).
    """
    ac = str(asset_class).lower()
    if ac == "crypto":
        base, quote = split_pair(symbol)
        return {"symbol": base, "market": quote}
    if ac == "fx":
        base, quote = split_pair(symbol)
        return {"from_symbol": base, "to_symbol": quote}
    return {"symbol": str(symbol)}
