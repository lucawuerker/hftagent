"""Load 10-second LOBSTER tick data into the OHLCV panel format factors expect.

The raw CSV columns are::

    date, time, stock, trade, orderFlow, hidden, auction, mid, midEnd,
    spread, effSpread, lobImb, effLobImb, trdLiq, ofLiq, depth,
    nbEvents, nbHidden, nbTrades

We construct OHLCV-equivalent fields per bar from the midprice:

    open   = mid        (start-of-bar midprice)
    high   = max(mid, midEnd)
    low    = min(mid, midEnd)
    close  = midEnd     (end-of-bar midprice)
    volume = abs(trade)  (unsigned traded volume proxy)
    spread, lobImb, depth are also carried through as extra fields.

Returns ``dict[str, pd.DataFrame]`` where each DataFrame has:
    index  = pd.DatetimeIndex  (one row per 10-sec bar)
    columns = tickers
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ticker_csv(path: Path) -> pd.DataFrame:
    """Read a single ticker CSV and return a DatetimeIndex'd DataFrame."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("datetime").sort_index()
    return df


def load_panel(
    data_dir: str | Path,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all tickers into an OHLCV panel dict.

    Args:
        data_dir: path to the ``ticker_data/`` root.
        tickers: explicit list of tickers, or ``None`` to auto-discover.

    Returns:
        dict with keys ``"open", "high", "low", "close", "volume",
        "spread", "lobImb", "depth"`` each being a DataFrame with
        DatetimeIndex rows and ticker columns.
    """
    data_dir = Path(data_dir)
    if tickers is None:
        tickers = sorted(
            d.name for d in data_dir.iterdir() if d.is_dir()
        )

    frames: dict[str, dict[str, pd.Series]] = {
        "open": {},
        "high": {},
        "low": {},
        "close": {},
        "volume": {},
        "spread": {},
        "lobImb": {},
        "depth": {},
    }

    for ticker in tickers:
        csv_files = list((data_dir / ticker).glob("*.csv"))
        if not csv_files:
            continue
        df = pd.concat([load_ticker_csv(f) for f in csv_files]).sort_index()

        frames["open"][ticker] = df["mid"]
        frames["high"][ticker] = df[["mid", "midEnd"]].max(axis=1)
        frames["low"][ticker] = df[["mid", "midEnd"]].min(axis=1)
        frames["close"][ticker] = df["midEnd"]
        frames["volume"][ticker] = df["trade"].abs()
        frames["spread"][ticker] = df["spread"]
        frames["lobImb"][ticker] = df["lobImb"]
        frames["depth"][ticker] = df["depth"]

    panel: dict[str, pd.DataFrame] = {}
    for field, series_dict in frames.items():
        panel[field] = pd.DataFrame(series_dict)

    return panel


def forward_returns(
    close: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Compute forward returns *horizon* bars ahead.

    ``ret[t] = close[t+horizon] / close[t] - 1``
    """
    return close.shift(-horizon) / close.replace(0, float("nan")) - 1.0
