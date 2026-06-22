"""Load 10-second LOBSTER tick data into the field panel that factors consume.

The raw CSV produced by the LOBSTER aggregator has these columns
(per ticker, per 10-second bar)::

    date, time, stock, trade, orderFlow, hidden, auction, mid, midEnd,
    spread, effSpread, lobImb, effLobImb, trdLiq, ofLiq, depth,
    nbEvents, nbHidden, nbTrades

To stay compatible with the WorldQuant-style alpha library we expose
an OHLCV view built from ``mid`` / ``midEnd``:

    open    = mid        (start-of-bar mid-price)
    high    = max(mid, midEnd)
    low     = min(mid, midEnd)
    close   = midEnd     (end-of-bar mid-price)
    volume  = abs(trade) (unsigned traded volume — legacy alias)

In addition, **every** raw LOBSTER column is forwarded into the panel
under its own field name so that factors can build microstructure
signals directly off them (signed order flow, queue depth, effective
spread, event counts, etc.):

    trade, orderFlow, hidden, auction,
    spread, effSpread, lobImb, effLobImb,
    trdLiq, ofLiq, depth,
    nbEvents, nbHidden, nbTrades

When the converter emitted per-level order-book columns
(``askPrice{i}``/``askDepth{i}``/``bidPrice{i}``/``bidDepth{i}`` — one set per
book level), those are forwarded too.  Their count depends on the level of the
pull, so on a full load they are **discovered from the CSV header** rather than
hard-coded; on a targeted ``fields=[...]`` load any name matching that pattern is
accepted and read.

Robustness
----------
``ticker_data/`` is auto-discovered, but the same directory tree often
contains sibling folders that are NOT per-symbol tickers — the raw
LOBSTER aggregator dumps (``fillSamples/``, ``binSamples/``) stack
*every* symbol into one multi-million-row file.  ``binSamples/`` is
especially dangerous: it shares the exact per-ticker column schema, so
a naive ``mid``/``midEnd`` check waves it through.  Ingesting it as a
single fake "ticker" and then aligning its 25M-row, heavily-overlapping
``DatetimeIndex`` against the real tickers explodes the panel to tens of
GB and gets the process OOM-killed.  The loader defends in three cheap
layers, all run on the *first* CSV **before** any full read:

1. Header-only required-column check: a ticker's first CSV must contain
   ``mid`` and ``midEnd``, else it is skipped — this rejects
   ``fillSamples/`` without parsing a single data row.
2. Single-symbol identity check: a genuine ticker directory holds
   exactly one symbol in its ``stock`` column; a dump holds many.  Any
   directory whose first CSV spans >1 symbol is skipped — this rejects
   ``binSamples/`` before its giant index can be loaded or aligned.
3. Per-ticker try/except: any unhandled error (parse failure, schema
   surprise) is caught, logged, and the ticker is skipped — one bad
   CSV cannot abort the whole load.

Returns ``dict[str, pd.DataFrame]`` where each DataFrame has:
    index   = pd.DatetimeIndex  (one row per 10-sec bar)
    columns = tickers (only those that loaded successfully)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("backtesting.data_loader")

# Per-level order-book columns the LOBSTER converter appends after the 19
# aggregates (``askPrice1``/``askDepth1``/``bidPrice1``/``bidDepth1``, …).  Their
# count depends on the level of the pull, so they are discovered from the CSV
# header rather than hard-coded — any column matching this pattern is forwarded
# into the panel under its own field name.
_LEVEL_FIELD_RE = re.compile(r"^(?:askPrice|bidPrice|askDepth|bidDepth)\d+$")


def _is_level_field(name: str) -> bool:
    return bool(_LEVEL_FIELD_RE.match(name))


def _level_sort_key(name: str) -> tuple[int, int]:
    """Order level fields by (level, side) for deterministic panel layout."""
    side, level = re.match(r"^([a-zA-Z]+?)(\d+)$", name).groups()
    rank = {"askPrice": 0, "askDepth": 1, "bidPrice": 2, "bidDepth": 3}.get(side, 9)
    return (int(level), rank)

# Columns a CSV must contain for us to derive the OHLCV view.  Other
# LOBSTER columns are optional — missing ones just produce NaN for that
# ticker, the panel still loads.
_REQUIRED_COLUMNS: tuple[str, ...] = ("mid", "midEnd")

# Always-needed CSV columns regardless of which fields are requested
# (used to build the datetime index).
_ALWAYS_NEEDED_CSV_COLS: tuple[str, ...] = ("date", "time")


# Raw CSV columns that map straight through to a panel field of the
# same name.  ``mid`` and ``midEnd`` are handled specially below
# (they back the OHLC view).
PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "trade",
    "orderFlow",
    "hidden",
    "auction",
    "spread",
    "effSpread",
    "lobImb",
    "effLobImb",
    "trdLiq",
    "ofLiq",
    "depth",
    "nbEvents",
    "nbHidden",
    "nbTrades",
)

# OHLCV-style fields synthesised from the raw columns.  Kept as a
# separate list so we don't accidentally overwrite a passthrough field
# of the same name.
SYNTHESISED_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

ALL_FIELDS: tuple[str, ...] = SYNTHESISED_FIELDS + PASSTHROUGH_FIELDS


def _csv_columns_needed(requested_fields: tuple[str, ...]) -> set[str]:
    """The minimum set of raw CSV columns required to satisfy ``requested_fields``.

    Mapping the *output* field set back to the *input* column set lets
    us pass ``usecols=`` to ``pd.read_csv`` and avoid loading columns
    we'd just throw away.  Empirically this is ~3× cheaper in memory
    when only a handful of fields are needed.
    """
    needed: set[str] = set(_ALWAYS_NEEDED_CSV_COLS)
    needed.update(_REQUIRED_COLUMNS)  # mid + midEnd are the OHLC backbone
    for f in requested_fields:
        if f == "volume":
            needed.add("trade")
        elif f in PASSTHROUGH_FIELDS or _is_level_field(f):
            needed.add(f)  # passthrough + per-level fields are CSV columns by name
        # synthesised OHLC fields (open/high/low/close) need only mid+midEnd,
        # which are already in ``needed`` above.
    return needed


def load_ticker_csv(
    path: Path,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Read a single ticker CSV and return a DatetimeIndex'd DataFrame.

    Passing ``usecols`` restricts the read to the columns we actually
    need, which is a large memory win when the panel only consumes a
    handful of fields out of the 19 raw LOBSTER columns.

    Note: pandas raises ``ValueError`` if a name in ``usecols`` is
    missing, so we intersect with the actual header first.
    """
    if usecols is not None:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in usecols if c in header]
    df = pd.read_csv(path, usecols=usecols)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("datetime").sort_index()
    return df


def _downcast_to_float32(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Halve panel memory by downcasting numeric columns to float32.

    Float32 has ~7 significant decimal digits which is plenty for IC
    estimation on prices in the ~100 USD range — well above the typical
    spread.  We skip this for fields the caller might want as integers,
    but in practice every panel field ends up numeric.
    """
    for field, df in panel.items():
        # numeric_cols may include the entire DataFrame when everything
        # came in as float64 already.
        num_cols = df.select_dtypes(include=[np.floating, np.integer]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].astype(np.float32)
    return panel


def _discover_level_fields(
    data_dir: Path, tickers: list[str]
) -> list[str]:
    """Per-level book columns present in the tickers' CSVs (header-only scan).

    The level count varies by pull, so on a full (``fields=None``) load we read
    each ticker's first CSV header and collect any ``askPrice{i}``/``askDepth{i}``/
    ``bidPrice{i}``/``bidDepth{i}`` columns it carries.  Cheap (one header read per
    ticker) and tolerant — a directory we can't read just contributes nothing.
    """
    found: set[str] = set()
    for ticker in tickers:
        csv_files = sorted((data_dir / ticker).glob("*.csv"))
        if not csv_files:
            continue
        try:
            header = pd.read_csv(csv_files[0], nrows=0).columns
        except Exception:
            continue
        found.update(c for c in header if _is_level_field(c))
    return sorted(found, key=_level_sort_key)


def load_panel(
    data_dir: str | Path,
    tickers: list[str] | None = None,
    fields: list[str] | None = None,
    n_tickers: int | None = None,
    dtype: str = "float32",
) -> dict[str, pd.DataFrame]:
    """Load all tickers into a wide field-panel dict.

    Args:
        data_dir: path to the ``ticker_data/`` root.
        tickers:  explicit list of tickers.  ``None`` auto-discovers
                  every subdirectory under ``data_dir``.
        fields:   subset of ``ALL_FIELDS`` to materialise; ``None`` means
                  load every field.  Strongly recommended to pass an
                  explicit list — the per-CSV ``usecols`` filter is
                  derived from this set.
        n_tickers: if set, cap the universe at this many tickers (taken
                  alphabetically from the resolved ticker list).  Useful
                  for memory-constrained research sessions; the full
                  universe stays available for the final 1-year backtest.
        dtype:    ``"float32"`` (default) downcasts numerics for ~50%
                  memory cut; ``"float64"`` preserves full precision.

    Returns:
        dict with one entry per requested field.  Every value is a
        ``pd.DataFrame`` with a ``DatetimeIndex`` and tickers in the
        columns.
    """
    data_dir = Path(data_dir)
    if tickers is None:
        tickers = sorted(
            d.name for d in data_dir.iterdir() if d.is_dir()
        )
    if n_tickers is not None and len(tickers) > n_tickers:
        log.info("capping universe at %d tickers (was %d)",
                 n_tickers, len(tickers))
        tickers = tickers[:n_tickers]

    if fields is not None:
        requested = tuple(fields)
        unknown = {f for f in requested if f not in ALL_FIELDS and not _is_level_field(f)}
        if unknown:
            raise ValueError(f"Unknown fields requested: {sorted(unknown)}")
    else:
        # Full load: every known field plus whatever per-level book columns the
        # CSVs actually contain (the level count depends on the LOBSTER pull).
        requested = ALL_FIELDS + tuple(_discover_level_fields(data_dir, tickers))

    # Minimum CSV columns we need to satisfy ``requested``.  This is the
    # big memory win — without it every ticker CSV is read in full (19
    # columns) even if we only emit 3 panel fields.
    csv_cols = sorted(_csv_columns_needed(requested))

    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in requested}

    skipped: list[tuple[str, str]] = []
    loaded: list[str] = []
    for ticker in tickers:
        csv_files = sorted((data_dir / ticker).glob("*.csv"))
        if not csv_files:
            skipped.append((ticker, "no csv files"))
            continue

        # Cheap pre-checks on the FIRST CSV, run *before* the full read so
        # a mis-schema'd or multi-symbol sibling folder can never be parsed
        # in bulk or aligned into the panel (see module docstring).  A
        # genuine ticker dir is tiny here; the aggregator dumps are caught
        # on a single-file scan instead of a 25M-row concat + alignment.
        try:
            header = pd.read_csv(csv_files[0], nrows=0).columns
            missing = [c for c in _REQUIRED_COLUMNS if c not in header]
            if missing:
                skipped.append((ticker, f"missing required column(s): {missing}"))
                continue
            if "stock" in header:
                n_symbols = pd.read_csv(
                    csv_files[0], usecols=["stock"]
                )["stock"].nunique(dropna=True)
                if n_symbols != 1:
                    skipped.append(
                        (ticker, f"not a single-symbol ticker dir "
                                 f"({n_symbols} symbols — likely an aggregator dump)")
                    )
                    continue
        except Exception as e:
            skipped.append((ticker, f"pre-check failed: {type(e).__name__}: {e}"))
            continue

        # Wrap the whole per-ticker block: any parse error / schema
        # surprise must be survivable.  A bad CSV is a warning, not a crash.
        try:
            df = pd.concat(
                [load_ticker_csv(f, usecols=csv_cols) for f in csv_files]
            ).sort_index()

            missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                skipped.append((ticker, f"missing required column(s): {missing}"))
                continue

            if "open" in frames:
                frames["open"][ticker] = df["mid"]
            if "high" in frames:
                frames["high"][ticker] = df[["mid", "midEnd"]].max(axis=1)
            if "low" in frames:
                frames["low"][ticker] = df[["mid", "midEnd"]].min(axis=1)
            if "close" in frames:
                frames["close"][ticker] = df["midEnd"]
            if "volume" in frames and "trade" in df.columns:
                frames["volume"][ticker] = df["trade"].abs()

            for f in PASSTHROUGH_FIELDS:
                if f in frames and f in df.columns:
                    frames[f][ticker] = df[f]

            # Per-level book columns (variable count) pass straight through too.
            for f in requested:
                if _is_level_field(f) and f in frames and f in df.columns:
                    frames[f][ticker] = df[f]

            # df has served its purpose; drop the reference so peak
            # memory across the per-ticker loop stays bounded by the
            # accumulating ``frames``, not by raw CSVs hanging around.
            del df
            loaded.append(ticker)
        except Exception as e:
            skipped.append((ticker, f"{type(e).__name__}: {e}"))
            continue

    if skipped:
        for t, reason in skipped:
            log.warning("skipped %s: %s", t, reason)
    log.info("loaded %d tickers, skipped %d", len(loaded), len(skipped))

    panel: dict[str, pd.DataFrame] = {}
    for field, series_dict in frames.items():
        panel[field] = pd.DataFrame(series_dict)

    if dtype == "float32":
        panel = _downcast_to_float32(panel)
    return panel


def forward_returns(
    close: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Compute forward returns *horizon* bars ahead.

    ``ret[t] = close[t+horizon] / close[t] - 1``
    """
    return close.shift(-horizon) / close.replace(0, float("nan")) - 1.0
