"""Convert raw LOBSTER message/orderbook CSVs → 10-second ``ticker_data`` bars.

Input (per ticker, per day, *no header* — LOBSTER native format)::

    message file : time_sec, event_type, order_id, size, price×1e4, direction[, null]
    orderbook    : AskP1,AskV1,BidP1,BidV1, AskP2,AskV2,BidP2,BidV2, …  (prices ×1e4)

The two files are **row-aligned**: row *i* of the orderbook is the book state
immediately *after* message *i*.  The orderbook width is ``4 * NumLevels`` so the
level count is inferred from the data — the same code handles level-3 (the GLD
sample on disk) and the level-5 pulls we download going forward.

Event types: 1=submit, 2=partial-cancel, 3=delete, 4=visible-exec,
5=hidden-exec, 6=cross/auction, 7=halt.  Direction: −1 sell-limit / +1 buy-limit
(execution of a *sell* limit ⇒ buyer-initiated trade).

Output (one row per 10-second bar, regular session ``09:30:00 … 15:59:50``)::

    date,time,stock,trade,orderFlow,hidden,auction,mid,midEnd,spread,effSpread,
    lobImb,effLobImb,trdLiq,ofLiq,depth,nbEvents,nbHidden,nbTrades,
    askPrice1,askDepth1,bidPrice1,bidDepth1, …, askPrice{N},…,bidDepth{N}

The 19 aggregate columns are followed by **per-level book columns** — for every
order-book level ``i`` the ask/bid price and displayed depth (volume) at that
level, as a bar-level event-time mean (same convention as ``spread``/``depth``).
The number of level columns is ``4 * NumLevels`` (inferred from the data), so a
level-5 file appends 20 columns and a level-10 file appends 40.

The exact column formulas are **reconstructed** (the original 2019 aggregator is
not in the repo) and documented in ``docs/lobster-ingestion/CONVERSION_SPEC.md``.

Memory: each day is processed independently; only that day's raw frame (~100–
300 MB) is held, then freed.  The per-day output is ~2340 tiny rows, so a whole
month accumulates to <50 k rows before being written.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("lobster_ingest.converter")

# ── session / grid constants ────────────────────────────────────────────────
SESSION_OPEN_SEC = 34_200          # 09:30:00
SESSION_CLOSE_SEC = 57_600         # 16:00:00
EARLY_CLOSE_SEC = 46_800           # 13:00:00 (US half-days)
BAR_SECONDS = 10
PRICE_SCALE = 10_000.0             # LOBSTER prices are dollars ×1e4
ASK_SENTINEL = 9_999_999_999       # empty ask level
BID_SENTINEL = -9_999_999_999      # empty bid level

# Exact target header (order matters — the loader maps by name).
OUTPUT_COLUMNS: tuple[str, ...] = (
    "date", "time", "stock",
    "trade", "orderFlow", "hidden", "auction",
    "mid", "midEnd", "spread", "effSpread",
    "lobImb", "effLobImb", "trdLiq", "ofLiq", "depth",
    "nbEvents", "nbHidden", "nbTrades",
)

# Columns that carry a book *state* (forward-fill across empty bars) vs. flow /
# count columns (zero-fill across empty bars).
_BOOK_COLS = ("mid", "midEnd", "spread", "lobImb", "effLobImb", "depth")
_FLOW_COLS = (
    "trade", "orderFlow", "hidden", "auction",
    "effSpread", "trdLiq", "ofLiq", "nbEvents", "nbHidden", "nbTrades",
)
_INT_COLS = (
    "trade", "orderFlow", "hidden", "auction",
    "trdLiq", "ofLiq", "nbEvents", "nbHidden", "nbTrades",
)


def level_field_names(n_levels: int) -> list[str]:
    """Per-level book column names for ``n_levels``, in raw-book order.

    For each level ``i`` (1-indexed) we emit the ask price, ask depth, bid price
    and bid depth — mirroring the raw orderbook layout ``AskP1,AskV1,BidP1,BidV1,…``.
    """
    cols: list[str] = []
    for i in range(1, n_levels + 1):
        cols += [f"askPrice{i}", f"askDepth{i}", f"bidPrice{i}", f"bidDepth{i}"]
    return cols


def output_columns(n_levels: int) -> list[str]:
    """Full output header for a given level count: the 19 aggregates + level cols."""
    return list(OUTPUT_COLUMNS) + level_field_names(n_levels)


# ── early-close handling (no hard calendar dependency) ──────────────────────
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """The ``n``-th ``weekday`` (Mon=0) of ``month``/``year`` (1-indexed)."""
    d = _dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + _dt.timedelta(days=offset + 7 * (n - 1))


def _computed_early_closes(year: int) -> set[_dt.date]:
    """US equity 1pm half-days, computed (no external calendar needed).

    The three recurring early closes: day after Thanksgiving (Fri after the 4th
    Thursday of Nov), Christmas Eve (Dec 24 when a weekday), and July 3 (when a
    weekday).  Good enough for bar-grid trimming; install
    ``pandas_market_calendars`` for the authoritative schedule (used when present).
    """
    out: set[_dt.date] = set()
    # Day after Thanksgiving.
    thanksgiving = _nth_weekday(year, 11, 3, 4)  # 4th Thursday (Thu=3)
    out.add(thanksgiving + _dt.timedelta(days=1))
    # Christmas Eve, if a weekday.
    xmas_eve = _dt.date(year, 12, 24)
    if xmas_eve.weekday() < 5:
        out.add(xmas_eve)
    # July 3, if a weekday.
    july3 = _dt.date(year, 7, 3)
    if july3.weekday() < 5:
        out.add(july3)
    return out


def session_close_sec(date: _dt.date) -> int:
    """Seconds-after-midnight of the regular-session close for ``date``.

    Prefers ``pandas_market_calendars`` (XNYS) when importable; otherwise falls
    back to the computed half-day set.  Returns 16:00 for a normal day, 13:00
    for a half-day.
    """
    try:  # authoritative path, only if the optional dep is installed
        import pandas_market_calendars as mcal  # type: ignore

        sched = mcal.get_calendar("XNYS").schedule(
            start_date=date.isoformat(), end_date=date.isoformat()
        )
        if not sched.empty:
            close = sched.iloc[0]["market_close"].tz_convert("America/New_York")
            return close.hour * 3600 + close.minute * 60 + close.second
    except Exception:  # pragma: no cover - optional dependency / network-free fallback
        pass
    if date in _computed_early_closes(date.year):
        return EARLY_CLOSE_SEC
    return SESSION_CLOSE_SEC


# ── raw-file discovery ──────────────────────────────────────────────────────
_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


@dataclass(frozen=True)
class DayFiles:
    """A matched message+orderbook pair for one trading day."""

    date: _dt.date
    message: Path
    orderbook: Path


def discover_day_files(raw_dir: str | Path) -> list[DayFiles]:
    """Find every ``(message, orderbook)`` day-pair under ``raw_dir``.

    LOBSTER names files ``{TICKER}_{YYYY-MM-DD}_{startms}_{endms}_message_{lvl}.csv``
    (and ``…_orderbook_{lvl}.csv``).  We pair them by replacing ``message`` with
    ``orderbook`` and extract the date from the filename.
    """
    raw_dir = Path(raw_dir)
    pairs: list[DayFiles] = []
    for msg in sorted(raw_dir.glob("*_message_*.csv")):
        ob = msg.with_name(msg.name.replace("_message_", "_orderbook_"))
        if not ob.exists():
            log.warning("no orderbook sibling for %s — skipping", msg.name)
            continue
        m = _DATE_RE.search(msg.name)
        if not m:
            log.warning("cannot parse date from %s — skipping", msg.name)
            continue
        date = _dt.date.fromisoformat(m.group(1))
        pairs.append(DayFiles(date=date, message=msg, orderbook=ob))
    return pairs


# ── readers ─────────────────────────────────────────────────────────────────
def _read_message(path: Path) -> pd.DataFrame:
    """Read the 6 canonical message columns (a stray trailing ``null`` is dropped)."""
    df = pd.read_csv(
        path,
        header=None,
        usecols=range(6),
        names=["time", "event", "order_id", "size", "price", "direction"],
        dtype={
            "time": "float64",
            "event": "int8",
            "order_id": "int64",
            "size": "int64",
            "price": "int64",
            "direction": "int8",
        },
    )
    return df


def _read_orderbook(path: Path) -> tuple[pd.DataFrame, int]:
    """Read the full LOBSTER order book; return ``(df, n_levels)``.

    Every level's price *and* volume column is read now that the converter also
    emits per-level price/depth bar columns (``askPrice{i}``/``askDepth{i}``/
    ``bidPrice{i}``/``bidDepth{i}``) alongside the aggregate book metrics.  The
    book width is ``4 * NumLevels`` so the level count is inferred from the data
    (level-3, level-5 and level-10 files all work).  One day is held at a time, so
    reading the deeper price columns too stays well within the per-day budget.
    """
    df = pd.read_csv(path, header=None)
    n_levels = df.shape[1] // 4
    return df, n_levels


# ── core conversion ─────────────────────────────────────────────────────────
def convert_day(
    message_path: str | Path,
    orderbook_path: str | Path,
    *,
    ticker: str,
    date: _dt.date,
) -> pd.DataFrame:
    """Convert one day's raw pair into a 2340-row (full-day) bar DataFrame.

    Returns a frame with exactly :data:`OUTPUT_COLUMNS`, sorted by time.  Raises
    ``ValueError`` if the two files are not row-aligned.
    """
    message_path, orderbook_path = Path(message_path), Path(orderbook_path)
    msg = _read_message(message_path)
    ob, n_levels = _read_orderbook(orderbook_path)
    if len(msg) != len(ob):
        raise ValueError(
            f"{message_path.name}: message rows ({len(msg)}) != orderbook rows "
            f"({len(ob)}) — files are not row-aligned"
        )

    close_sec = session_close_sec(date)
    level_cols = level_field_names(n_levels)

    # ── assemble a single working frame (row-aligned, positional join) ──
    ask = ob[0].where(ob[0] != ASK_SENTINEL) / PRICE_SCALE
    bid = ob[2].where(ob[2] != BID_SENTINEL) / PRICE_SCALE
    ask_v_tot = ob[[4 * i + 1 for i in range(n_levels)]].sum(axis=1)
    bid_v_tot = ob[[4 * i + 3 for i in range(n_levels)]].sum(axis=1)
    ask_v1, bid_v1 = ob[1], ob[3]

    w_data = {
        "time": msg["time"].to_numpy(),
        "event": msg["event"].to_numpy(),
        "size": msg["size"].to_numpy(),
        "price": msg["price"].to_numpy() / PRICE_SCALE,
        "dir": msg["direction"].to_numpy(),
        "mid_row": ((ask + bid) / 2).to_numpy(),
        "spread_row": (ask - bid).to_numpy(),
        "lobImb_row": _safe_imbalance(bid_v1, ask_v1),
        "effLobImb_row": _safe_imbalance(bid_v_tot, ask_v_tot),
        "depth_row": (ask_v_tot + bid_v_tot).to_numpy(),
    }
    # Per-level price (sentinel-masked → NaN) and displayed depth, one column per
    # level/side, mirroring the raw book layout AskP1,AskV1,BidP1,BidV1,…
    for i in range(1, n_levels + 1):
        a_p, a_v, b_p, b_v = 4 * (i - 1), 4 * (i - 1) + 1, 4 * (i - 1) + 2, 4 * (i - 1) + 3
        w_data[f"askPrice{i}_row"] = (ob[a_p].where(ob[a_p] != ASK_SENTINEL) / PRICE_SCALE).to_numpy()
        w_data[f"askDepth{i}_row"] = ob[a_v].to_numpy()
        w_data[f"bidPrice{i}_row"] = (ob[b_p].where(ob[b_p] != BID_SENTINEL) / PRICE_SCALE).to_numpy()
        w_data[f"bidDepth{i}_row"] = ob[b_v].to_numpy()
    w = pd.DataFrame(w_data)
    del msg, ob, ask, bid, ask_v_tot, bid_v_tot, ask_v1, bid_v1

    # ── restrict to the regular session and drop halt (type-7) rows ──
    in_session = (w["time"] >= SESSION_OPEN_SEC) & (w["time"] < close_sec)
    w = w[in_session & (w["event"] != 7)].copy()

    n_bars = (close_sec - SESSION_OPEN_SEC) // BAR_SECONDS
    if w.empty:
        return _empty_day(ticker, date, n_bars, level_cols)

    w["bar"] = ((w["time"] - SESSION_OPEN_SEC) // BAR_SECONDS).astype(int)

    # ── per-row event helpers (vectorised) ──
    ev, size, direction = w["event"].to_numpy(), w["size"].to_numpy(), w["dir"].to_numpy()
    price, midr = w["price"].to_numpy(), w["mid_row"].to_numpy()
    is_exec, is_hidden, is_auction = ev == 4, ev == 5, ev == 6
    is_sub, is_cxl = ev == 1, (ev == 2) | (ev == 3)

    w["signed_trade"] = np.where(is_exec, size * (-direction), 0)
    w["trd_gross"] = np.where(is_exec, size, 0)
    w["nb_trades"] = is_exec.astype(int)
    w["hidden_v"] = np.where(is_hidden, size, 0)
    w["nb_hidden"] = is_hidden.astype(int)
    w["auction_v"] = np.where(is_auction, size, 0)
    w["of_signed"] = np.where(is_sub, size * direction, 0) + np.where(is_cxl, -size * direction, 0)
    w["of_gross"] = np.where(is_sub | is_cxl, size, 0)
    w["eff_num"] = np.where(is_exec, size * 2.0 * np.abs(price - midr), 0.0)
    w["eff_den"] = w["trd_gross"]

    # ── aggregate to bars ──
    g = w.groupby("bar", sort=True)
    agg_spec: dict[str, tuple[str, str]] = {
        "mid": ("mid_row", "first"),
        "midEnd": ("mid_row", "last"),
        "spread": ("spread_row", "mean"),
        "lobImb": ("lobImb_row", "mean"),
        "effLobImb": ("effLobImb_row", "mean"),
        "depth": ("depth_row", "mean"),
        "trade": ("signed_trade", "sum"),
        "trdLiq": ("trd_gross", "sum"),
        "nbTrades": ("nb_trades", "sum"),
        "hidden": ("hidden_v", "sum"),
        "nbHidden": ("nb_hidden", "sum"),
        "auction": ("auction_v", "sum"),
        "orderFlow": ("of_signed", "sum"),
        "ofLiq": ("of_gross", "sum"),
        "nbEvents": ("mid_row", "size"),
        "_eff_num": ("eff_num", "sum"),
        "_eff_den": ("eff_den", "sum"),
    }
    # Per-level book columns: event-time mean over the bar (same convention as the
    # aggregate book metrics spread/depth/imbalances).
    for name in level_cols:
        agg_spec[name] = (f"{name}_row", "mean")
    agg = g.agg(**agg_spec)
    agg["effSpread"] = np.where(agg["_eff_den"] > 0, agg["_eff_num"] / agg["_eff_den"], 0.0)
    agg = agg.drop(columns=["_eff_num", "_eff_den"])

    # ── reindex onto the full bar grid; fill book vs. flow columns ──
    agg = agg.reindex(pd.RangeIndex(0, n_bars))
    book_cols = list(_BOOK_COLS) + level_cols
    agg[book_cols] = agg[book_cols].ffill().bfill()
    agg[list(_FLOW_COLS)] = agg[list(_FLOW_COLS)].fillna(0)

    return _finalise(agg, ticker, date, n_bars, level_cols)


def _safe_imbalance(bid_v: pd.Series, ask_v: pd.Series) -> np.ndarray:
    """(bid − ask)/(bid + ask) ∈ [−1, 1]; NaN where both queues are empty."""
    bid_v, ask_v = bid_v.to_numpy(dtype="float64"), ask_v.to_numpy(dtype="float64")
    total = bid_v + ask_v
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, (bid_v - ask_v) / total, np.nan)
    return out


def _bar_time_strings(n_bars: int) -> list[str]:
    secs = SESSION_OPEN_SEC + np.arange(n_bars) * BAR_SECONDS
    return [f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}" for s in secs]


def _finalise(
    agg: pd.DataFrame,
    ticker: str,
    date: _dt.date,
    n_bars: int,
    level_cols: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Attach date/time/stock, round, integer-cast counts, order columns.

    ``level_cols`` are the per-level price/depth columns to append after the 19
    aggregates (empty for callers that don't emit them).
    """
    level_cols = list(level_cols)
    out = pd.DataFrame(index=agg.index)
    out["date"] = date.isoformat()
    out["time"] = _bar_time_strings(n_bars)
    out["stock"] = ticker
    for col in (
        "trade", "orderFlow", "hidden", "auction", "mid", "midEnd", "spread",
        "effSpread", "lobImb", "effLobImb", "trdLiq", "ofLiq", "depth",
        "nbEvents", "nbHidden", "nbTrades", *level_cols,
    ):
        out[col] = agg[col].to_numpy()

    # Round prices/ratios to a sensible precision; cast share-count columns to int.
    # Per-level prices and depths are bar means → keep them as rounded floats.
    for col in ("mid", "midEnd", "spread", "depth", *level_cols):
        out[col] = out[col].round(4)
    for col in ("effSpread", "lobImb", "effLobImb"):
        out[col] = out[col].round(7)
    for col in _INT_COLS:
        # Raw conversion fills every flow column, so they integer-cast cleanly.
        # The sampled-book product has no messages → these stay NaN (written as
        # empty cells), so only cast when the whole column is present.
        out[col] = out[col].round()
        if out[col].notna().all():
            out[col] = out[col].astype("int64")

    return out[list(OUTPUT_COLUMNS) + level_cols]


def _empty_day(
    ticker: str,
    date: _dt.date,
    n_bars: int,
    level_cols: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """A day with no in-session events — emit a NaN/zero grid (rare; keeps shape)."""
    agg = pd.DataFrame(index=pd.RangeIndex(0, n_bars))
    for col in (*_BOOK_COLS, *level_cols):
        agg[col] = np.nan
    for col in _FLOW_COLS:
        agg[col] = 0
    return _finalise(agg, ticker, date, n_bars, level_cols)


# ── folder → monthly CSVs ────────────────────────────────────────────────────
@dataclass
class ConvertSummary:
    ticker: str
    days: int = 0
    rows: int = 0
    months: list[str] = field(default_factory=list)
    out_files: list[Path] = field(default_factory=list)


# ── sampled-book product (10s order-book snapshots, no messages) ────────────
@dataclass(frozen=True)
class SampledFile:
    """A single-day sampled-order-book CSV (``time,ask_1,…,bid_size_N``)."""

    date: _dt.date
    path: Path


def _is_sampled_header(path: Path) -> bool:
    cols = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    return "time" in cols and "ask_1" in cols and "bid_1" in cols


def discover_sampled_files(raw_dir: str | Path) -> list[SampledFile]:
    """Find single-day sampled-order-book CSVs under ``raw_dir`` (one file/day).

    These are LOBSTER's *sampled* product (e.g. ``CORN_2024-06-03_…_10_10.csv``):
    a header row + one book snapshot per 10 seconds, with **no** message/event
    data — so only the book-derived columns can be produced.
    """
    raw_dir = Path(raw_dir)
    out: list[SampledFile] = []
    for csv in sorted(raw_dir.glob("*.csv")):
        if "_message_" in csv.name or "_orderbook_" in csv.name:
            continue
        m = _DATE_RE.search(csv.name)
        if not m:
            continue
        try:
            if not _is_sampled_header(csv):
                continue
        except Exception:
            continue
        out.append(SampledFile(date=_dt.date.fromisoformat(m.group(1)), path=csv))
    return out


def convert_sampled_day(path: str | Path, *, ticker: str, date: _dt.date) -> pd.DataFrame:
    """Convert one sampled-order-book day → the 19-column bar frame.

    Book columns (``mid, midEnd, spread, lobImb, effLobImb, depth``) are derived
    from the snapshots; flow/trade columns are left **NaN** (no message data).
    ``midEnd`` of a bar is the *next* snapshot's mid (start of the next bar).
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    n_levels = (df.shape[1] - 1) // 4
    level_cols = level_field_names(n_levels)
    close_sec = session_close_sec(date)
    n_bars = (close_sec - SESSION_OPEN_SEC) // BAR_SECONDS

    sec = pd.to_timedelta(df["time"]).dt.total_seconds().astype(int)
    ask1 = df["ask_1"].where(df["ask_1"] != ASK_SENTINEL) / PRICE_SCALE
    bid1 = df["bid_1"].where(df["bid_1"] != BID_SENTINEL) / PRICE_SCALE
    ask_v_tot = df[[f"ask_size_{i}" for i in range(1, n_levels + 1)]].sum(axis=1)
    bid_v_tot = df[[f"bid_size_{i}" for i in range(1, n_levels + 1)]].sum(axis=1)

    snap_data = {
        "sec": sec,
        "mid": (ask1 + bid1) / 2,
        "spread": ask1 - bid1,
        "lobImb": _safe_imbalance(df["bid_size_1"], df["ask_size_1"]),
        "effLobImb": _safe_imbalance(bid_v_tot, ask_v_tot),
        "depth": (ask_v_tot + bid_v_tot).to_numpy(),
    }
    # Per-level price (sentinel-masked) + displayed depth from the snapshot.
    for i in range(1, n_levels + 1):
        snap_data[f"askPrice{i}"] = df[f"ask_{i}"].where(df[f"ask_{i}"] != ASK_SENTINEL) / PRICE_SCALE
        snap_data[f"askDepth{i}"] = df[f"ask_size_{i}"]
        snap_data[f"bidPrice{i}"] = df[f"bid_{i}"].where(df[f"bid_{i}"] != BID_SENTINEL) / PRICE_SCALE
        snap_data[f"bidDepth{i}"] = df[f"bid_size_{i}"]
    snap = pd.DataFrame(snap_data)
    snap = snap[(snap["sec"] >= SESSION_OPEN_SEC) & (snap["sec"] < close_sec)].copy()
    snap["bar"] = ((snap["sec"] - SESSION_OPEN_SEC) // BAR_SECONDS).astype(int)
    snap = snap.drop_duplicates("bar", keep="first").set_index("bar").reindex(pd.RangeIndex(0, n_bars))

    agg = pd.DataFrame(index=pd.RangeIndex(0, n_bars))
    agg["mid"] = snap["mid"]
    agg["midEnd"] = snap["mid"].shift(-1)
    for col in ("spread", "lobImb", "effLobImb", "depth", *level_cols):
        agg[col] = snap[col]
    book_cols = list(_BOOK_COLS) + level_cols
    agg[book_cols] = agg[book_cols].ffill().bfill()
    for col in _FLOW_COLS:
        agg[col] = np.nan

    return _finalise(agg, ticker, date, n_bars, level_cols)


# ── folder → monthly CSVs (auto-detects raw vs. sampled product) ─────────────
def _write_months(
    by_month: dict[str, list[pd.DataFrame]],
    ticker_dir: Path,
    merge_existing: bool,
    summary: ConvertSummary,
) -> None:
    for month, frames in sorted(by_month.items()):
        path = ticker_dir / f"bin{month}.csv"
        df = pd.concat(frames, ignore_index=True)
        if merge_existing and path.exists():
            prev = pd.read_csv(path)
            df = pd.concat([prev, df], ignore_index=True)
        df = (
            df.drop_duplicates(subset=["date", "time"], keep="last")
            .sort_values(["date", "time"])
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        summary.months.append(month)
        summary.out_files.append(path)
        log.info("wrote %s (%d rows)", path, len(df))


def convert_archive(
    archive: str | Path,
    *,
    ticker: str,
    out_dir: str | Path = "ticker_data",
    merge_existing: bool = True,
    workdir: str | Path | None = None,
) -> ConvertSummary:
    """Stream a LOBSTER ``.7z`` → monthly CSVs, **one day at a time**.

    A 2-year raw archive is small compressed but huge unzipped, so we never
    extract the whole thing: entries are grouped by date and each day's file(s)
    are extracted to a temp dir, converted, then deleted — peak uncompressed
    footprint is a single day.  Auto-detects raw (message+orderbook) vs. sampled
    (single file/day).
    """
    import shutil
    import tempfile
    from collections import defaultdict

    import py7zr

    archive = Path(archive)
    with py7zr.SevenZipFile(archive, "r") as z:
        names = z.getnames()

    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        base = n.split("/")[-1]
        if not base.endswith(".csv"):
            continue
        m = _DATE_RE.search(base)
        if m:
            groups[m.group(1)].append(n)
    if not groups:
        log.warning("archive %s has no dated CSV day files", archive.name)
        return ConvertSummary(ticker=ticker)

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lobster_arx_"))
    by_month: dict[str, list[pd.DataFrame]] = {}
    summary = ConvertSummary(ticker=ticker)
    try:
        for date_str in sorted(groups):
            date = _dt.date.fromisoformat(date_str)
            day_dir = work / date_str
            day_dir.mkdir(parents=True, exist_ok=True)
            with py7zr.SevenZipFile(archive, "r") as z:
                z.extract(path=day_dir, targets=groups[date_str])

            msg = list(day_dir.rglob("*_message_*.csv"))
            if msg:
                ob = list(day_dir.rglob("*_orderbook_*.csv"))
                bars = convert_day(msg[0], ob[0], ticker=ticker, date=date)
            else:
                csvs = [p for p in day_dir.rglob("*.csv")]
                bars = convert_sampled_day(csvs[0], ticker=ticker, date=date)
            by_month.setdefault(f"{date.year}{date.month:02d}", []).append(bars)
            summary.days += 1
            summary.rows += len(bars)
            shutil.rmtree(day_dir, ignore_errors=True)  # free the day immediately
    finally:
        if workdir is None:
            shutil.rmtree(work, ignore_errors=True)

    out_dir = Path(out_dir)
    ticker_dir = out_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    _write_months(by_month, ticker_dir, merge_existing, summary)
    log.info("archive %s → %s: %d days, %d rows", archive.name, ticker, summary.days, summary.rows)
    return summary


def convert_folder(
    raw_dir: str | Path,
    *,
    ticker: str,
    out_dir: str | Path = "ticker_data",
    merge_existing: bool = True,
) -> ConvertSummary:
    """Convert every day in ``raw_dir`` → ``out_dir/{ticker}/bin{YYYYMM}.csv``.

    Auto-detects the LOBSTER product: **raw** message+orderbook pairs (full 19
    columns) or the **sampled** 10s order-book (book columns only, flow columns
    empty).  Days are grouped by calendar month, de-duplicated on ``(date,time)``
    so re-running a chunk is idempotent.
    """
    out_dir = Path(out_dir)
    ticker_dir = out_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    summary = ConvertSummary(ticker=ticker)
    by_month: dict[str, list[pd.DataFrame]] = {}

    days = discover_day_files(raw_dir)
    if days:
        for day in days:
            log.info("converting (raw) %s %s", ticker, day.date.isoformat())
            bars = convert_day(day.message, day.orderbook, ticker=ticker, date=day.date)
            by_month.setdefault(f"{day.date.year}{day.date.month:02d}", []).append(bars)
            summary.days += 1
            summary.rows += len(bars)
        _write_months(by_month, ticker_dir, merge_existing, summary)
        return summary

    sampled = discover_sampled_files(raw_dir)
    if sampled:
        log.warning(
            "no raw message/orderbook pairs under %s; found %d SAMPLED order-book "
            "files — only book columns will be populated (flow/trade columns empty)",
            raw_dir, len(sampled),
        )
        for sf in sampled:
            log.info("converting (sampled) %s %s", ticker, sf.date.isoformat())
            bars = convert_sampled_day(sf.path, ticker=ticker, date=sf.date)
            by_month.setdefault(f"{sf.date.year}{sf.date.month:02d}", []).append(bars)
            summary.days += 1
            summary.rows += len(bars)
        _write_months(by_month, ticker_dir, merge_existing, summary)
        return summary

    log.warning("no LOBSTER day files (raw or sampled) found under %s", raw_dir)
    return summary
