"""Local parquet cache for vendor market data.

API providers (yfinance / FMP / …) fetch through :func:`cached_fetch`, which
stores one parquet per ``(provider, asset_class, freq, symbol)`` under the
configured ``cache_dir`` (default ``data/market/``).  Re-runs read the cache;
only symbols whose cached range doesn't cover the request hit the network.  This
amortises rate limits and makes a run reproducible.

Design (v1, deliberately simple): **refetch-on-miss** — if a symbol's cached
range doesn't cover ``[start, end]`` (within a small frequency-scaled tolerance
for weekends/holidays), the whole range is refetched and unioned with the cache.
Fine-grained gap-filling is a future optimisation.

Look-ahead invariant: this layer only changes *fetching*.  Slicing by ``as_of`` /
``cutoff`` happens downstream (``modeling/service._truncate_as_of``), untouched.

The provider supplies a ``fetch_fn(symbols) -> {symbol: tidy DataFrame}`` where
each tidy frame is ``DatetimeIndex`` × OHLCV columns.  :func:`cached_fetch`
returns the wide field panel ``{field: DataFrame(index × symbols)}``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger("data.cache")

FetchFn = Callable[[list[str]], dict[str, pd.DataFrame]]

# Slow-moving non-OHLCV groups (fundamentals/estimates/events) refresh on a long
# TTL rather than per-bar coverage — see :func:`cached_records`.
DEFAULT_RECORDS_TTL_DAYS = 90


def _coverage_tolerance(freq: str) -> pd.Timedelta:
    """Slack allowed between the cache's last bar and the requested end.

    Absorbs weekends/holidays (daily) so a fully-cached past range isn't
    needlessly refetched just because the last bar predates a non-trading ``end``.
    """
    f = (freq or "1d").lower()
    if f.endswith("d") or f.endswith("wk") or f.endswith("mo"):
        return pd.Timedelta(days=4)
    return pd.Timedelta(hours=12)  # intraday


def _symbol_path(cache_dir, provider, asset_class, freq, symbol) -> Path:
    safe = str(symbol).replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / provider / asset_class / freq / f"{safe}.parquet"


def _read_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:  # corrupt / partial write → treat as a miss
        log.warning("cache read failed (%s); refetching: %s", path, e)
        return None


def _write_cached(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)  # atomic on the same filesystem


def _covers(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp,
            tol: pd.Timedelta) -> bool:
    if df is None or df.empty:
        return False
    return df.index.min() <= start and df.index.max() >= (end - tol)


def _union(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _assemble(per_symbol: dict[str, pd.DataFrame], symbols: list[str],
              start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Build wide ``{field: DataFrame(index × symbols)}`` from per-symbol frames."""
    cols: dict[str, dict[str, pd.Series]] = {}
    for sym in symbols:
        df = per_symbol.get(sym)
        if df is None or df.empty:
            continue
        window = df.loc[(df.index >= start) & (df.index <= end)]
        for field in window.columns:
            cols.setdefault(field, {})[sym] = window[field]
    return {field: pd.DataFrame(series) for field, series in cols.items() if series}


def cached_fetch(
    provider: str,
    symbols: list[str],
    start: str,
    end: str,
    freq: str,
    asset_class: str,
    cache_dir: str,
    fetch_fn: FetchFn,
) -> dict[str, pd.DataFrame]:
    """Return the wide field panel for ``symbols`` over ``[start, end]``.

    Reads each symbol's parquet; symbols whose cached range doesn't cover the
    request are (re)fetched via ``fetch_fn`` and the result is cached.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    tol = _coverage_tolerance(freq)

    per_symbol: dict[str, pd.DataFrame] = {}
    to_fetch: list[tuple[str, Path, pd.DataFrame | None]] = []
    for sym in symbols:
        path = _symbol_path(cache_dir, provider, asset_class, freq, sym)
        cached = _read_cached(path)
        if _covers(cached, start_ts, end_ts, tol):
            per_symbol[sym] = cached
        else:
            to_fetch.append((sym, path, cached))

    if to_fetch:
        log.info("cache: %d/%d symbols hit, fetching %d from %s",
                 len(per_symbol), len(symbols), len(to_fetch), provider)
        fetched = fetch_fn([s for s, _, _ in to_fetch]) or {}
        for sym, path, cached in to_fetch:
            new = fetched.get(sym)
            if new is None or new.empty:
                if cached is not None:
                    per_symbol[sym] = cached  # keep what we had
                continue
            new = new.copy()
            new.index = pd.to_datetime(new.index)
            merged = _union(cached, new)
            try:
                _write_cached(path, merged)
            except Exception as e:  # caching is best-effort; never fail the load
                log.warning("cache write failed (%s): %s", path, e)
            per_symbol[sym] = merged

    return _assemble(per_symbol, symbols, start_ts, end_ts)


# ── slow-moving record cache (fundamentals / estimates / events) ─────────────

def _records_path(cache_dir, provider, asset_class, group, symbol) -> Path:
    """Parquet path for a non-OHLCV ``group`` — a sibling of the price cache.

    ``group`` (``fundamentals``/``estimates``/…) sits where the price cache's
    ``freq`` ("1d") sits, and is never a freq value, so the OHLCV cache tree is
    untouched: ``<cache_dir>/<provider>/<asset_class>/<group>/<symbol>.parquet``.
    """
    safe = str(symbol).replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / provider / asset_class / group / f"{safe}.parquet"


def _is_fresh(path: Path, ttl: pd.Timedelta) -> bool:
    """True if ``path`` exists and was written within ``ttl`` (file mtime)."""
    if not path.exists():
        return False
    try:
        age_s = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_s <= ttl.total_seconds()


def cached_records(
    provider: str,
    symbols: list[str],
    asset_class: str,
    group: str,
    cache_dir: str,
    fetch_fn: FetchFn,
    *,
    ttl_days: int = DEFAULT_RECORDS_TTL_DAYS,
) -> dict[str, pd.DataFrame]:
    """Cache per-symbol **record** frames (availability-indexed, not OHLCV bars).

    Unlike :func:`cached_fetch`, this returns the *per-symbol* frames (the caller
    aligns them onto the price index later) and refreshes on a **TTL** — point
    events have no contiguous "coverage" to check, so a file written within
    ``ttl_days`` is reused; a stale/absent one is refetched and unioned with the
    cache.  Caching is best-effort and never fails the load.
    """
    ttl = pd.Timedelta(days=int(ttl_days))
    out: dict[str, pd.DataFrame] = {}
    to_fetch: list[tuple[str, Path, pd.DataFrame | None]] = []
    for sym in symbols:
        path = _records_path(cache_dir, provider, asset_class, group, sym)
        cached = _read_cached(path)
        if cached is not None and _is_fresh(path, ttl):
            out[sym] = cached
        else:
            to_fetch.append((sym, path, cached))

    if to_fetch:
        log.info("records cache (%s): %d/%d fresh, fetching %d from %s",
                 group, len(out), len(symbols), len(to_fetch), provider)
        fetched = fetch_fn([s for s, _, _ in to_fetch]) or {}
        for sym, path, cached in to_fetch:
            new = fetched.get(sym)
            if new is None or new.empty:
                if cached is not None:
                    out[sym] = cached  # serve stale rather than nothing
                continue
            new = new.copy()
            new.index = pd.to_datetime(new.index)
            merged = _union(cached, new)
            try:
                _write_cached(path, merged)
            except Exception as e:  # best-effort
                log.warning("records cache write failed (%s): %s", path, e)
            out[sym] = merged

    return out
