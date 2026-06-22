"""Point-in-time index membership — survivorship-bias-free universes.

A *static* ticker list (e.g. ``universes/sp100.txt``) applied across a multi-year
backtest is **survivorship-biased**: it over-represents names that survived to the
snapshot date and silently omits every company that was an index member but later
merged, was acquired, or went bankrupt.  This module makes index membership
**time-varying and point-in-time (PIT)**, mirroring the availability→forward-fill
discipline of :mod:`quant_fund_agent.data.fundamentals`.

Canonical artifact
------------------
``data/universes/membership/<index>.csv`` — a **membership-interval table**, one
row per contiguous spell a ticker is a constituent::

    ticker,name,start_date,end_date,add_reason,remove_reason,source

A ticker may have several disjoint spells (left and rejoined).  ``end_date`` is
**exclusive** — the first day the name is *no longer* a member (the index
effective date); an empty ``end_date`` means "still a member".  So a name is a
constituent on ``[start_date, end_date)``.  Only ``ticker``/``start_date``/
``end_date`` are load-bearing for the mask; ``name``/``*_reason``/``source`` are
provenance/context.

The table is built from **free public sources** by
``scripts/build_sp500_membership.py``; see
``docs/data-layer/SP500_MEMBERSHIP.md`` to reproduce it end to end.

How it plugs in
---------------
:func:`apply_membership_mask` builds a per-bar boolean mask and ``NaN``-outs every
``(date, ticker)`` cell where the ticker was *not* a member.  It is applied once
at panel load (:func:`quant_fund_agent.data.load_panel`, when
``DataSettings.membership`` is set), so every downstream consumer — factor
research, the Architect/Statistician, the walk-forward trade loop and the model
comparison harness — is survivorship-correct with **no per-call effort**.  Because
the masked panel rides the same ``DatetimeIndex`` as prices, cross-sectional ops
(rank-IC, z-score, ``indneutralize``) drop non-members automatically (pandas skips
``NaN``), and positions on non-members are ``NaN``/0.

Tickers-only on the free path
-----------------------------
With a key-free provider (yfinance) a *delisted* name has no fetchable price
column, so it is simply absent from the panel — the residual survivorship bias the
build report **quantifies**.  Premium providers (CRSP/FMP) that serve delisted
prices close that gap; :class:`MembershipSource` below is the seam where a premium
*membership* source would plug in (the query API here is source-agnostic — it just
reads the canonical CSV).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("data.membership")

MEMBERSHIP_DIR = Path(__file__).parent / "universes" / "membership"
_REQUIRED_COLS = ("ticker", "start_date", "end_date")
# Far-future sentinel for still-active spells (well inside pd.Timestamp range).
_OPEN_END = pd.Timestamp("2999-12-31")


# ── canonical-table access ───────────────────────────────────────────────────

def membership_path(index: str = "sp500") -> Path:
    """Path to the canonical membership CSV for ``index``."""
    return MEMBERSHIP_DIR / f"{index}.csv"


def available_indices() -> list[str]:
    """Names of indices with a built membership table (sans ``.csv``)."""
    if not MEMBERSHIP_DIR.exists():
        return []
    return sorted(p.stem for p in MEMBERSHIP_DIR.glob("*.csv"))


@lru_cache(maxsize=None)
def load_membership(index: str = "sp500") -> pd.DataFrame:
    """Read + validate the membership-interval table for ``index`` (cached).

    Returns a frame with ``ticker`` (upper-cased), ``start_date`` /``end_date``
    (``datetime64``; ``end_date`` is ``NaT`` for still-active spells) and any
    provenance columns present.  Raises ``FileNotFoundError`` with a build hint if
    the artifact is missing.  Call ``load_membership.cache_clear()`` after
    rebuilding the CSV in the same process (tests do this).
    """
    path = membership_path(index)
    if not path.exists():
        raise FileNotFoundError(
            f"No membership table for index {index!r} at {path}. "
            "Build it with `./venv/bin/python scripts/build_sp500_membership.py` "
            "(see docs/data-layer/SP500_MEMBERSHIP.md)."
        )
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns {missing} (has {list(df.columns)}).")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.normalize()
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.normalize()
    df = df[df["ticker"].astype(bool) & df["start_date"].notna()].reset_index(drop=True)
    return df


# ── point-in-time queries ────────────────────────────────────────────────────

def members_as_of(date: Any, index: str = "sp500") -> list[str]:
    """Tickers that are constituents of ``index`` on ``date`` (``[start, end)``)."""
    df = load_membership(index)
    ts = pd.Timestamp(date).normalize()
    end = df["end_date"].fillna(_OPEN_END)
    active = (df["start_date"] <= ts) & (ts < end)
    return sorted(df.loc[active, "ticker"].unique())


def union_members(
    start: Any | None = None, end: Any | None = None, index: str = "sp500"
) -> list[str]:
    """Every ticker that was *ever* a member of ``index`` in ``[start, end)``.

    This is the set the panel must **fetch data for** (so names that later left the
    index still have price history while they were members).  ``None`` bounds mean
    "all of recorded history".
    """
    df = load_membership(index)
    if start is None and end is None:
        return sorted(df["ticker"].unique())
    s = pd.Timestamp(start).normalize() if start is not None else pd.Timestamp.min
    e = pd.Timestamp(end).normalize() if end is not None else _OPEN_END
    spell_end = df["end_date"].fillna(_OPEN_END)
    # interval [start_date, end_date) overlaps the window [s, e)
    overlap = (df["start_date"] < e) & (spell_end > s)
    return sorted(df.loc[overlap, "ticker"].unique())


def membership_mask(
    dates: Any, symbols: list[str], index: str = "sp500"
) -> pd.DataFrame:
    """Per-bar boolean membership mask aligned to a panel (``dates`` × ``symbols``).

    ``True`` where the symbol was a constituent on that date.  Same wide-frame
    contract as :func:`fundamentals.align_to_index` so it composes with the panel.
    """
    df = load_membership(index)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    symbols = list(symbols)
    mask = pd.DataFrame(False, index=dates, columns=symbols)
    dvals = dates.values
    by_ticker = df.groupby("ticker")
    for ticker in symbols:
        if ticker not in by_ticker.groups:
            continue
        col = np.zeros(len(dates), dtype=bool)
        for _, row in by_ticker.get_group(ticker).iterrows():
            s = np.datetime64(row["start_date"])
            if pd.isna(row["end_date"]):
                col |= dvals >= s
            else:
                e = np.datetime64(row["end_date"])
                col |= (dvals >= s) & (dvals < e)
        mask[ticker] = col
    return mask


def apply_membership_mask(
    panel: dict[str, pd.DataFrame], index: str = "sp500"
) -> dict[str, pd.DataFrame]:
    """``NaN``-out non-member ``(date, ticker)`` cells across every field.

    Returns a new panel dict.  If the membership artifact is missing this is a
    logged no-op (the data layer must never crash because a universe table hasn't
    been built) — callers that require PIT should check :func:`available_indices`.
    """
    if not panel:
        return panel
    ref = next(iter(panel.values()))
    try:
        mask = membership_mask(ref.index, list(ref.columns), index)
    except FileNotFoundError as e:
        log.warning("membership %r unavailable (%s); panel left un-masked.", index, e)
        return panel
    out: dict[str, pd.DataFrame] = {}
    for field, frame in panel.items():
        m = mask.reindex(index=frame.index, columns=frame.columns, fill_value=False)
        out[field] = frame.where(m)
    return out


# ── source abstraction (free now; premium documented) ────────────────────────

class MembershipSource(ABC):
    """Where the canonical membership-interval table comes from.

    The query API above is source-agnostic — it reads the built CSV.  This seam
    documents *how that CSV is produced* and lets a premium vendor (which also
    serves delisted prices) replace the free reconstruction without touching any
    consumer.
    """

    name: str = ""

    @abstractmethod
    def intervals(self, index: str) -> pd.DataFrame:
        """Return the canonical interval table for ``index``."""
        ...


class PublicReconstructionSource(MembershipSource):
    """Free public reconstruction (Wikipedia + GitHub ``fja05680/sp500``).

    The reconstruction/reconciliation logic lives in
    ``scripts/build_sp500_membership.py`` (it needs network + raw-snapshot I/O);
    this class loads the canonical CSV that script writes.
    """

    name = "public_reconstruction"

    def intervals(self, index: str) -> pd.DataFrame:
        return load_membership(index)


class CrspSource(MembershipSource):  # pragma: no cover - premium stub
    """PIT constituents from WRDS/CRSP (stable PERMNOs + delisted securities)."""

    name = "crsp"

    def intervals(self, index: str) -> pd.DataFrame:
        raise NotImplementedError(
            "CRSP membership requires WRDS access — premium path. "
            "See the 'Premium extension' section of docs/data-layer/SP500_MEMBERSHIP.md."
        )


class FmpSource(MembershipSource):  # pragma: no cover - premium stub
    """PIT constituents from FMP's historical-constituents endpoint (paid tier)."""

    name = "fmp"

    def intervals(self, index: str) -> pd.DataFrame:
        raise NotImplementedError(
            "FMP historical constituents need a paid key — premium path. "
            "See the 'Premium extension' section of docs/data-layer/SP500_MEMBERSHIP.md."
        )
