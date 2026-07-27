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
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    symbols = list(symbols)
    mask = pd.DataFrame(False, index=dates, columns=symbols)  # preserve caller index
    # Compare on tz-naive, day-normalized values (spell bounds are tz-naive dates),
    # while the mask keeps the caller's exact index so it re-aligns to the panel.
    cmp = dates.tz_localize(None) if dates.tz is not None else dates
    dvals = cmp.normalize().values
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


# ── interval algebra (shared by every builder) ───────────────────────────────
#
# These operate on a *spells* frame (``ticker``/``start_date``/``end_date``, end
# exclusive, ``NaT`` = still a member) before it is written to CSV, so the free
# public reconstruction (``scripts/build_sp500_membership.py``), the FMP-native
# builder (``data/fmp_ingest/constituents.py``) and any future vendor source all
# audit and reconcile through exactly one implementation.

def normalize_ticker(ticker: str) -> str:
    """Canonicalize to the repo convention (``BRK.B`` → ``BRK-B``)."""
    return str(ticker).strip().upper().replace(".", "-")


def coalesce_spells(spells: pd.DataFrame) -> pd.DataFrame:
    """Merge each ticker's touching/overlapping spells into single spells."""
    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for ticker, sub in spells.groupby("ticker"):
        sub = sub.sort_values("start_date")
        cur_s: pd.Timestamp | None = None
        cur_e: pd.Timestamp | None = None
        for _, r in sub.iterrows():
            s = r["start_date"]
            e = r["end_date"] if pd.notna(r["end_date"]) else _OPEN_END
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:  # touching/overlapping → extend
                cur_e = max(cur_e, e)
            else:
                out.append((ticker, cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((ticker, cur_s, cur_e))
    df = pd.DataFrame(out, columns=["ticker", "start_date", "end_date"])
    df["end_date"] = df["end_date"].replace(_OPEN_END, pd.NaT)
    return df


def members_from_spells(spells: pd.DataFrame, date: Any) -> set[str]:
    """Constituent set on ``date`` from an in-memory spells frame."""
    ts = pd.Timestamp(date).normalize()
    end = spells["end_date"].fillna(_OPEN_END)
    active = (spells["start_date"] <= ts) & (ts < end)
    return set(spells.loc[active, "ticker"])


def audit_spells(
    spells: pd.DataFrame,
    since: Any,
    *,
    band: tuple[int, int] = (475, 515),
    until: Any | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Structural + count invariants.  Returns ``(errors, month-end counts)``.

    ``band`` is the plausible constituent-count range for the index (the S&P 500
    briefly runs 500–505 with share classes / pending spin-offs; a Nasdaq-100
    table should pass ``(95, 115)``).  A table that leaves the band is almost
    always a change-log parsing bug, not a real index event.
    """
    errors: list[str] = []
    lo, hi = band

    bad = spells[spells["end_date"].notna() & (spells["end_date"] <= spells["start_date"])]
    if len(bad):
        errors.append(f"{len(bad)} spell(s) with end_date <= start_date")

    for ticker, sub in spells.groupby("ticker"):
        sub = sub.sort_values("start_date")
        ends = sub["end_date"].fillna(_OPEN_END).tolist()
        starts = sub["start_date"].tolist()
        if any(starts[i] < ends[i - 1] for i in range(1, len(starts))):
            errors.append(f"overlapping spells for {ticker}")

    stop = pd.Timestamp(until).normalize() if until is not None else pd.Timestamp.today().normalize()
    months = pd.date_range(pd.Timestamp(since).normalize(), stop, freq="ME")
    counts = pd.DataFrame(
        [(m, len(members_from_spells(spells, m))) for m in months], columns=["date", "n"]
    )
    if len(counts):
        out_of_band = counts[(counts["n"] < lo) | (counts["n"] > hi)]
        if len(out_of_band):
            errors.append(
                f"{len(out_of_band)}/{len(counts)} month-ends outside [{lo},{hi}] "
                f"(min {counts['n'].min()}, max {counts['n'].max()})"
            )
    return errors, counts


def compare_spells(
    primary: pd.DataFrame,
    other: pd.DataFrame,
    since: Any,
    *,
    until: Any | None = None,
    label_a: str = "primary",
    label_b: str = "other",
) -> pd.DataFrame:
    """Month-end Jaccard agreement between two interval tables.

    The independent ground truth for a membership table is its audit invariants,
    not source agreement — but a per-month Jaccard makes *where* two
    reconstructions diverge (usually the oldest years, where change logs thin
    out) immediately visible instead of averaging it away.
    """
    stop = pd.Timestamp(until).normalize() if until is not None else pd.Timestamp.today().normalize()
    rows = []
    for m in pd.date_range(pd.Timestamp(since).normalize(), stop, freq="ME"):
        a = members_from_spells(primary, m)
        b = members_from_spells(other, m)
        if not (a or b):
            continue
        union = len(a | b)
        rows.append((
            m, len(a), len(b), (len(a & b) / union) if union else 1.0,
            ",".join(sorted(a - b)[:8]), ",".join(sorted(b - a)[:8]),
        ))
    return pd.DataFrame(
        rows,
        columns=["date", f"n_{label_a}", f"n_{label_b}", "jaccard",
                 f"only_{label_a}", f"only_{label_b}"],
    )


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


class FmpSource(MembershipSource):
    """PIT constituents reconstructed from FMP's change log (paid tier).

    Reads the **archived** payloads under ``data/vendor/fmp/index/`` (written by
    ``scripts/fmp_bulk_download.py --groups index``) and replays them into the
    canonical interval table — so this class never touches the network and stays
    reproducible offline, exactly like the free reconstruction it replaces.
    Building the canonical CSV from it is ``scripts/build_fmp_membership.py``.
    """

    name = "fmp"

    def __init__(self, archive_root: Any | None = None) -> None:
        from quant_fund_agent.data.fmp_ingest.store import DEFAULT_ROOT

        self.archive_root = Path(archive_root) if archive_root else DEFAULT_ROOT

    def intervals(self, index: str, *, since: Any = "2004-01-01") -> pd.DataFrame:
        from quant_fund_agent.data.fmp_ingest.constituents import (
            INDEX_ENDPOINTS,
            build_intervals,
        )
        from quant_fund_agent.data.fmp_ingest.store import Archive

        if index not in INDEX_ENDPOINTS:
            raise ValueError(
                f"No FMP constituent endpoints for index {index!r}. "
                f"Known: {sorted(INDEX_ENDPOINTS)}."
            )
        current_ep, changes_ep, _band = INDEX_ENDPOINTS[index]
        archive = Archive(self.archive_root)
        current = archive.read(current_ep)
        changes = archive.read(changes_ep)
        if current is None or changes is None:
            raise FileNotFoundError(
                f"FMP constituent payloads for {index!r} are not in the archive at "
                f"{self.archive_root}. Fetch them with "
                "`./venv/bin/python scripts/fmp_bulk_download.py --groups index`."
            )
        build = build_intervals(
            index,
            current.reset_index().to_dict("records"),
            changes.reset_index().to_dict("records"),
            since=since,
        )
        return build.spells
