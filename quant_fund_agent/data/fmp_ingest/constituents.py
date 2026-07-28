"""FMP index change logs → the canonical point-in-time membership table.

FMP serves index membership as two pieces: a **current constituent list**
(``sp500-constituent`` / ``nasdaq-constituent``) and a dated **change log**
(``historical-sp500-constituent`` / ``historical-nasdaq-constituent``) of
add/remove events.  Neither is a membership table; the table is reconstructed by
**backward-walking** the log from today's set:

    S(n)   = today's constituents                     (effective from d_n)
    S(k-1) = S(k) - added(d_k) + removed(d_k)         (effective on [d_{k-1}, d_k))

then run-length-encoding each ticker's presence into spells.  The output is the
schema every consumer already reads —
``ticker,name,start_date,end_date,add_reason,remove_reason,source`` with an
**exclusive** ``end_date`` — so ``data/membership.py``, ``resolve_universe`` and
the per-bar panel mask work unchanged.

Two honest caveats, both surfaced in the build report rather than hidden:

* **Left censoring.**  Names already in the index when the log begins have no
  add event.  Their spell starts at ``dateFirstAdded`` when the current-list
  endpoint supplies one, otherwise at the log's earliest date, flagged
  ``left-censored``.  If the log starts *after* the requested ``since``, the
  early window is not trustworthy and the report says so.
* **Backward-walk fragility.**  Every missing event in the log propagates into
  *all* earlier dates.  That is why the result is audited (count band, no
  overlaps) and reconciled month-by-month against the independent public
  reconstruction instead of being trusted on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from quant_fund_agent.data.fields import pick
from quant_fund_agent.data.membership import (
    coalesce_spells,
    normalize_ticker,
)

log = logging.getLogger("data.fmp_ingest.constituents")

#: index name → (current-list endpoint, change-log endpoint, count band)
INDEX_ENDPOINTS: dict[str, tuple[str, str, tuple[int, int]]] = {
    "sp500": ("sp500_constituent", "historical_sp500_constituent", (475, 515)),
    "nasdaq100": ("nasdaq_constituent", "historical_nasdaq_constituent", (95, 115)),
}

_SYMBOL_KEYS = ("symbol", "addedSymbol", "addedTicker", "ticker")
_ADDED_NAME_KEYS = ("addedSecurity", "addedSecurityName", "companyName", "name")
_REMOVED_SYMBOL_KEYS = ("removedTicker", "removedSymbol")
_REMOVED_NAME_KEYS = ("removedSecurity", "removedSecurityName")
_DATE_KEYS = ("date", "dateAdded", "effectiveDate")
_REASON_KEYS = ("reason", "note")
_FIRST_ADDED_KEYS = ("dateFirstAdded", "founded", "firstAdded")
_CIK_KEYS = ("cik", "CIK")


@dataclass
class ConstituentBuild:
    """Result of one index reconstruction, with everything the report needs."""

    index: str
    spells: pd.DataFrame            # canonical schema (+ cik)
    changes: pd.DataFrame           # parsed change log
    current: pd.DataFrame           # parsed current constituents
    log_earliest: pd.Timestamp | None
    left_censored: list[str]
    notes: list[str]


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_current(rows: list[dict]) -> pd.DataFrame:
    """``sp500-constituent``-shaped rows → ``ticker,name,cik,date_first_added``."""
    out = []
    for rec in rows or []:
        ticker = pick(rec, _SYMBOL_KEYS)
        if not ticker:
            continue
        out.append({
            "ticker": normalize_ticker(ticker),
            "name": (pick(rec, ("name", "companyName", "security")) or ""),
            "cik": str(pick(rec, _CIK_KEYS) or "").strip(),
            "sector": (pick(rec, ("sector",)) or ""),
            "date_first_added": pd.to_datetime(
                pick(rec, _FIRST_ADDED_KEYS), errors="coerce"),
        })
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "name", "cik", "sector", "date_first_added"])
    return df.drop_duplicates("ticker").reset_index(drop=True)


def parse_changes(rows: list[dict]) -> pd.DataFrame:
    """Change-log rows → ``date,added,added_name,removed,removed_name,reason``.

    Rows carry an add, a remove, or both.  Dates are parsed leniently because FMP
    mixes ISO (``date``) and prose (``dateAdded``: *"October 1, 2024"*) forms.
    """
    out = []
    for rec in rows or []:
        stamp = None
        for key in _DATE_KEYS:
            stamp = pd.to_datetime(pick(rec, (key,)), errors="coerce")
            if pd.notna(stamp):
                break
        if stamp is None or pd.isna(stamp):
            continue
        added = pick(rec, _SYMBOL_KEYS)
        removed = pick(rec, _REMOVED_SYMBOL_KEYS)
        out.append({
            "date": pd.Timestamp(stamp).normalize(),
            "added": normalize_ticker(added) if added else None,
            "added_name": pick(rec, _ADDED_NAME_KEYS) or "",
            "removed": normalize_ticker(removed) if removed else None,
            "removed_name": pick(rec, _REMOVED_NAME_KEYS) or "",
            "reason": pick(rec, _REASON_KEYS) or "",
        })
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(
            columns=["date", "added", "added_name", "removed", "removed_name", "reason"])
    return df.sort_values("date").reset_index(drop=True)


# ── reconstruction ───────────────────────────────────────────────────────────

def walk_backward(
    current: set[str], changes: pd.DataFrame, *, pre_start: pd.Timestamp
) -> list[tuple[pd.Timestamp, frozenset[str]]]:
    """Replay the change log backwards into ``(effective_from, member set)`` rows.

    Returns rows in ascending date order; row ``i``'s set is effective on
    ``[date_i, date_{i+1})``.  The first row is the **pre-log** set — the
    constituents implied *before* the earliest recorded event — stamped at
    ``pre_start``.  It must be strictly earlier than the first change date, or a
    name removed by that very first event would collapse to a zero-length spell.
    """
    if changes.empty:
        return []
    dates = [pd.Timestamp(d) for d in sorted(changes["date"].unique())]
    by_date = {pd.Timestamp(d): g for d, g in changes.groupby("date")}

    def _delta(day: pd.Timestamp) -> tuple[set[str], set[str]]:
        g = by_date[day]
        return (
            {t for t in g["added"].dropna().tolist() if t},
            {t for t in g["removed"].dropna().tolist() if t},
        )

    sets: dict[pd.Timestamp, frozenset[str]] = {}
    running = set(current)
    # S(n) = today's set, effective from the last change date.
    sets[dates[-1]] = frozenset(running)
    for i in range(len(dates) - 1, 0, -1):
        added, removed = _delta(dates[i])
        running = (running - added) | removed
        sets[dates[i - 1]] = frozenset(running)
    # Undo the earliest change too, to recover the pre-log set.
    added, removed = _delta(dates[0])
    pre = (running - added) | removed
    pre_start = min(pd.Timestamp(pre_start), dates[0] - pd.Timedelta(days=1))
    return [(pre_start, frozenset(pre))] + [(d, sets[d]) for d in sorted(sets)]


def spells_from_timeline(
    timeline: list[tuple[pd.Timestamp, frozenset[str]]]
) -> pd.DataFrame:
    """Run-length-encode a ``(date, set)`` timeline into per-ticker spells.

    Mirrors ``intervals_from_components`` in the public builder: a maximal run of
    consecutive rows ``[a..b]`` becomes ``start=date[a]``, ``end=date[b+1]``
    (exclusive; ``NaT`` when the run reaches the newest row).
    """
    if not timeline:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    dates = [d for d, _ in timeline]
    sets = [s for _, s in timeline]
    n = len(dates)

    present: dict[str, list[int]] = {}
    for i, members in enumerate(sets):
        for t in members:
            present.setdefault(t, []).append(i)

    rows = []
    for ticker, idxs in present.items():
        run_start = prev = idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
                continue
            rows.append((ticker, dates[run_start], dates[prev + 1]))
            run_start = prev = i
        end = dates[prev + 1] if prev + 1 < n else pd.NaT
        rows.append((ticker, dates[run_start], end))
    return pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])


def build_intervals(
    index: str,
    current_rows: list[dict],
    change_rows: list[dict],
    *,
    since: str | pd.Timestamp = "2004-01-01",
) -> ConstituentBuild:
    """Full reconstruction: raw FMP payloads → canonical spells + provenance."""
    since_ts = pd.Timestamp(since).normalize()
    current = parse_current(current_rows)
    changes = parse_changes(change_rows)
    notes: list[str] = []

    if changes.empty:
        raise ValueError(
            f"{index}: the FMP change log parsed to zero rows — the payload schema "
            "likely differs from the expected keys "
            f"({_DATE_KEYS} / {_SYMBOL_KEYS} / {_REMOVED_SYMBOL_KEYS})."
        )
    log_earliest = pd.Timestamp(changes["date"].min())
    if log_earliest > since_ts:
        notes.append(
            f"change log starts {log_earliest.date()}, AFTER the requested "
            f"since={since_ts.date()} — membership before {log_earliest.date()} is "
            "left-censored and should be taken from the public reconstruction."
        )

    # The pre-log set has no add event: floor it below both the log start and the
    # requested window so those names are not truncated into the study period.
    pre_start = min(log_earliest, since_ts) - pd.Timedelta(days=1)
    timeline = walk_backward(set(current["ticker"]), changes, pre_start=pre_start)
    spells = coalesce_spells(spells_from_timeline(timeline))

    # Left-censored names: their spell start is a floor, not a fact.  The current
    # constituent list supplies a real `dateFirstAdded` for names still in the
    # index; everything else stays flagged.
    pre_set = set(timeline[0][1]) if timeline else set()
    first_added = dict(
        zip(current["ticker"], current["date_first_added"])) if len(current) else {}
    left_censored: list[str] = []
    starts = []
    for _, row in spells.iterrows():
        start = row["start_date"]
        if row["ticker"] in pre_set and start <= pre_start:
            known = first_added.get(row["ticker"])
            known = pd.Timestamp(known).normalize() if pd.notna(known) else None
            # `dateFirstAdded` is only usable when it actually falls inside this
            # spell.  For a name that left and rejoined (DAL, PCG) the current
            # list reports its *latest* addition, which lands after this spell
            # ended — applying it blindly inverts the spell and makes it overlap
            # the later one.  Fall back to the floor and flag it instead.
            if known is not None and (pd.isna(row["end_date"]) or known < row["end_date"]):
                start = known
            else:
                left_censored.append(row["ticker"])
        starts.append(start)
    spells = spells.assign(start_date=starts)

    # Keep spells overlapping [since, today]; true (pre-since) starts are retained.
    keep = spells["end_date"].isna() | (spells["end_date"] > since_ts)
    spells = spells[keep].reset_index(drop=True)

    enriched = _enrich(index, spells, current, changes)
    return ConstituentBuild(
        index=index,
        spells=enriched,
        changes=changes,
        current=current,
        log_earliest=log_earliest,
        left_censored=sorted(set(left_censored)),
        notes=notes,
    )


def _enrich(
    index: str, spells: pd.DataFrame, current: pd.DataFrame, changes: pd.DataFrame
) -> pd.DataFrame:
    """Attach company name, CIK and the add/remove reasons to each spell."""
    names: dict[str, str] = {}
    for _, r in current.iterrows():
        if r["name"]:
            names[r["ticker"]] = r["name"]
    for _, r in changes.iterrows():
        if r["added"] and r["added_name"]:
            names.setdefault(r["added"], r["added_name"])
        if r["removed"] and r["removed_name"]:
            names.setdefault(r["removed"], r["removed_name"])

    ciks = {r["ticker"]: r["cik"] for _, r in current.iterrows() if r["cik"]}

    add_reason = {(r["added"], r["date"]): r["reason"]
                  for _, r in changes.iterrows() if r["added"]}
    rm_reason = {(r["removed"], r["date"]): r["reason"]
                 for _, r in changes.iterrows() if r["removed"]}

    out = spells.copy()
    out["name"] = [names.get(t, "") for t in out["ticker"]]
    out["cik"] = [ciks.get(t, "") for t in out["ticker"]]
    out["add_reason"] = [add_reason.get((t, d), "")
                         for t, d in zip(out["ticker"], out["start_date"])]
    out["remove_reason"] = [
        "" if pd.isna(d) else rm_reason.get((t, d), "")
        for t, d in zip(out["ticker"], out["end_date"])
    ]
    out["source"] = f"fmp:{index}"
    return out[["ticker", "name", "start_date", "end_date",
                "add_reason", "remove_reason", "source", "cik"]]


def to_csv_frame(spells: pd.DataFrame) -> pd.DataFrame:
    """Canonical on-disk form: sorted, ISO dates, empty string for open spells."""
    df = spells.sort_values(["ticker", "start_date"]).copy()
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")
    end = pd.to_datetime(df["end_date"])
    df["end_date"] = end.dt.strftime("%Y-%m-%d").where(end.notna(), "")
    return df
