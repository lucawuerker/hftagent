"""Resolving *historical* tickers to symbols FMP will actually serve.

A point-in-time membership table names securities by the ticker they carried
**while they were index members** — including names that have since been
acquired, renamed or gone bankrupt (``AABA``, ``ENRNQ``, ``VIAC``, ``FBHS``).
Those are exactly the rows a survivorship-bias-free study needs and exactly the
rows a naive symbol lookup loses.

Resolution is deliberately *cheap and ordered*: the plain ticker is tried first
(it works for the overwhelming majority), and only a miss spends extra calls on
variants.  Each attempt and its outcome is recorded, and every resolved name gets
a **spell coverage** number — the fraction of its actual membership window for
which bars came back.  That single column is what quantifies how much of the
survivorship gap this pull closed, the same way ``--check-coverage`` quantified
the yfinance gap in ``docs/data-layer/SP500_MEMBERSHIP.md`` §7.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import pandas as pd

from quant_fund_agent.data.fmp_ingest.client import FMPClient
from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS

log = logging.getLogger("data.fmp_ingest.symbols")

SYMBOL_MAP_FILE = "symbol_map.csv"

#: Bankruptcy/delisting suffixes exchanges append to a ticker (``ENRNQ`` was
#: ``ENRN``).  Only stripped as a *fallback*, never before the literal ticker.
_DEAD_SUFFIXES = ("Q", "E")


@dataclass
class Resolution:
    """How one membership ticker was mapped onto an FMP symbol."""

    ticker: str                       # as it appears in the membership table
    resolved: str | None = None       # symbol FMP served (None = unresolved)
    method: str = "unresolved"
    n_bars: int = 0
    first_date: str | None = None
    last_date: str | None = None
    spell_start: str | None = None
    spell_end: str | None = None
    spell_coverage: float = 0.0       # fraction of the membership window covered
    attempts: str = ""                # candidates tried, in order
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def candidate_symbols(
    ticker: str,
    *,
    renames: dict[str, str] | None = None,
    max_candidates: int = 5,
) -> list[str]:
    """Ordered symbols to try for one membership ticker (literal first)."""
    seen: list[str] = []

    def add(sym: str | None) -> None:
        if sym and sym not in seen:
            seen.append(sym)

    t = str(ticker).strip().upper()
    add(t)
    # The repo normalises share classes to `BRK-B`; FMP has served both forms.
    if "-" in t:
        add(t.replace("-", "."))
    if "." in t:
        add(t.replace(".", "-"))
    # Follow a rename chain (FBHS -> FBIN), guarding against cycles.
    chain, cur = 0, t
    while renames and cur in renames and chain < 4:
        cur = renames[cur]
        add(cur)
        chain += 1
    # Bankruptcy suffix stripped last: ENRNQ -> ENRN.
    if len(t) > 3 and t[-1] in _DEAD_SUFFIXES:
        add(t[:-1])
    return seen[:max_candidates]


def load_symbol_changes(client: FMPClient) -> dict[str, str]:
    """``old ticker -> new ticker`` from FMP's symbol-change feed (best effort)."""
    ep = ENDPOINTS["symbol_change"]
    result = client.get(ep.url(), {"limit": ep.limit})
    if not result.ok:
        log.info("symbol-change unavailable (%s); rename fallback disabled",
                 (result.error or "")[:120])
        return {}
    out: dict[str, str] = {}
    for row in result.rows:
        old = str(row.get("oldSymbol") or "").strip().upper()
        new = str(row.get("newSymbol") or "").strip().upper()
        if old and new and old != new:
            out[old] = new
    log.info("symbol-change: %d renames loaded", len(out))
    return out


def load_delisted(client: FMPClient, *, max_pages: int = 60) -> dict[str, dict]:
    """``symbol -> record`` for delisted companies (paginated, bounded)."""
    ep = ENDPOINTS["delisted_companies"]
    out: dict[str, dict] = {}
    for page in range(max_pages):
        result = client.get(ep.url(), {"page": page, "limit": ep.limit})
        if not result.ok or not result.rows:
            break
        for row in result.rows:
            sym = str(row.get("symbol") or "").strip().upper()
            if sym:
                out.setdefault(sym, row)
        if len(result.rows) < (ep.limit or 100):
            break
    log.info("delisted-companies: %d symbols indexed", len(out))
    return out


# ── coverage measurement ─────────────────────────────────────────────────────

def spell_windows(
    spells: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """A ticker's membership spells clipped to ``[start, end]`` (end exclusive)."""
    rows = spells[spells["ticker"] == ticker]
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, row in rows.iterrows():
        s = max(pd.Timestamp(row["start_date"]), start)
        raw_end = row["end_date"]
        e = end if pd.isna(raw_end) else min(pd.Timestamp(raw_end), end)
        if s < e:
            out.append((s, e))
    return out


def spell_coverage(
    bars: pd.DatetimeIndex | None,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    """Fraction of the membership window's business days that have a bar.

    Business days are the denominator (not calendar days), so a name held for two
    months isn't penalised for weekends.  Holidays make a perfect score ~0.97
    rather than 1.0, which is why the report reads this as a *coverage band*, not
    an exact ratio.
    """
    if not windows:
        return 0.0
    expected = sum(len(pd.bdate_range(s, e - pd.Timedelta(days=1))) for s, e in windows)
    if expected <= 0:
        return 0.0
    if bars is None or len(bars) == 0:
        return 0.0
    idx = pd.DatetimeIndex(bars)
    got = sum(int(((idx >= s) & (idx < e)).sum()) for s, e in windows)
    return round(min(got / expected, 1.0), 4)


def summarise(resolutions: list[Resolution]) -> dict:
    """Aggregate stats for the coverage report."""
    total = len(resolutions)
    resolved = [r for r in resolutions if r.resolved]
    covered = [r for r in resolved if r.spell_coverage >= 0.90]
    partial = [r for r in resolved if 0.10 <= r.spell_coverage < 0.90]
    return {
        "tickers": total,
        "resolved": len(resolved),
        "resolved_pct": round(100 * len(resolved) / total, 1) if total else 0.0,
        "coverage_ge_90pct": len(covered),
        "coverage_ge_90pct_pct": round(100 * len(covered) / total, 1) if total else 0.0,
        "coverage_partial": len(partial),
        "unresolved": sorted(r.ticker for r in resolutions if not r.resolved),
        "mean_coverage": round(
            sum(r.spell_coverage for r in resolutions) / total, 4) if total else 0.0,
        "by_method": _counts(r.method for r in resolutions),
    }


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def to_frame(resolutions: list[Resolution]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in resolutions]).sort_values("ticker")
