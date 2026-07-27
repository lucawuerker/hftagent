"""The FMP endpoint registry — the single source of truth for *what* we fetch.

Every module in this package reads this registry: the capability probe calls each
entry once, the downloader fans out over ``(endpoint, symbol)`` pairs, and the
store derives its directory layout from :attr:`Endpoint.dest`.  Adding a new
endpoint to the archive means adding one row here.

Two axes matter:

* **kind** — ``"symbol"`` endpoints are fetched once per ticker (the bulk of the
  work); ``"global"`` endpoints are fetched once per run (index constituents,
  the delisted-company list, the symbol-change map).
* **window_years** — endpoints that accept ``from``/``to`` can be chunked into
  N-year requests so a 22-year pull never depends on an undocumented
  response-size cap.  ``None`` means "one request for the whole range"; the
  capability probe decides which is safe and the downloader honours it.

Field names are *not* declared here.  The archive keeps every column the vendor
returns (Layer A); the canonical panel vocabulary lives in ``data/fields.py``
and is applied later by the archive provider (Layer B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

#: FMP's current API root.  The legacy ``/api/v3`` endpoints 403 for keys issued
#: after Aug 2025 (see ``data/providers/fmp.py``), so everything here is stable.
FMP_BASE = "https://financialmodelingprep.com/stable"

#: Fiscal periods pulled for statement-shaped endpoints.  Quarterly is what
#: factors want; annual carries FY aggregates and a few annual-only line items.
PERIODS: tuple[str, ...] = ("quarter", "annual")


@dataclass(frozen=True)
class Endpoint:
    """One FMP endpoint as the archive consumes it."""

    #: registry key; also the manifest key prefix
    name: str
    #: path under :data:`FMP_BASE` (no leading slash)
    path: str
    #: logical group, used by ``--groups`` to switch whole families on/off
    group: str
    #: archive-relative destination directory (``prices/adjusted``, …)
    dest: str
    #: ``"symbol"`` (one call per ticker) or ``"global"`` (one call per run)
    kind: str = "symbol"
    #: static query params merged into every request
    params: dict[str, str | int] = field(default_factory=dict)
    #: column used as the parquet index (``None`` → RangeIndex, e.g. profile)
    date_field: str | None = "date"
    #: chunk ``from``/``to`` into N-year requests (``None`` → single request)
    window_years: int | None = None
    #: whether the endpoint accepts ``from``/``to`` at all
    windowed: bool = False
    #: ``limit`` value to request (clamped by the probed plan cap)
    limit: int | None = None
    #: fiscal periods to pull; one archive file per period
    periods: tuple[str, ...] = ()
    #: a *current* snapshot rather than a dated history — never PIT-backfillable
    snapshot: bool = False
    #: the endpoint is paginated via ``page``/``limit``
    paginated: bool = False
    #: human note surfaced in the docs/report
    note: str = ""

    def url(self) -> str:
        return f"{FMP_BASE}/{self.path}"

    def dest_for(self, period: str | None = None) -> str:
        """Archive-relative directory for this endpoint (and fiscal period)."""
        return f"{self.dest}/{period}" if period else self.dest

    def key(self, symbol: str | None = None, period: str | None = None) -> str:
        """Stable manifest key for one unit of work."""
        parts = [self.name]
        if period:
            parts.append(period)
        if symbol:
            parts.append(symbol)
        return ":".join(parts)


# ── per-symbol endpoints ─────────────────────────────────────────────────────
#
# `limit` values are deliberately generous (the plan cap is probed and applied at
# request time); 400 quarters is a century of filings, so nothing truncates.

_PRICES: tuple[Endpoint, ...] = (
    Endpoint(
        name="prices_adjusted",
        path="historical-price-eod/dividend-adjusted",
        group="prices",
        dest="prices/adjusted",
        windowed=True,
        note="split+dividend adjusted OHLC (adjOpen…adjClose) + volume",
    ),
    Endpoint(
        name="prices_unadjusted",
        path="historical-price-eod/full",
        group="prices",
        dest="prices/unadjusted",
        windowed=True,
        note="raw OHLCV + vwap/change; with dividends+splits this rebuilds a "
             "point-in-time adjustment factor",
    ),
    Endpoint(
        name="dividends",
        path="dividends",
        group="prices",
        dest="prices/dividends",
        limit=1000,
        note="declaration/record/payment dates — PIT-safe corporate actions",
    ),
    Endpoint(
        name="splits",
        path="splits",
        group="prices",
        dest="prices/splits",
        limit=1000,
    ),
)

_FUNDAMENTALS: tuple[Endpoint, ...] = (
    Endpoint(
        name="income_statement",
        path="income-statement",
        group="fundamentals",
        dest="fundamentals/income-statement",
        limit=400,
        periods=PERIODS,
        note="39 fields incl. filingDate + acceptedDate (the availability anchor)",
    ),
    Endpoint(
        name="balance_sheet",
        path="balance-sheet-statement",
        group="fundamentals",
        dest="fundamentals/balance-sheet-statement",
        limit=400,
        periods=PERIODS,
        note="61 fields incl. filingDate + acceptedDate",
    ),
    Endpoint(
        name="cash_flow",
        path="cash-flow-statement",
        group="fundamentals",
        dest="fundamentals/cash-flow-statement",
        limit=400,
        periods=PERIODS,
        note="47 fields incl. filingDate + acceptedDate",
    ),
    Endpoint(
        name="ratios",
        path="ratios",
        group="fundamentals",
        dest="fundamentals/ratios",
        limit=400,
        periods=PERIODS,
        note="64 fields; NO filingDate → availability joined from income_statement",
    ),
    Endpoint(
        name="key_metrics",
        path="key-metrics",
        group="fundamentals",
        dest="fundamentals/key-metrics",
        limit=400,
        periods=PERIODS,
        note="47 fields; NO filingDate → availability joined from income_statement",
    ),
    Endpoint(
        name="financial_growth",
        path="financial-growth",
        group="fundamentals",
        dest="fundamentals/financial-growth",
        limit=400,
        periods=PERIODS,
        note="44 fields; NO filingDate → availability joined from income_statement",
    ),
    Endpoint(
        name="enterprise_values",
        path="enterprise-values",
        group="fundamentals",
        dest="fundamentals/enterprise-values",
        limit=400,
        periods=PERIODS,
    ),
    Endpoint(
        name="earnings",
        path="earnings",
        group="fundamentals",
        dest="fundamentals/earnings",
        limit=400,
        note="actual vs estimated EPS/revenue at the report date → surprise",
    ),
)

_REFERENCE: tuple[Endpoint, ...] = (
    Endpoint(
        name="profile",
        path="profile",
        group="reference",
        dest="reference/profile",
        date_field=None,
        snapshot=True,
        note="CURRENT snapshot — only the (near-)static labels (sector/industry/"
             "cik/exchange) are safe to use historically",
    ),
    Endpoint(
        name="market_cap",
        path="historical-market-capitalization",
        group="reference",
        dest="reference/market-cap",
        windowed=True,
        # ~1 260 bars per 5-year chunk, comfortably inside the 5 000-row limit.
        # (A 1-year chunk would make this endpoint alone half of the whole run's
        # HTTP calls; a single 22-year request would sit right on the cap.)
        window_years=5,
        limit=5000,
        note="daily market cap history — PIT-safe (price x shares on the day)",
    ),
    Endpoint(
        name="shares_float",
        path="shares-float",
        group="reference",
        dest="reference/shares-float",
        date_field=None,
        snapshot=True,
        note="CURRENT float snapshot; the PIT-safe share count is "
             "weightedAverageShsOut on the filing-stamped income statement",
    ),
)

# ── run-level (non-per-symbol) endpoints ─────────────────────────────────────

_INDEX: tuple[Endpoint, ...] = (
    Endpoint(
        name="sp500_constituent",
        path="sp500-constituent",
        group="index",
        dest="index/sp500-constituent",
        kind="global",
        date_field=None,
        snapshot=True,
    ),
    Endpoint(
        name="historical_sp500_constituent",
        path="historical-sp500-constituent",
        group="index",
        dest="index/historical-sp500-constituent",
        kind="global",
        note="dated add/remove change log — backward-walked into membership spells",
    ),
    Endpoint(
        name="nasdaq_constituent",
        path="nasdaq-constituent",
        group="index",
        dest="index/nasdaq-constituent",
        kind="global",
        date_field=None,
        snapshot=True,
    ),
    Endpoint(
        name="historical_nasdaq_constituent",
        path="historical-nasdaq-constituent",
        group="index",
        dest="index/historical-nasdaq-constituent",
        kind="global",
    ),
    Endpoint(
        name="delisted_companies",
        path="delisted-companies",
        group="index",
        dest="index/delisted-companies",
        kind="global",
        date_field=None,
        paginated=True,
        limit=100,
        note="used to resolve tickers of names that left the index and delisted",
    ),
    Endpoint(
        name="symbol_change",
        path="symbol-change",
        group="index",
        dest="index/symbol-change",
        kind="global",
        limit=1000,
        note="ticker renames — keeps a renamed name one continuous spell",
    ),
)

ENDPOINTS: dict[str, Endpoint] = {
    e.name: e for e in (*_PRICES, *_FUNDAMENTALS, *_REFERENCE, *_INDEX)
}

#: Groups pulled unless ``--groups`` narrows it.  ``index`` is fetched by the
#: membership builder rather than the per-symbol download, so it is not a default
#: here; the CLI adds it explicitly.
DEFAULT_GROUPS: tuple[str, ...] = ("prices", "fundamentals", "reference")


def known_groups() -> list[str]:
    return sorted({e.group for e in ENDPOINTS.values()})


def endpoints_for_groups(
    groups: Iterable[str] | None = None, *, kind: str | None = None
) -> list[Endpoint]:
    """Registry entries in the given ``groups`` (and optionally of one ``kind``).

    Raises ``ValueError`` on an unknown group rather than silently pulling
    nothing — a typo in ``--groups`` must not look like a completed download.
    """
    wanted = tuple(groups) if groups is not None else DEFAULT_GROUPS
    unknown = sorted(set(wanted) - set(known_groups()))
    if unknown:
        raise ValueError(f"Unknown endpoint group(s) {unknown}. Known: {known_groups()}.")
    out = [e for e in ENDPOINTS.values() if e.group in wanted]
    if kind is not None:
        out = [e for e in out if e.kind == kind]
    return out


def work_units(endpoint: Endpoint) -> list[str | None]:
    """The per-endpoint fan-out: one entry per fiscal period, or ``[None]``."""
    return list(endpoint.periods) if endpoint.periods else [None]
