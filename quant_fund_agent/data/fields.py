"""Canonical non-OHLCV field vocabulary + per-vendor name normalization.

The OHLCV layer is uniform across vendors, but fundamentals/estimates are named
differently by every provider (FMP ``peRatio`` vs AlphaVantage ``PERatio``).  This
module is the single source of truth for the *canonical* field names the panel
exposes, plus the maps that translate each vendor's raw keys to them — the
ticker-level analogue of what :mod:`quant_fund_agent.data.symbols` does for
symbols.

A canonical field name (camelCase, matching the existing ``peRatio`` style) is
what a factor declares in ``inputs`` and reads as ``data["peRatio"]``.  Tier
membership (``data/tiers.py``) drives capability gating; this module only handles
*vocabulary*, not availability.

Two field kinds matter for downstream handling:
  * **numeric** — coerced to float (NaN on parse failure);
  * **categorical** (``sector``/``industry``) — kept as object dtype (a label),
    forward-filled like the numerics but never cast to float.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# ── canonical field groups ──────────────────────────────────────────────────
# (mirror the tier sets in data/tiers.py; kept here so the normalization maps and
#  the tiers can't drift apart silently — `tiers` imports these.)

CATEGORICAL_FIELDS: frozenset[str] = frozenset({"sector", "industry", "subindustry"})

FUNDAMENTAL_FIELDS: frozenset[str] = frozenset(
    {
        "sector", "industry", "subindustry", "cap", "marketCap",
        "peRatio", "pbRatio", "psRatio",
        "roe", "roic", "debtToEquity", "currentRatio",
        "grossMargin", "netMargin",
        "revenue", "eps", "freeCashFlow",
    }
)

ESTIMATE_FIELDS: frozenset[str] = frozenset(
    {"epsEstimate", "revenueEstimate"}
)

EVENT_FIELDS: frozenset[str] = frozenset({"epsSurprise"})

#: every non-OHLCV field this stage can produce
NON_OHLCV_FIELDS: frozenset[str] = (
    FUNDAMENTAL_FIELDS | ESTIMATE_FIELDS | EVENT_FIELDS
)


# ── per-vendor candidate-key maps ───────────────────────────────────────────
# canonical -> tuple of candidate vendor keys (first present wins). Multiple
# candidates make the reshapes robust to endpoint/version drift (FMP's v3→stable
# migration renamed several keys; AV uses TitleCase with TTM suffixes).

# Profile supplies only the static labels — ``marketCap`` there is a *current*
# snapshot, so we take the per-period one from key-metrics to avoid look-ahead.
FMP_PROFILE_MAP: dict[str, tuple[str, ...]] = {
    "sector": ("sector",),
    "industry": ("industry",),
}

FMP_METRICS_MAP: dict[str, tuple[str, ...]] = {
    "marketCap": ("marketCap", "marketCapitalization"),
    "peRatio": ("peRatio", "priceEarningsRatio", "priceToEarningsRatio", "pe"),
    "pbRatio": ("pbRatio", "priceToBookRatio", "ptbRatio"),
    "psRatio": ("priceToSalesRatio", "priceSalesRatio", "psRatio"),
    "roe": ("roe", "returnOnEquity"),
    "roic": ("roic", "returnOnInvestedCapital"),
    "debtToEquity": ("debtToEquity", "debtEquityRatio"),
    "currentRatio": ("currentRatio",),
    "grossMargin": ("grossProfitMargin", "grossMargin"),
    "netMargin": ("netProfitMargin", "netIncomeMargin", "netMargin"),
    "freeCashFlow": ("freeCashFlowPerShare", "freeCashFlow"),
}

FMP_INCOME_MAP: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue",),
    "eps": ("eps", "epsdiluted", "epsDiluted"),
    "netMargin": ("netIncomeRatio", "netProfitMargin"),
}

FMP_ESTIMATE_MAP: dict[str, tuple[str, ...]] = {
    "epsEstimate": ("estimatedEpsAvg", "epsAvg", "estimatedEps"),
    "revenueEstimate": ("estimatedRevenueAvg", "revenueAvg", "estimatedRevenue"),
}

# the 'earnings' endpoint carries actual + estimated → surprise is derived
FMP_EARNINGS_ACTUAL_KEYS: tuple[str, ...] = ("eps", "epsActual")
FMP_EARNINGS_ESTIMATE_KEYS: tuple[str, ...] = ("epsEstimated", "epsEstimate")

# availability (filing) vs fiscal-period-end date keys on FMP statements
FMP_FILING_DATE_KEYS: tuple[str, ...] = ("fillingDate", "filingDate", "acceptedDate")
FMP_PERIOD_END_KEYS: tuple[str, ...] = ("date", "fiscalDateEnding")

# AV's COMPANY_OVERVIEW is an undated *current* snapshot: its ratios would leak if
# backfilled, so we take only the (near-)static labels from it and get the rest
# from the dated statement/earnings endpoints below.
AV_PROFILE_MAP: dict[str, tuple[str, ...]] = {
    "sector": ("Sector",),
    "industry": ("Industry",),
}

# AV INCOME_STATEMENT quarterlyReports: dated by fiscalDateEnding (no filing date
# → reporting-lag fallback). ``netMargin`` is derived netIncome / totalRevenue.
AV_REVENUE_KEYS: tuple[str, ...] = ("totalRevenue", "revenue")
AV_NET_INCOME_KEYS: tuple[str, ...] = ("netIncome",)

# AV EARNINGS quarterlyEarnings carries actual + estimated EPS + a real reportedDate.
AV_EARNINGS_ACTUAL_KEYS: tuple[str, ...] = ("reportedEPS",)
AV_EARNINGS_ESTIMATE_KEYS: tuple[str, ...] = ("estimatedEPS",)
AV_REPORTED_DATE_KEYS: tuple[str, ...] = ("reportedDate",)
AV_PERIOD_END_KEYS: tuple[str, ...] = ("fiscalDateEnding",)


# ── helpers ─────────────────────────────────────────────────────────────────

# AlphaVantage uses the literal strings "None"/"-" for missing numerics.
_MISSING_TOKENS = {"", "none", "nan", "-", "n/a", "null"}


def pick(record: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    """First non-missing value among ``candidates`` keys in ``record`` (else None)."""
    for key in candidates:
        if key in record:
            val = record[key]
            if val is None:
                continue
            if isinstance(val, str) and val.strip().lower() in _MISSING_TOKENS:
                continue
            return val
    return None


def coerce_numeric(value: Any) -> float | None:
    """Parse a vendor value to float; ``None``/sentinels → ``None`` (becomes NaN)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower() in _MISSING_TOKENS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize(record: Mapping[str, Any], field_map: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """Translate one raw vendor record to ``{canonical: value}``.

    Categorical fields (``sector``/``industry``) keep their string label; all
    other fields are coerced to float (NaN-on-failure).  Missing/empty values are
    omitted so an upstream merge keeps the prior non-null value.
    """
    out: dict[str, Any] = {}
    for canonical, candidates in field_map.items():
        if not candidates:
            continue
        raw = pick(record, candidates)
        if raw is None:
            continue
        if canonical in CATEGORICAL_FIELDS:
            label = str(raw).strip()
            if label and label.lower() not in _MISSING_TOKENS:
                out[canonical] = label
        else:
            num = coerce_numeric(raw)
            if num is not None:
                out[canonical] = num
    return out
