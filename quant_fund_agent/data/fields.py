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

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# ── canonical field groups ──────────────────────────────────────────────────
# (mirror the tier sets in data/tiers.py; kept here so the normalization maps and
#  the tiers can't drift apart silently — `tiers` imports these.)

CATEGORICAL_FIELDS: frozenset[str] = frozenset({"sector", "industry", "subindustry"})

#: The original, deliberately small vocabulary the *live* API providers fill
#: (yfinance / AlphaVantage / the network-backed FMP provider).  The premium
#: archive extends it via :data:`ARCHIVE_FIELD_SPECS` below; these names stay
#: listed explicitly so a change there can never silently drop one.
LEGACY_FUNDAMENTAL_FIELDS: frozenset[str] = frozenset(
    {
        "sector", "industry", "subindustry", "cap", "marketCap",
        "peRatio", "pbRatio", "psRatio",
        "roe", "roic", "debtToEquity", "currentRatio",
        "grossMargin", "netMargin",
        "revenue", "eps", "freeCashFlow",
    }
)


# ── the premium-archive vocabulary ──────────────────────────────────────────
#
# One row per canonical panel field: where it comes from, which vendor key(s)
# fill it, and how to describe it to the Factor Researcher.  This single table
# drives three things that used to be maintained separately — the tier sets
# (capability gating), the per-endpoint normalization maps used by the archive
# provider, and the DATA CONTEXT prose in
# ``agents/factor_research/prompts.py`` — so they cannot drift apart.
#
# `source` is the ``data/fmp_ingest/endpoints.py`` registry name that fills the
# field.  Fields with several plausible vendor keys list them in preference
# order (first present wins), which keeps the mapping robust to FMP's periodic
# key renames.

@dataclass(frozen=True)
class FieldSpec:
    """One canonical panel field and how it is filled."""

    name: str
    group: str          # profitability | valuation | leverage | … (see FIELD_GROUPS)
    source: str         # fmp_ingest endpoint registry name
    keys: tuple[str, ...]
    desc: str

    @property
    def kind(self) -> str:
        """Tier bucket: ``fundamental``, ``estimates`` or ``events``."""
        if self.group == "estimates":
            return "estimates"
        if self.group == "events":
            return "events"
        return "fundamental"


#: Display order + heading for each group in the researcher's DATA CONTEXT.
FIELD_GROUPS: dict[str, str] = {
    "labels": "Classification labels (text, static)",
    "size": "Size / market value",
    "income": "Income statement (per fiscal quarter, USD)",
    "balance": "Balance sheet (per fiscal quarter, USD)",
    "cashflow": "Cash-flow statement (per fiscal quarter, USD)",
    "profitability": "Profitability & returns",
    "valuation": "Valuation multiples & yields",
    "leverage": "Leverage & coverage",
    "liquidity": "Liquidity",
    "efficiency": "Efficiency / working-capital cycle",
    "pershare": "Per-share quantities",
    "growth": "Growth rates (fraction, period over period)",
    "estimates": "Analyst estimates",
    "events": "Earnings events",
}

# (name, group, source endpoint, vendor keys, description)
_SPECS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # ── labels ──────────────────────────────────────────────────────────────
    ("sector", "labels", "profile", ("sector",), 'GICS-style sector label, e.g. "Technology".'),
    ("industry", "labels", "profile", ("industry",), "finer industry label."),

    # ── size ────────────────────────────────────────────────────────────────
    ("marketCap", "size", "market_cap", ("marketCap",), "market capitalisation in USD (DAILY series)."),
    ("enterpriseValue", "size", "key_metrics", ("enterpriseValue",), "market cap + debt − cash."),

    # ── income statement ────────────────────────────────────────────────────
    ("revenue", "income", "income_statement", ("revenue",), "quarterly revenue."),
    ("costOfRevenue", "income", "income_statement", ("costOfRevenue",), "cost of goods sold."),
    ("grossProfit", "income", "income_statement", ("grossProfit",), "revenue − COGS."),
    ("rdExpense", "income", "income_statement", ("researchAndDevelopmentExpenses",), "R&D spend."),
    ("sgaExpense", "income", "income_statement", ("sellingGeneralAndAdministrativeExpenses",), "SG&A spend."),
    ("operatingExpenses", "income", "income_statement", ("operatingExpenses",), "total operating expenses."),
    ("ebitda", "income", "income_statement", ("ebitda",), "earnings before interest, tax, D&A."),
    ("ebit", "income", "income_statement", ("ebit",), "earnings before interest and tax."),
    ("operatingIncome", "income", "income_statement", ("operatingIncome",), "operating profit."),
    ("interestExpense", "income", "income_statement", ("interestExpense",), "interest paid on debt."),
    ("incomeBeforeTax", "income", "income_statement", ("incomeBeforeTax",), "pre-tax profit."),
    ("incomeTaxExpense", "income", "income_statement", ("incomeTaxExpense",), "tax charge."),
    ("netIncome", "income", "income_statement", ("netIncome",), "bottom-line profit."),
    ("eps", "income", "income_statement", ("eps",), "reported EPS."),
    ("epsDiluted", "income", "income_statement", ("epsDiluted", "epsdiluted"), "diluted EPS."),
    ("sharesOutstanding", "income", "income_statement", ("weightedAverageShsOut",),
     "weighted average shares outstanding — the point-in-time share count."),
    ("sharesDiluted", "income", "income_statement", ("weightedAverageShsOutDil",), "diluted share count."),
    ("depreciationAmortization", "income", "income_statement", ("depreciationAndAmortization",), "D&A charge."),

    # ── balance sheet ───────────────────────────────────────────────────────
    ("cash", "balance", "balance_sheet", ("cashAndCashEquivalents",), "cash and equivalents."),
    ("receivables", "balance", "balance_sheet", ("netReceivables", "accountsReceivables"), "net receivables."),
    ("inventory", "balance", "balance_sheet", ("inventory",), "inventory."),
    ("totalCurrentAssets", "balance", "balance_sheet", ("totalCurrentAssets",), "current assets."),
    ("ppe", "balance", "balance_sheet", ("propertyPlantEquipmentNet",), "net property, plant & equipment."),
    ("goodwill", "balance", "balance_sheet", ("goodwill",), "goodwill."),
    ("intangibleAssets", "balance", "balance_sheet", ("intangibleAssets",), "intangibles ex-goodwill."),
    ("totalAssets", "balance", "balance_sheet", ("totalAssets",), "total assets."),
    ("accountsPayable", "balance", "balance_sheet", ("accountPayables", "accountsPayable"), "payables."),
    ("shortTermDebt", "balance", "balance_sheet", ("shortTermDebt",), "debt due within a year."),
    ("totalCurrentLiabilities", "balance", "balance_sheet", ("totalCurrentLiabilities",), "current liabilities."),
    ("longTermDebt", "balance", "balance_sheet", ("longTermDebt",), "long-term debt."),
    ("totalLiabilities", "balance", "balance_sheet", ("totalLiabilities",), "total liabilities."),
    ("totalDebt", "balance", "balance_sheet", ("totalDebt",), "short + long-term debt."),
    ("totalEquity", "balance", "balance_sheet", ("totalStockholdersEquity", "totalEquity"), "book equity."),
    ("retainedEarnings", "balance", "balance_sheet", ("retainedEarnings",), "cumulative retained profit."),

    # ── cash flow ───────────────────────────────────────────────────────────
    ("operatingCashFlow", "cashflow", "cash_flow",
     ("operatingCashFlow", "netCashProvidedByOperatingActivities"), "cash from operations."),
    ("capex", "cashflow", "cash_flow",
     ("capitalExpenditure", "investmentsInPropertyPlantAndEquipment"), "capital expenditure (negative)."),
    ("freeCashFlow", "cashflow", "cash_flow", ("freeCashFlow",),
     "operating cash flow − capex, in USD (NOT per share)."),
    ("stockBasedCompensation", "cashflow", "cash_flow", ("stockBasedCompensation",), "SBC charge."),
    ("changeInWorkingCapital", "cashflow", "cash_flow", ("changeInWorkingCapital",), "working-capital swing."),
    ("acquisitionsNet", "cashflow", "cash_flow", ("acquisitionsNet",), "cash spent on acquisitions."),
    ("dividendsPaid", "cashflow", "cash_flow", ("commonDividendsPaid", "netDividendsPaid"), "dividends paid (negative)."),
    ("shareRepurchase", "cashflow", "cash_flow", ("commonStockRepurchased",), "buybacks (negative)."),
    ("netStockIssuance", "cashflow", "cash_flow", ("netStockIssuance",), "net equity issued − repurchased."),
    ("netDebtIssuance", "cashflow", "cash_flow", ("netDebtIssuance",), "net debt raised − repaid."),

    # ── profitability & returns ─────────────────────────────────────────────
    ("grossMargin", "profitability", "ratios", ("grossProfitMargin", "grossMargin"), "gross profit / revenue."),
    ("operatingMargin", "profitability", "ratios", ("operatingProfitMargin",), "operating income / revenue."),
    ("ebitMargin", "profitability", "ratios", ("ebitMargin",), "EBIT / revenue."),
    ("ebitdaMargin", "profitability", "ratios", ("ebitdaMargin",), "EBITDA / revenue."),
    ("pretaxMargin", "profitability", "ratios", ("pretaxProfitMargin",), "pre-tax profit / revenue."),
    ("netMargin", "profitability", "ratios", ("netProfitMargin", "netIncomeRatio"), "net income / revenue."),
    ("roa", "profitability", "key_metrics", ("returnOnAssets",), "return on assets."),
    ("roe", "profitability", "key_metrics", ("returnOnEquity", "roe"), "return on equity."),
    ("roic", "profitability", "key_metrics", ("returnOnInvestedCapital", "roic"), "return on invested capital."),
    ("roce", "profitability", "key_metrics", ("returnOnCapitalEmployed",), "return on capital employed."),
    ("incomeQuality", "profitability", "key_metrics", ("incomeQuality",),
     "operating cash flow / net income — accrual-quality proxy."),
    ("rdToRevenue", "profitability", "key_metrics", ("researchAndDevelopementToRevenue",), "R&D intensity."),
    ("sgaToRevenue", "profitability", "key_metrics", ("salesGeneralAndAdministrativeToRevenue",), "SG&A intensity."),
    ("sbcToRevenue", "profitability", "key_metrics", ("stockBasedCompensationToRevenue",), "SBC intensity."),
    ("capexToRevenue", "profitability", "key_metrics", ("capexToRevenue",), "capex intensity."),
    ("capexToOperatingCashFlow", "profitability", "key_metrics", ("capexToOperatingCashFlow",), "reinvestment rate."),
    ("intangiblesToAssets", "profitability", "key_metrics", ("intangiblesToTotalAssets",), "asset intangibility."),

    # ── valuation ───────────────────────────────────────────────────────────
    ("peRatio", "valuation", "ratios", ("priceToEarningsRatio", "peRatio", "priceEarningsRatio"),
     "price / earnings (negative for loss-makers)."),
    ("pegRatio", "valuation", "ratios", ("priceToEarningsGrowthRatio",), "PE / earnings growth."),
    ("pbRatio", "valuation", "ratios", ("priceToBookRatio", "pbRatio"), "price / book."),
    ("psRatio", "valuation", "ratios", ("priceToSalesRatio", "psRatio"), "price / sales."),
    ("pfcfRatio", "valuation", "ratios", ("priceToFreeCashFlowRatio",), "price / free cash flow."),
    ("pocfRatio", "valuation", "ratios", ("priceToOperatingCashFlowRatio",), "price / operating cash flow."),
    ("earningsYield", "valuation", "key_metrics", ("earningsYield",), "earnings / price — the PE inverse."),
    ("freeCashFlowYield", "valuation", "key_metrics", ("freeCashFlowYield",), "FCF / market cap."),
    ("dividendYield", "valuation", "ratios", ("dividendYield",), "dividend / price."),
    ("dividendPayoutRatio", "valuation", "ratios", ("dividendPayoutRatio",), "dividends / net income."),
    ("evToSales", "valuation", "key_metrics", ("evToSales",), "enterprise value / revenue."),
    ("evToEbitda", "valuation", "key_metrics", ("evToEBITDA", "evToEbitda"), "enterprise value / EBITDA."),
    ("evToOperatingCashFlow", "valuation", "key_metrics", ("evToOperatingCashFlow",), "EV / operating cash flow."),
    ("evToFreeCashFlow", "valuation", "key_metrics", ("evToFreeCashFlow",), "EV / free cash flow."),
    ("grahamNumber", "valuation", "key_metrics", ("grahamNumber",), "sqrt(22.5 · EPS · BVPS) value anchor."),

    # ── leverage & coverage ─────────────────────────────────────────────────
    ("debtToEquity", "leverage", "ratios", ("debtToEquityRatio", "debtToEquity"), "total debt / equity."),
    ("debtToAssets", "leverage", "ratios", ("debtToAssetsRatio",), "total debt / assets."),
    ("debtToCapital", "leverage", "ratios", ("debtToCapitalRatio",), "debt / (debt + equity)."),
    ("financialLeverage", "leverage", "ratios", ("financialLeverageRatio",), "assets / equity."),
    ("interestCoverage", "leverage", "ratios", ("interestCoverageRatio",), "EBIT / interest expense."),
    ("netDebtToEbitda", "leverage", "key_metrics", ("netDebtToEBITDA", "netDebtToEbitda"), "net debt / EBITDA."),

    # ── liquidity ───────────────────────────────────────────────────────────
    ("currentRatio", "liquidity", "ratios", ("currentRatio",), "current assets / current liabilities."),
    ("quickRatio", "liquidity", "ratios", ("quickRatio",), "liquid assets / current liabilities."),
    ("cashRatio", "liquidity", "ratios", ("cashRatio",), "cash / current liabilities."),
    ("workingCapital", "liquidity", "key_metrics", ("workingCapital",), "current assets − current liabilities."),
    ("investedCapital", "liquidity", "key_metrics", ("investedCapital",), "capital employed in the business."),

    # ── efficiency / cycle ──────────────────────────────────────────────────
    ("assetTurnover", "efficiency", "ratios", ("assetTurnover",), "revenue / assets."),
    ("receivablesTurnover", "efficiency", "ratios", ("receivablesTurnover",), "revenue / receivables."),
    ("payablesTurnover", "efficiency", "ratios", ("payablesTurnover",), "COGS / payables."),
    ("inventoryTurnover", "efficiency", "ratios", ("inventoryTurnover",), "COGS / inventory."),
    ("daysSalesOutstanding", "efficiency", "key_metrics", ("daysOfSalesOutstanding",), "receivable days."),
    ("daysPayablesOutstanding", "efficiency", "key_metrics", ("daysOfPayablesOutstanding",), "payable days."),
    ("daysInventoryOutstanding", "efficiency", "key_metrics", ("daysOfInventoryOutstanding",), "inventory days."),
    ("cashConversionCycle", "efficiency", "key_metrics", ("cashConversionCycle",), "DSO + DIO − DPO."),
    ("operatingCycle", "efficiency", "key_metrics", ("operatingCycle",), "DSO + DIO."),

    # ── per share ───────────────────────────────────────────────────────────
    ("revenuePerShare", "pershare", "ratios", ("revenuePerShare",), "revenue / shares."),
    ("netIncomePerShare", "pershare", "ratios", ("netIncomePerShare",), "earnings / shares."),
    ("bookValuePerShare", "pershare", "ratios", ("bookValuePerShare",), "book equity / shares."),
    ("tangibleBookValuePerShare", "pershare", "ratios", ("tangibleBookValuePerShare",), "tangible book / shares."),
    ("cashPerShare", "pershare", "ratios", ("cashPerShare",), "cash / shares."),
    ("operatingCashFlowPerShare", "pershare", "ratios", ("operatingCashFlowPerShare",), "OCF / shares."),
    ("freeCashFlowPerShare", "pershare", "ratios", ("freeCashFlowPerShare",), "FCF / shares."),
    ("capexPerShare", "pershare", "ratios", ("capexPerShare",), "capex / shares."),

    # ── growth ──────────────────────────────────────────────────────────────
    ("revenueGrowth", "growth", "financial_growth", ("revenueGrowth",), "revenue growth."),
    ("grossProfitGrowth", "growth", "financial_growth", ("grossProfitGrowth",), "gross profit growth."),
    ("ebitGrowth", "growth", "financial_growth", ("ebitgrowth", "ebitGrowth"), "EBIT growth."),
    ("ebitdaGrowth", "growth", "financial_growth", ("ebitdaGrowth",), "EBITDA growth."),
    ("operatingIncomeGrowth", "growth", "financial_growth", ("operatingIncomeGrowth",), "operating income growth."),
    ("netIncomeGrowth", "growth", "financial_growth", ("netIncomeGrowth",), "net income growth."),
    ("epsGrowth", "growth", "financial_growth", ("epsgrowth", "epsGrowth"), "EPS growth."),
    ("epsDilutedGrowth", "growth", "financial_growth", ("epsdilutedGrowth",), "diluted EPS growth."),
    ("sharesGrowth", "growth", "financial_growth", ("weightedAverageSharesGrowth",),
     "share-count growth — negative means buybacks."),
    ("dividendPerShareGrowth", "growth", "financial_growth", ("dividendsPerShareGrowth",), "DPS growth."),
    ("operatingCashFlowGrowth", "growth", "financial_growth", ("operatingCashFlowGrowth",), "OCF growth."),
    ("freeCashFlowGrowth", "growth", "financial_growth", ("freeCashFlowGrowth",), "FCF growth."),
    ("assetGrowth", "growth", "financial_growth", ("assetGrowth",),
     "total-asset growth — the classic asset-growth anomaly."),
    ("debtGrowth", "growth", "financial_growth", ("debtGrowth",), "debt growth."),
    ("bookValuePerShareGrowth", "growth", "financial_growth", ("bookValueperShareGrowth",), "BVPS growth."),
    ("receivablesGrowth", "growth", "financial_growth", ("receivablesGrowth",), "receivables growth."),
    ("inventoryGrowth", "growth", "financial_growth", ("inventoryGrowth",), "inventory growth."),
    ("rdExpenseGrowth", "growth", "financial_growth", ("rdexpenseGrowth",), "R&D growth."),
    ("sgaExpenseGrowth", "growth", "financial_growth", ("sgaexpensesGrowth",), "SG&A growth."),
    ("revenueGrowth3Y", "growth", "financial_growth", ("threeYRevenueGrowthPerShare",), "3-year revenue/share growth."),
    ("revenueGrowth5Y", "growth", "financial_growth", ("fiveYRevenueGrowthPerShare",), "5-year revenue/share growth."),
    ("netIncomeGrowth3Y", "growth", "financial_growth", ("threeYNetIncomeGrowthPerShare",), "3-year earnings/share growth."),
    ("netIncomeGrowth5Y", "growth", "financial_growth", ("fiveYNetIncomeGrowthPerShare",), "5-year earnings/share growth."),

    # ── estimates & events ──────────────────────────────────────────────────
    ("epsEstimate", "estimates", "earnings", ("epsEstimated", "epsEstimate"), "consensus EPS for the quarter."),
    ("revenueEstimate", "estimates", "earnings", ("revenueEstimated", "estimatedRevenue"), "consensus revenue."),
    ("epsActual", "events", "earnings", ("epsActual", "eps"), "reported EPS at the release."),
    ("revenueActual", "events", "earnings", ("revenueActual", "revenue"), "reported revenue at the release."),
    ("epsSurprise", "events", "earnings", (), "reported EPS − estimate (derived; PEAD signal)."),
    ("revenueSurprise", "events", "earnings", (), "reported revenue − estimate (derived)."),
)

ARCHIVE_FIELD_SPECS: tuple[FieldSpec, ...] = tuple(
    FieldSpec(name=n, group=g, source=s, keys=k, desc=d) for n, g, s, k, d in _SPECS
)

#: canonical name → spec
FIELD_SPECS_BY_NAME: dict[str, FieldSpec] = {s.name: s for s in ARCHIVE_FIELD_SPECS}


def specs_for_source(source: str) -> tuple[FieldSpec, ...]:
    """Every spec filled by one ``fmp_ingest`` endpoint."""
    return tuple(s for s in ARCHIVE_FIELD_SPECS if s.source == source)


def archive_map_for(source: str) -> dict[str, tuple[str, ...]]:
    """``{canonical: candidate vendor keys}`` for one endpoint (derived-only fields omitted)."""
    return {s.name: s.keys for s in specs_for_source(source) if s.keys}


def archive_sources() -> list[str]:
    """Endpoints that fill at least one canonical field."""
    return sorted({s.source for s in ARCHIVE_FIELD_SPECS})


def _names_of_kind(kind: str) -> frozenset[str]:
    return frozenset(s.name for s in ARCHIVE_FIELD_SPECS if s.kind == kind)


FUNDAMENTAL_FIELDS: frozenset[str] = LEGACY_FUNDAMENTAL_FIELDS | _names_of_kind("fundamental")

ESTIMATE_FIELDS: frozenset[str] = frozenset({"epsEstimate", "revenueEstimate"}) | _names_of_kind("estimates")

EVENT_FIELDS: frozenset[str] = frozenset({"epsSurprise"}) | _names_of_kind("events")

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
    # NB: key-metrics/ratios only carry the PER-SHARE figure, so it fills the
    # per-share field.  ``freeCashFlow`` is absolute USD everywhere (the premium
    # archive takes it from the cash-flow statement); mapping the per-share value
    # onto it, as this map used to, silently mislabelled the units.
    "freeCashFlowPerShare": ("freeCashFlowPerShare",),
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
