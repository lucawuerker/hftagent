"""Offline tests for the FMP premium archive → panel path.

Everything here runs against a **synthetic archive** built from payload shapes
captured off the live API, so the suite never needs a key or a network.  The
assertions that matter are the point-in-time ones: a fundamental must be ``NaN``
before its filing date and non-``NaN`` after, and the unfiled endpoints
(``ratios``/``key-metrics``/``financial-growth``) must inherit the *income
statement's* filing date rather than a fixed reporting-lag guess.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_fund_agent.config import Settings
from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS
from quant_fund_agent.data.fmp_ingest.store import Archive
from quant_fund_agent.data.providers.fmp_archive import FMPArchiveProvider

BARS = pd.bdate_range("2020-01-01", "2020-12-31")


def _price_rows(symbol: str, base: float) -> list[dict]:
    return [
        {
            "symbol": symbol, "date": d.strftime("%Y-%m-%d"),
            "adjOpen": base + i * 0.1, "adjHigh": base + i * 0.1 + 1,
            "adjLow": base + i * 0.1 - 1, "adjClose": base + i * 0.1,
            "volume": 1_000_000 + i,
        }
        for i, d in enumerate(BARS)
    ]


def _statement_rows(symbol: str) -> list[dict]:
    """Two quarters; Q1 files late (2020-05-08), Q2 files on 2020-08-07."""
    return [
        {"date": "2020-03-31", "symbol": symbol, "cik": "1", "filingDate": "2020-05-08",
         "acceptedDate": "2020-05-08 18:00:00", "fiscalYear": "2020", "period": "Q1",
         "revenue": 1000.0, "netIncome": 100.0, "eps": 1.0, "epsDiluted": 0.98,
         "weightedAverageShsOut": 100.0, "ebitda": 250.0},
        {"date": "2020-06-30", "symbol": symbol, "cik": "1", "filingDate": "2020-08-07",
         "acceptedDate": "2020-08-07 18:00:00", "fiscalYear": "2020", "period": "Q2",
         "revenue": 1200.0, "netIncome": 150.0, "eps": 1.5, "epsDiluted": 1.47,
         "weightedAverageShsOut": 100.0, "ebitda": 300.0},
    ]


def _ratio_rows(symbol: str) -> list[dict]:
    """Same periods, but NO filing date — the join under test."""
    return [
        {"symbol": symbol, "date": "2020-03-31", "fiscalYear": "2020", "period": "Q1",
         "priceToEarningsRatio": 20.0, "netProfitMargin": 0.10, "currentRatio": 1.5,
         "debtToEquityRatio": 0.8},
        {"symbol": symbol, "date": "2020-06-30", "fiscalYear": "2020", "period": "Q2",
         "priceToEarningsRatio": 18.0, "netProfitMargin": 0.125, "currentRatio": 1.6,
         "debtToEquityRatio": 0.7},
    ]


@pytest.fixture
def archive(tmp_path) -> Archive:
    arc = Archive(tmp_path / "fmp")
    for sym, base in (("AAA", 100.0), ("BBB", 50.0)):
        arc.write(ENDPOINTS["prices_adjusted"], _price_rows(sym, base), symbol=sym)
        arc.write(ENDPOINTS["income_statement"], _statement_rows(sym),
                  symbol=sym, period="quarter")
        arc.write(ENDPOINTS["ratios"], _ratio_rows(sym), symbol=sym, period="quarter")
        arc.write(ENDPOINTS["profile"],
                  [{"symbol": sym, "sector": "Technology", "industry": "Software"}],
                  symbol=sym)
        arc.write(ENDPOINTS["earnings"], [
            {"symbol": sym, "date": "2020-05-08", "epsActual": 1.0, "epsEstimated": 0.8,
             "revenueActual": 1000.0, "revenueEstimated": 950.0},
        ], symbol=sym)
        arc.write(ENDPOINTS["market_cap"], [
            {"symbol": sym, "date": d.strftime("%Y-%m-%d"), "marketCap": 1e9 + i}
            for i, d in enumerate(BARS)
        ], symbol=sym)
    return arc


def _provider(archive: Archive, **overrides) -> FMPArchiveProvider:
    settings = Settings()
    settings.data.provider = "fmp_archive"
    settings.data.start = "2020-01-01"
    settings.data.end = "2020-12-31"
    settings.data.tickers = ["AAA", "BBB"]
    settings.data.membership = None
    settings.data.dtype = "float64"
    settings.data.archive_dir = str(archive.root)
    for key, value in overrides.items():
        setattr(settings.data, key, value)
    provider = FMPArchiveProvider(settings)
    provider.archive = archive          # bypass the env/default root lookup
    return provider


def test_available_fields_reflect_what_is_on_disk(archive):
    fields = _provider(archive).available_fields()
    assert {"open", "high", "low", "close", "volume"} <= fields
    assert {"revenue", "netIncome", "eps", "peRatio", "sector", "marketCap"} <= fields
    # Nothing was archived for these endpoints, so their fields must not be
    # advertised — factor gating has to stay honest about a partial download.
    assert "assetGrowth" not in fields        # financial_growth absent
    assert "operatingCashFlow" not in fields  # cash_flow absent


def test_prices_load_as_a_wide_panel(archive):
    panel = _provider(archive).load(fields=["close", "volume"])
    assert set(panel) == {"close", "volume"}
    assert list(panel["close"].columns) == ["AAA", "BBB"]
    assert len(panel["close"]) == len(BARS)
    assert panel["close"].notna().all().all()


def test_fundamentals_are_point_in_time(archive):
    panel = _provider(archive).load(fields=["revenue", "close"])
    revenue = panel["revenue"]["AAA"]
    # Q1 (period end 2020-03-31) was filed on 2020-05-08.
    assert revenue.loc[:"2020-05-07"].isna().all(), "value visible before its filing date"
    assert revenue.loc["2020-05-08":"2020-08-06"].eq(1000.0).all()
    assert revenue.loc["2020-08-07":].eq(1200.0).all()


def test_unfiled_endpoints_inherit_the_statement_filing_date(archive):
    """`ratios` has no filingDate; it must NOT fall back to period_end + 60d."""
    panel = _provider(archive).load(fields=["peRatio", "close"])
    pe = panel["peRatio"]["AAA"]
    assert pe.loc[:"2020-05-07"].isna().all()
    assert pe.loc["2020-05-08"] == 20.0
    # The generic fallback would have been 2020-03-31 + 60d = 2020-05-30, i.e. it
    # would have hidden the value for three weeks after it was actually public.
    assert pe.loc["2020-05-11"] == 20.0


def test_reporting_lag_fallback_when_no_statement_matches(archive):
    """With no matching filed statement, the lag fallback still applies."""
    archive.write(
        ENDPOINTS["ratios"],
        [{"symbol": "AAA", "date": "2019-12-31", "fiscalYear": "2019", "period": "Q4",
          "priceToEarningsRatio": 25.0}],
        symbol="AAA", period="quarter", merge=True,
    )
    panel = _provider(archive, reporting_lag_days=45).load(fields=["peRatio", "close"])
    pe = panel["peRatio"]["AAA"]
    assert pe.loc["2020-02-14"] == 25.0        # 2019-12-31 + 45d
    assert pd.isna(pe.loc["2020-02-13"])


def test_a_filing_date_at_the_period_end_is_rejected_as_a_leak(archive):
    """FMP backfills old statements with `filingDate == date` — a 2-month leak.

    Observed live: AAPL/ATVI carry `filingDate == date` for every quarter up to
    1994. Taking it at face value would say the company filed the instant its
    quarter closed, so it must fall back to the conservative lag instead.
    """
    archive.write(
        ENDPOINTS["income_statement"],
        [{"date": "2020-09-30", "symbol": "AAA", "filingDate": "2020-09-30",
          "acceptedDate": "2020-09-30 00:00:00", "fiscalYear": "2020",
          "period": "Q3", "revenue": 1500.0}],
        symbol="AAA", period="quarter", merge=True,
    )
    revenue = _provider(archive, reporting_lag_days=60).load(
        fields=["revenue", "close"])["revenue"]["AAA"]
    # Taking the period-end filing date literally would show Q3 from 2020-10-01;
    # the guard keeps the previous quarter's number visible until the lag elapses.
    assert revenue.loc["2020-10-01"] == 1200.0, "period-end filing date taken literally"
    assert revenue.loc["2020-11-30"] == 1500.0          # 2020-09-30 + 60d


def test_earnings_surprise_is_derived_and_stamped_at_the_report(archive):
    panel = _provider(archive).load(fields=["epsSurprise", "revenueSurprise", "close"])
    surprise = panel["epsSurprise"]["AAA"]
    assert pd.isna(surprise.loc["2020-05-07"])
    assert surprise.loc["2020-05-08"] == pytest.approx(0.2)
    assert panel["revenueSurprise"]["AAA"].loc["2020-05-08"] == pytest.approx(50.0)


def test_categorical_labels_survive_as_text(archive):
    panel = _provider(archive).load(fields=["sector", "close"])
    assert panel["sector"]["AAA"].iloc[0] == "Technology"


def test_daily_market_cap_is_not_a_quarterly_step(archive):
    panel = _provider(archive).load(fields=["marketCap", "close"])
    cap = panel["marketCap"]["AAA"]
    assert cap.notna().all()
    assert cap.nunique() == len(BARS)  # a genuine daily series, not forward-filled


def test_targeted_load_skips_unrequested_fundamentals(archive, monkeypatch):
    """`fields=` must gate work *before* materialising — the memory guarantee."""
    provider = _provider(archive)
    calls: list[str] = []
    original = provider._records_for_symbol

    def spy(symbol, sources, lag):
        calls.append(",".join(sources))
        return original(symbol, sources, lag)

    monkeypatch.setattr(provider, "_records_for_symbol", spy)
    provider.load(fields=["close", "revenue"])
    assert calls and all(c == "income_statement" for c in calls)


def test_fundamentals_off_switch(archive, monkeypatch):
    monkeypatch.setenv("QF_FUNDAMENTALS", "0")
    provider = _provider(archive)
    assert "revenue" not in provider.available_fields()
    assert provider.load(fields=["close", "revenue"]).keys() == {"close"}


def test_empty_archive_raises_a_pointer_to_the_downloader(tmp_path):
    provider = _provider(Archive(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="fmp_bulk_download"):
        provider.load(fields=["close"])
