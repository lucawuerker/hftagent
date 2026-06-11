"""Offline tests for the non-OHLCV (fundamentals / estimates / events) layer.

Everything here runs without network or API keys: vendor JSON is synthetic, and
the providers' network methods are monkeypatched.  The focus is the parts that
are easy to get silently wrong — **point-in-time availability** (no look-ahead),
forward-fill alignment with a staleness cap, per-vendor field normalization, the
``indneutralize`` wide-frame fix, capability gating, and the end-to-end panel
merge.

Run: ``PYTHONPATH=. ./venv/bin/pytest tests/test_fundamentals.py``
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.fields import coerce_numeric, normalize, pick
from quant_fund_agent.data.fundamentals import (
    align_to_index,
    availability_date,
    build_record_frame,
    parse_date,
)


# ── synthetic helpers ────────────────────────────────────────────────────────

def _ohlcv(symbol: str, start="2023-01-02", end="2023-09-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    base = 100 + np.arange(len(idx), dtype=float)
    return pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1,
         "close": base + 0.5, "volume": np.full(len(idx), 1e6)},
        index=idx,
    )


def _record_frame(rows: list[dict]) -> pd.DataFrame:
    return build_record_frame(rows)


# ── 1. availability stamping (the PIT date) ──────────────────────────────────

def test_availability_uses_filing_date_when_present():
    av = availability_date("2023-05-15", "2023-03-31", reporting_lag_days=60)
    assert av == pd.Timestamp("2023-05-15")


def test_availability_falls_back_to_fiscal_end_plus_lag():
    av = availability_date(None, "2023-03-31", reporting_lag_days=60)
    assert av == pd.Timestamp("2023-03-31") + pd.Timedelta(days=60)


def test_availability_none_when_no_dates():
    assert availability_date(None, None, reporting_lag_days=60) is None
    assert availability_date("not-a-date", "also-bad", reporting_lag_days=60) is None


def test_parse_date_handles_garbage():
    assert parse_date("2023-01-05") == pd.Timestamp("2023-01-05")
    assert parse_date(None) is None
    assert parse_date("nonsense") is None


# ── 2. build_record_frame ────────────────────────────────────────────────────

def test_build_record_frame_sorts_and_drops_undated():
    rows = [
        {"availability": pd.Timestamp("2023-08-14"), "peRatio": 22.0},
        {"availability": None, "peRatio": 99.0},  # dropped
        {"availability": pd.Timestamp("2023-05-04"), "peRatio": 20.0},
    ]
    frame = _record_frame(rows)
    assert list(frame.index) == [pd.Timestamp("2023-05-04"), pd.Timestamp("2023-08-14")]
    assert frame["peRatio"].tolist() == [20.0, 22.0]


def test_build_record_frame_merges_same_date_columns():
    rows = [
        {"availability": pd.Timestamp("2023-05-04"), "revenue": 1e9},
        {"availability": pd.Timestamp("2023-05-04"), "epsSurprise": 0.1},
    ]
    frame = _record_frame(rows)
    assert len(frame) == 1
    assert frame.loc[pd.Timestamp("2023-05-04"), "revenue"] == 1e9
    assert frame.loc[pd.Timestamp("2023-05-04"), "epsSurprise"] == 0.1


def test_build_record_frame_empty():
    assert build_record_frame([]) is None
    assert build_record_frame([{"availability": None}]) is None


# ── 3. alignment is point-in-time (no look-ahead) ────────────────────────────

def test_alignment_is_point_in_time():
    frame = _record_frame([
        {"availability": pd.Timestamp("2023-05-15"), "peRatio": 20.0},
        {"availability": pd.Timestamp("2023-08-14"), "peRatio": 22.0},
    ])
    idx = pd.bdate_range("2023-05-01", "2023-09-01")
    pe = align_to_index({"AAA": frame}, idx, staleness_cap_days=400)["peRatio"]["AAA"]
    # invisible before the filing date — the core guarantee
    assert pe.loc[:"2023-05-12"].isna().all()
    assert pe.loc["2023-05-15"] == 20.0
    assert pe.loc["2023-08-01"] == 20.0   # holds the Q1 value
    assert pe.loc["2023-08-15"] == 22.0   # steps to Q2 after its filing


def test_alignment_fields_fill_independently():
    frame = _record_frame([
        {"availability": pd.Timestamp("2023-05-04"), "revenue": 1e9},
        {"availability": pd.Timestamp("2023-06-15"), "epsEstimate": 1.4},
    ])
    idx = pd.bdate_range("2023-05-01", "2023-07-01")
    out = align_to_index({"AAA": frame}, idx, staleness_cap_days=400)
    assert not np.isnan(out["revenue"]["AAA"].loc["2023-05-10"])
    assert np.isnan(out["epsEstimate"]["AAA"].loc["2023-05-10"])  # not yet available
    assert out["epsEstimate"]["AAA"].loc["2023-06-20"] == 1.4


def test_alignment_staleness_cap_blanks_old_values():
    frame = _record_frame([{"availability": pd.Timestamp("2023-05-15"), "peRatio": 20.0}])
    idx = pd.bdate_range("2023-05-01", "2023-09-01")
    pe = align_to_index({"AAA": frame}, idx, staleness_cap_days=30)["peRatio"]["AAA"]
    assert pe.loc["2023-05-16"] == 20.0           # fresh
    assert np.isnan(pe.loc["2023-06-30"])         # 46 days old → blanked


def test_alignment_categorical_is_string_and_never_stale():
    # sector stamped at the far-past sentinel must survive any staleness cap.
    frame = _record_frame([{"availability": pd.Timestamp("1990-01-01"), "sector": "Tech"}])
    idx = pd.bdate_range("2023-05-01", "2023-06-01")
    out = align_to_index({"AAA": frame}, idx, staleness_cap_days=30)
    sec = out["sector"]["AAA"]
    assert (sec == "Tech").all()
    # string-typed, NOT coerced to float (pandas may use object or StringDtype)
    assert not pd.api.types.is_float_dtype(sec)
    assert isinstance(sec.iloc[-1], str)


# ── 4. field normalization (FMP vs AV → canonical) ───────────────────────────

def test_normalize_collapses_vendor_names_to_canonical():
    from quant_fund_agent.data.fields import AV_PROFILE_MAP, FMP_METRICS_MAP

    fmp = normalize({"peRatio": "28.0", "returnOnEquity": 1.4}, FMP_METRICS_MAP)
    assert fmp["peRatio"] == 28.0 and fmp["roe"] == 1.4
    # AV TitleCase keys → same canonical labels
    av = normalize({"Sector": "TECHNOLOGY", "Industry": "COMPUTERS"}, AV_PROFILE_MAP)
    assert av["sector"] == "TECHNOLOGY" and av["industry"] == "COMPUTERS"


def test_normalize_skips_missing_tokens():
    from quant_fund_agent.data.fields import FMP_METRICS_MAP

    out = normalize({"peRatio": "None", "roe": "-", "currentRatio": ""}, FMP_METRICS_MAP)
    assert out == {}


def test_coerce_numeric_and_pick():
    assert coerce_numeric("3.5") == 3.5
    assert coerce_numeric("None") is None
    assert coerce_numeric(None) is None
    assert pick({"a": None, "b": "x"}, ("a", "b")) == "x"
    assert pick({"a": "None"}, ("a",)) is None


# ── 5. FMP reshapes ──────────────────────────────────────────────────────────

def test_fmp_reshapes_produce_pit_rows():
    from quant_fund_agent.data.providers import fmp

    rows = []
    rows += fmp._profile_rows([{"sector": "Technology", "industry": "Consumer Electronics"}])
    rows += fmp._metric_rows([{"date": "2023-03-31", "peRatio": 28.0, "roe": 1.4}], 60)
    rows += fmp._income_rows(
        [{"date": "2023-03-31", "fillingDate": "2023-05-04", "revenue": 9.4e10, "eps": 1.52}], 60)
    rows += fmp._earnings_rows(
        [{"date": "2023-05-04", "eps": 1.52, "epsEstimated": 1.43, "revenueEstimated": 9.3e10}])
    frame = build_record_frame(rows)

    assert frame.loc[pd.Timestamp("1990-01-01"), "sector"] == "Technology"
    # key-metrics has no filing date → fiscal-end + 60d lag
    assert pd.Timestamp("2023-05-30") in frame.index
    assert frame.loc[pd.Timestamp("2023-05-30"), "peRatio"] == 28.0
    # income filing date used directly; earnings surprise derived
    assert frame.loc[pd.Timestamp("2023-05-04"), "revenue"] == 9.4e10
    assert frame.loc[pd.Timestamp("2023-05-04"), "epsSurprise"] == pytest.approx(0.09)
    assert frame.loc[pd.Timestamp("2023-05-04"), "epsEstimate"] == 1.43


def test_fmp_error_payload_raises():
    from quant_fund_agent.data.providers import fmp

    with pytest.raises(RuntimeError):
        fmp._as_rows({"Error Message": "Invalid API KEY"})


# ── 6. AlphaVantage reshapes ─────────────────────────────────────────────────

def test_av_reshapes_produce_pit_rows():
    from quant_fund_agent.data.providers import alphavantage as av

    rows = []
    rows += av._av_profile_rows({"Sector": "TECHNOLOGY", "Industry": "COMPUTERS", "PERatio": "28"})
    rows += av._av_income_rows(
        {"quarterlyReports": [
            {"fiscalDateEnding": "2023-03-31", "totalRevenue": "94800000000",
             "netIncome": "24160000000"}]}, 60)
    rows += av._av_earnings_rows(
        {"quarterlyEarnings": [
            {"fiscalDateEnding": "2023-03-31", "reportedDate": "2023-05-04",
             "reportedEPS": "1.52", "estimatedEPS": "1.43"}]})
    frame = build_record_frame(rows)

    # static label only — the leaky snapshot PERatio must NOT be carried
    assert frame.loc[pd.Timestamp("1990-01-01"), "sector"] == "TECHNOLOGY"
    assert "peRatio" not in frame.columns
    # netMargin derived = netIncome / totalRevenue, stamped fiscal-end + lag
    nm = frame.loc[pd.Timestamp("2023-05-30"), "netMargin"]
    assert nm == pytest.approx(24160000000 / 94800000000)
    # earnings at the real reportedDate
    assert frame.loc[pd.Timestamp("2023-05-04"), "epsSurprise"] == pytest.approx(0.09)
    assert frame.loc[pd.Timestamp("2023-05-04"), "eps"] == 1.52


# ── 7. capability gating with the new tiers ──────────────────────────────────

def test_required_tier_for_new_fields():
    from quant_fund_agent.data.tiers import required_tier

    assert required_tier(["peRatio"]) == "fundamental"
    assert required_tier(["epsEstimate"]) == "estimates"
    assert required_tier(["epsSurprise"]) == "events"
    assert required_tier(["close"]) == "standard"


def test_gating_admits_fundamental_only_on_capable_provider():
    from quant_fund_agent.data.tiers import TIERS, compatible_factors, is_compatible

    assert not is_compatible(["peRatio"], TIERS["standard"])
    assert is_compatible(["peRatio"], TIERS["standard"] | TIERS["fundamental"])

    records = [
        {"id": "ohlcv", "required_inputs": ["close", "volume"]},
        {"id": "value", "required_inputs": ["peRatio"]},
        {"id": "surprise", "required_inputs": ["epsSurprise"]},
    ]
    std = {r["id"] for r in compatible_factors(records, TIERS["standard"])}
    assert std == {"ohlcv"}
    rich = {r["id"] for r in compatible_factors(
        records, TIERS["standard"] | TIERS["fundamental"] | TIERS["events"])}
    assert rich == {"ohlcv", "value", "surprise"}


def test_provider_available_fields_is_asset_class_aware():
    from quant_fund_agent.data.panel import get_provider

    eq = get_provider(Settings(data=DataSettings(provider="fmp", asset_class="equity")))
    cr = get_provider(Settings(data=DataSettings(provider="fmp", asset_class="crypto")))
    assert "peRatio" in eq.available_fields()
    assert "peRatio" not in cr.available_fields()
    av = get_provider(Settings(data=DataSettings(provider="alphavantage", asset_class="equity")))
    assert {"sector", "epsSurprise"} <= av.available_fields()


# ── 8. records cache (TTL freshness) ─────────────────────────────────────────

def test_cached_records_ttl_refetch(tmp_path):
    from quant_fund_agent.data.cache import _records_path, cached_records

    calls = {"n": 0}

    def fetch(syms):
        calls["n"] += 1
        return {s: _record_frame([
            {"availability": pd.Timestamp("2023-05-04"), "peRatio": 20.0}]) for s in syms}

    kw = dict(provider="fmp", asset_class="equity", group="fundamentals",
              cache_dir=str(tmp_path), fetch_fn=fetch, ttl_days=90)
    cached_records(symbols=["AAA"], **kw)
    cached_records(symbols=["AAA"], **kw)            # fresh → served from cache
    assert calls["n"] == 1
    # age the cache file past the TTL → refetch
    path = _records_path(str(tmp_path), "fmp", "equity", "fundamentals", "AAA")
    old = time.time() - 200 * 86400
    os.utime(path, (old, old))
    cached_records(symbols=["AAA"], **kw)
    assert calls["n"] == 2


def test_records_cache_path_is_separate_from_ohlcv(tmp_path):
    from quant_fund_agent.data.cache import _records_path, _symbol_path

    ohlcv = _symbol_path(str(tmp_path), "fmp", "equity", "1d", "AAA")
    recs = _records_path(str(tmp_path), "fmp", "equity", "fundamentals", "AAA")
    assert "1d" in str(ohlcv) and "fundamentals" in str(recs)
    assert ohlcv != recs


# ── 9. indneutralize accepts the wide panel frame ────────────────────────────

def test_indneutralize_series_and_wide_match():
    from quant_fund_agent.factors.ops import indneutralize

    idx = pd.bdate_range("2023-01-02", periods=6)
    cols = ["A", "B", "C", "D"]
    df = pd.DataFrame(np.arange(24.0).reshape(6, 4), index=idx, columns=cols)
    ser = pd.Series({"A": "Tech", "B": "Tech", "C": "Fin", "D": "Fin"})
    wide = pd.DataFrame({c: ser[c] for c in cols}, index=idx)
    assert np.allclose(indneutralize(df, ser).values, indneutralize(df, wide).values)


def test_indneutralize_leaves_unlabelled_tickers_alone():
    from quant_fund_agent.factors.ops import indneutralize

    idx = pd.bdate_range("2023-01-02", periods=4)
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.arange(12.0).reshape(4, 3), index=idx, columns=cols)
    wide = pd.DataFrame({"A": "Tech", "B": "Tech", "C": np.nan}, index=idx)
    out = indneutralize(df, wide)
    assert np.allclose(out["C"].values, df["C"].values)  # untouched


# ── 10. end-to-end panel merge + look-ahead truncation ───────────────────────

def _wire_fmp(monkeypatch, fund=True):
    from quant_fund_agent.data.providers.fmp import FMPProvider

    monkeypatch.setattr(FMPProvider, "_fetch",
                        lambda self, syms: {s: _ohlcv(s) for s in syms})
    if fund:
        def fake_fund(self, syms):
            return {s: _record_frame([
                {"availability": pd.Timestamp("1990-01-01"), "sector": "Tech"},
                {"availability": pd.Timestamp("2023-05-04"),
                 "peRatio": 20.0, "roe": 0.3, "epsSurprise": 0.1},
            ]) for s in syms}
        monkeypatch.setattr(FMPProvider, "_fetch_fundamentals", fake_fund)


def _settings(tmp_path, **kw):
    base = dict(provider="fmp", asset_class="equity", tickers=["AAA", "BBB"],
                start="2023-01-02", end="2023-09-01", frequency="1d",
                cache_dir=str(tmp_path))
    base.update(kw)
    return Settings(data=DataSettings(**base))


def test_full_panel_merges_ohlcv_and_fundamentals(monkeypatch, tmp_path):
    from quant_fund_agent.data.frequency import periods_per_year_from_index

    _wire_fmp(monkeypatch)
    panel = load_panel(settings=_settings(tmp_path))
    assert {"open", "high", "low", "close", "volume", "vwap", "returns"} <= set(panel)
    assert {"sector", "peRatio", "roe", "epsSurprise"} <= set(panel)
    # OHLCV annualization unchanged by the enrichment
    assert periods_per_year_from_index(panel["close"].index) == 252
    pe = panel["peRatio"]["AAA"]
    assert pe.loc[:"2023-05-01"].isna().all()      # PIT preserved through the panel
    assert pe.loc["2023-06-01"] == 20.0


def test_truncate_as_of_hides_future_fundamentals(monkeypatch, tmp_path):
    from quant_fund_agent.modeling.service import _truncate_as_of

    _wire_fmp(monkeypatch)
    panel = load_panel(settings=_settings(tmp_path))
    # cut off BEFORE the 2023-05-04 filing → peRatio must be entirely unseen
    cut = _truncate_as_of(panel, "2023-05-01")
    assert cut["peRatio"].notna().sum().sum() == 0
    assert cut["close"].notna().any().any()        # prices still present


def test_targeted_load_skips_fundamental_fetch(monkeypatch, tmp_path):
    from quant_fund_agent.data.providers.fmp import FMPProvider

    _wire_fmp(monkeypatch, fund=False)
    called = {"n": 0}
    monkeypatch.setattr(
        FMPProvider, "_fetch_fundamentals",
        lambda self, syms: called.__setitem__("n", called["n"] + 1) or {})
    panel = load_panel(fields=["close", "vwap"], settings=_settings(tmp_path))
    assert set(panel) == {"close", "vwap"}
    assert called["n"] == 0  # no non-OHLCV field requested → no fundamentals call


def test_fundamentals_disabled_by_flag(monkeypatch, tmp_path):
    _wire_fmp(monkeypatch)
    panel = load_panel(settings=_settings(tmp_path, fundamentals=False))
    assert "peRatio" not in panel and "close" in panel


def test_fundamentals_disabled_for_crypto(monkeypatch, tmp_path):
    from quant_fund_agent.data.providers.fmp import FMPProvider

    monkeypatch.setattr(FMPProvider, "_fetch",
                        lambda self, syms: {s: _ohlcv(s) for s in syms})
    monkeypatch.setattr(
        FMPProvider, "_fetch_fundamentals",
        lambda self, syms: (_ for _ in ()).throw(AssertionError("should not fetch")))
    s = _settings(tmp_path, asset_class="crypto", tickers=["BTC-USD"])
    panel = load_panel(settings=s)
    assert "peRatio" not in panel


def test_qf_fundamentals_env_off(monkeypatch, tmp_path):
    _wire_fmp(monkeypatch)
    monkeypatch.setenv("QF_FUNDAMENTALS", "0")
    panel = load_panel(settings=_settings(tmp_path))
    assert "peRatio" not in panel


# ── 10b. provider fetch orchestration (offline, request_json patched) ────────

def test_fmp_fetch_fundamentals_orchestration(monkeypatch):
    """`_fetch_fundamentals` wires the five endpoints into one PIT frame — no net."""
    from quant_fund_agent.data.providers import fmp

    monkeypatch.setenv("FMP_API_KEY", "test")
    payloads = {
        "profile": [{"sector": "Technology", "industry": "Consumer Electronics"}],
        "key-metrics": [{"date": "2023-03-31", "peRatio": 28.0, "roe": 1.4,
                         "marketCap": 2.6e12}],
        "ratios": [{"date": "2023-03-31", "grossProfitMargin": 0.43,
                    "netProfitMargin": 0.24}],
        "income-statement": [{"date": "2023-03-31", "fillingDate": "2023-05-04",
                              "revenue": 9.4e10, "eps": 1.52}],
        "earnings": [{"date": "2023-05-04", "eps": 1.52, "epsEstimated": 1.43,
                      "revenueEstimated": 9.3e10}],
    }

    def fake_request(url, params, **kw):
        endpoint = url.rsplit("/", 1)[-1]
        return payloads[endpoint]

    monkeypatch.setattr(fmp, "request_json", fake_request)
    prov = fmp.FMPProvider(Settings(data=DataSettings(provider="fmp")))
    out = prov._fetch_fundamentals(["AAPL"])
    frame = out["AAPL"]
    assert {"sector", "peRatio", "roe", "marketCap", "grossMargin", "revenue",
            "eps", "epsEstimate", "epsSurprise"} <= set(frame.columns)
    assert frame["epsSurprise"].dropna().iloc[0] == pytest.approx(0.09)


def test_fmp_fund_get_degrades_on_endpoint_error(monkeypatch):
    """A single failing endpoint must not lose the whole symbol."""
    from quant_fund_agent.data.providers import fmp

    monkeypatch.setenv("FMP_API_KEY", "test")

    def fake_request(url, params, **kw):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "profile":
            raise RuntimeError("boom")  # one endpoint down
        if endpoint == "earnings":
            return [{"date": "2023-05-04", "eps": 1.5, "epsEstimated": 1.4}]
        return []

    monkeypatch.setattr(fmp, "request_json", fake_request)
    prov = fmp.FMPProvider(Settings(data=DataSettings(provider="fmp")))
    out = prov._fetch_fundamentals(["AAPL"])
    assert "epsSurprise" in out["AAPL"].columns  # earnings still made it through


def test_fmp_fund_get_propagates_rate_limit(monkeypatch):
    from quant_fund_agent.data.providers import fmp
    from quant_fund_agent.data.providers._http import RateLimited

    monkeypatch.setenv("FMP_API_KEY", "test")

    def fake_request(url, params, **kw):
        raise RateLimited("fmp: too many requests")

    monkeypatch.setattr(fmp, "request_json", fake_request)
    prov = fmp.FMPProvider(Settings(data=DataSettings(provider="fmp")))
    with pytest.raises(RateLimited):
        prov._fetch_fundamentals(["AAPL"])


def test_fmp_earnings_row_without_date_skipped():
    from quant_fund_agent.data.providers import fmp

    rows = fmp._earnings_rows([
        {"eps": 1.5, "epsEstimated": 1.4},                       # no date → skipped
        {"date": "2023-05-04", "eps": 1.5, "epsEstimated": 1.4},  # kept
    ])
    assert len(rows) == 1
    assert rows[0]["availability"] == pd.Timestamp("2023-05-04")


def test_fmp_fetch_fundamentals_needs_key(monkeypatch):
    from quant_fund_agent.data.providers import fmp

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    prov = fmp.FMPProvider(Settings(data=DataSettings(provider="fmp")))
    with pytest.raises(ValueError):
        prov._fetch_fundamentals(["AAPL"])


def test_av_fetch_fundamentals_orchestration(monkeypatch):
    from quant_fund_agent.data.providers import alphavantage as av

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test")
    payloads = {
        "OVERVIEW": {"Sector": "TECHNOLOGY", "Industry": "COMPUTERS"},
        "INCOME_STATEMENT": {"quarterlyReports": [
            {"fiscalDateEnding": "2023-03-31", "totalRevenue": "9.4e10",
             "netIncome": "2.4e10"}]},
        "EARNINGS": {"quarterlyEarnings": [
            {"fiscalDateEnding": "2023-03-31", "reportedDate": "2023-05-04",
             "reportedEPS": "1.52", "estimatedEPS": "1.43"}]},
    }
    monkeypatch.setattr(av, "request_json",
                        lambda url, params, **kw: payloads[params["function"]])
    prov = av.AlphaVantageProvider(Settings(data=DataSettings(provider="alphavantage")))
    frame = prov._fetch_fundamentals(["AAPL"])["AAPL"]
    assert {"sector", "revenue", "netMargin", "eps", "epsEstimate",
            "epsSurprise"} <= set(frame.columns)
    assert "peRatio" not in frame.columns  # no leaky snapshot ratio


def test_av_rate_limit_propagates(monkeypatch):
    from quant_fund_agent.data.providers import alphavantage as av
    from quant_fund_agent.data.providers._http import RateLimited

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test")
    monkeypatch.setattr(
        av, "request_json",
        lambda url, params, **kw: {"Note": "Thank you for using Alpha Vantage! 25/day"})
    prov = av.AlphaVantageProvider(Settings(data=DataSettings(provider="alphavantage")))
    with pytest.raises(RateLimited):
        prov._fetch_fundamentals(["AAPL"])


# ── 11. the example factors compute on a fundamental panel ───────────────────

def test_example_fundamental_factors_produce_signals(monkeypatch, tmp_path):
    from quant_fund_agent.factors._discover import discover_factors
    from quant_fund_agent.factors.registry import instantiate_factor

    discover_factors()
    _wire_fmp(monkeypatch)
    panel = load_panel(settings=_settings(tmp_path))
    for fid in ("value_earnings_yield", "quality_roe", "earnings_surprise_drift"):
        sig = instantiate_factor(fid).calc(panel)
        assert sig.shape == panel["close"].shape
        # a real signal after the fundamentals become available
        assert sig.loc["2023-06-01"].notna().any(), fid


def test_example_factors_degrade_when_field_absent():
    """Advertised-but-undelivered field (e.g. FMP free tier) → no crash, all-NaN.

    Gating admits a factor when the provider *advertises* its field, but a vendor
    can still fail to deliver it (paywalled endpoint); the factor must degrade
    rather than KeyError.
    """
    from quant_fund_agent.factors._discover import discover_factors
    from quant_fund_agent.factors.registry import instantiate_factor

    discover_factors()
    idx = pd.bdate_range("2023-01-02", periods=10)
    ohlcv_only = {"close": pd.DataFrame(
        100.0, index=idx, columns=["AAA", "BBB"])}  # no peRatio/roe/epsSurprise
    for fid in ("value_earnings_yield", "quality_roe", "earnings_surprise_drift"):
        sig = instantiate_factor(fid).calc(ohlcv_only)
        assert sig.shape == ohlcv_only["close"].shape
        assert sig.isna().all().all(), fid  # blank, not a crash
