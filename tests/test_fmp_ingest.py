"""Offline tests for the FMP bulk-ingest machinery.

No network and no API key: ``requests.Session.get`` is replaced by a scripted
fake, so every branch that matters for a multi-hour download — plan restriction
vs rate limit vs transient error, resume after a kill, window chunking, symbol
resolution for delisted names — is exercised deterministically.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from quant_fund_agent.data.fmp_ingest.capabilities import Capabilities, probe
from quant_fund_agent.data.fmp_ingest.client import FMPClient, RateLimiter
from quant_fund_agent.data.fmp_ingest.constituents import (
    build_intervals,
    parse_changes,
    walk_backward,
)
from quant_fund_agent.data.fmp_ingest.download import DownloadConfig, Downloader
from quant_fund_agent.data.fmp_ingest.endpoints import (
    ENDPOINTS,
    endpoints_for_groups,
    work_units,
)
from quant_fund_agent.data.fmp_ingest.store import (
    STATUS_OK,
    STATUS_RESTRICTED,
    Archive,
    ManifestEntry,
    coerce_frame,
)
from quant_fund_agent.data.fmp_ingest.symbols import (
    candidate_symbols,
    spell_coverage,
    spell_windows,
    summarise,
)
from quant_fund_agent.data.membership import members_from_spells


# ── a scripted HTTP double ───────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status: int, payload, *, text: str = "", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload if payload is not None else [])
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Answers by URL suffix; records every call for assertions."""

    def __init__(self, routes: dict, default=None):
        self.routes = routes
        self.default = default if default is not None else FakeResponse(200, [])
        self.calls: list[tuple[str, dict]] = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):  # noqa: ANN001
        self.calls.append((url, dict(params or {})))
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                if callable(response):
                    return response(len([c for c in self.calls if c[0].endswith(suffix)]))
                return response
        return self.default


@pytest.fixture
def patch_session(monkeypatch):
    def _install(session: FakeSession):
        monkeypatch.setattr(
            "quant_fund_agent.data.fmp_ingest.client.requests.Session",
            lambda: session,
        )
        return session
    return _install


def _client(**kwargs) -> FMPClient:
    return FMPClient("test-key", per_minute=100000, retries=3, **kwargs)


# ── rate limiter ─────────────────────────────────────────────────────────────

def test_rate_limiter_admits_up_to_the_budget_without_blocking():
    limiter = RateLimiter(per_minute=5)
    t0 = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - t0 < 0.5


def test_rate_limiter_blocks_once_the_window_is_full(monkeypatch):
    """A third call inside the window must wait for the oldest one to age out."""
    limiter = RateLimiter(per_minute=2)

    class Slept(Exception):
        pass

    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)
        raise Slept  # stop the retry loop; we only care that it waited

    monkeypatch.setattr("quant_fund_agent.data.fmp_ingest.client.time.sleep", fake_sleep)
    limiter.acquire()
    limiter.acquire()
    limiter._times[0] -= 59.9   # the oldest call is now 59.9s old → ~0.1s to wait
    with pytest.raises(Slept):
        limiter.acquire()
    assert len(slept) == 1 and 0 < slept[0] <= 0.5


# ── client classification ────────────────────────────────────────────────────

def test_plan_restriction_is_terminal_and_not_retried(patch_session, monkeypatch):
    monkeypatch.setattr("quant_fund_agent.data.fmp_ingest.client.time.sleep", lambda s: None)
    session = patch_session(FakeSession({"/income-statement": FakeResponse(
        402, None,
        text="Restricted Endpoint: This endpoint is not available under your "
             "current subscription")}))
    client = _client()
    result = client.get("https://x/stable/income-statement", {"symbol": "AAPL"})
    assert result.restricted and not result.ok
    assert len(session.calls) == 1, "a plan gate must not burn retries"
    assert client.stats()["restricted"] == 1


def test_rate_limit_backs_off_then_succeeds(patch_session, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("quant_fund_agent.data.fmp_ingest.client.time.sleep", sleeps.append)

    def route(n):
        return FakeResponse(429, None, text="limit", headers={"Retry-After": "3"}) \
            if n == 1 else FakeResponse(200, [{"date": "2020-01-02", "adjClose": 1.0}])

    session = patch_session(FakeSession({"/prices": route}))
    result = _client().get("https://x/stable/prices", {})
    assert result.ok and len(result.rows) == 1
    assert len(session.calls) == 2 and 3.0 in sleeps  # honoured Retry-After


def test_vendor_error_dict_is_a_failure_not_a_row(patch_session):
    patch_session(FakeSession({"/x": FakeResponse(200, {"Error Message": "bad symbol"})}))
    result = _client().get("https://x/stable/x", {})
    assert not result.ok and "bad symbol" in (result.error or "")


def test_transient_failure_is_retried_then_reported(patch_session, monkeypatch):
    monkeypatch.setattr("quant_fund_agent.data.fmp_ingest.client.time.sleep", lambda s: None)
    session = patch_session(FakeSession({"/x": FakeResponse(500, None, text="boom")}))
    result = _client().get("https://x/stable/x", {})
    assert not result.ok and not result.restricted
    assert len(session.calls) == 3  # retries exhausted


# ── store ────────────────────────────────────────────────────────────────────

def test_coerce_frame_keeps_labels_as_text_and_numbers_as_floats():
    df = coerce_frame([
        {"date": "2020-01-01", "period": "Q1", "revenue": 10, "eps": None},
        {"date": "2020-04-01", "period": "FY", "revenue": "20.5", "eps": 1.5},
    ])
    assert str(df["revenue"].dtype).startswith("float")
    assert str(df["period"].dtype) in ("string", "str")  # pandas 2 vs 3 spelling
    assert df["revenue"].tolist() == [10.0, 20.5]


def test_write_merges_chunks_and_dedupes_by_date(tmp_path):
    archive = Archive(tmp_path)
    ep = ENDPOINTS["prices_adjusted"]
    archive.write(ep, [{"date": "2020-01-02", "adjClose": 1.0},
                       {"date": "2020-01-03", "adjClose": 2.0}], symbol="AAA")
    rows, first, last = archive.write(
        ep, [{"date": "2020-01-03", "adjClose": 9.9},   # later chunk wins
             {"date": "2020-01-06", "adjClose": 3.0}], symbol="AAA")
    assert (rows, first, last) == (3, "2020-01-02", "2020-01-06")
    frame = archive.read(ep, "AAA")
    assert frame.loc["2020-01-03", "adjClose"] == 9.9


def test_manifest_journal_survives_a_kill_and_compacts(tmp_path):
    archive = Archive(tmp_path)
    archive.record(ManifestEntry("prices_adjusted:AAA", "prices_adjusted", "AAA",
                                 status=STATUS_OK, rows=10))
    archive.record(ManifestEntry("ratios:quarter:AAA", "ratios", "AAA", "quarter",
                                 status=STATUS_RESTRICTED))
    # A fresh Archive object = a resumed process.
    resumed = Archive(tmp_path)
    entries = resumed.load_manifest()
    assert set(entries) == {"prices_adjusted:AAA", "ratios:quarter:AAA"}
    assert resumed.is_done("prices_adjusted:AAA")
    assert resumed.is_done("ratios:quarter:AAA"), "a plan gate is terminal for this plan"
    assert not resumed.is_done("prices_adjusted:BBB")

    resumed.compact_manifest()
    assert not (tmp_path / "manifest.jsonl").exists()
    assert Archive(tmp_path).load_manifest().keys() == entries.keys()


def test_errored_units_are_retried_unless_disabled(tmp_path):
    archive = Archive(tmp_path)
    archive.record(ManifestEntry("k", "prices_adjusted", "AAA", status="error"))
    Archive(tmp_path)
    archive.load_manifest()
    assert not archive.is_done("k", retry_errors=True)
    assert archive.is_done("k", retry_errors=False)


# ── endpoints registry ───────────────────────────────────────────────────────

def test_unknown_group_is_an_error_not_an_empty_download():
    with pytest.raises(ValueError, match="Unknown endpoint group"):
        endpoints_for_groups(["prices", "typo"])


def test_statement_endpoints_fan_out_over_both_fiscal_periods():
    assert work_units(ENDPOINTS["income_statement"]) == ["quarter", "annual"]
    assert work_units(ENDPOINTS["prices_adjusted"]) == [None]


# ── symbols ──────────────────────────────────────────────────────────────────

def test_candidates_try_the_literal_ticker_first_then_variants():
    assert candidate_symbols("BRK-B")[:2] == ["BRK-B", "BRK.B"]
    assert candidate_symbols("ENRNQ")[-1] == "ENRN"      # bankruptcy suffix last
    assert "FBIN" in candidate_symbols("FBHS", renames={"FBHS": "FBIN"})


def test_spell_coverage_scores_only_the_membership_window():
    spells = pd.DataFrame({
        "ticker": ["AAA"],
        "start_date": [pd.Timestamp("2020-01-01")],
        "end_date": [pd.Timestamp("2020-07-01")],
    })
    windows = spell_windows(spells, "AAA", pd.Timestamp("2004-01-01"),
                            pd.Timestamp("2026-01-01"))
    assert windows == [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"))]
    full = pd.bdate_range("2020-01-01", "2020-06-30")
    assert spell_coverage(full, windows) == pytest.approx(1.0)
    # Bars outside the spell must not count towards coverage.
    outside = pd.bdate_range("2021-01-01", "2021-12-31")
    assert spell_coverage(outside, windows) == 0.0
    assert spell_coverage(None, windows) == 0.0


def test_summarise_separates_resolved_from_covered():
    from quant_fund_agent.data.fmp_ingest.symbols import Resolution

    stats = summarise([
        Resolution("AAA", "AAA", "direct", spell_coverage=0.99),
        Resolution("BBB", "BBB", "variant", spell_coverage=0.40),
        Resolution("CCC", None, "unresolved"),
    ])
    assert stats["resolved"] == 2 and stats["coverage_ge_90pct"] == 1
    assert stats["unresolved"] == ["CCC"]


# ── constituent reconstruction ───────────────────────────────────────────────

_CHANGES = [
    {"date": "2010-06-01", "symbol": "CCC", "addedSecurity": "Gamma",
     "removedTicker": "DDD", "removedSecurity": "Delta", "reason": "mktcap"},
    {"date": "2012-01-03", "symbol": "EEE", "addedSecurity": "Eps",
     "removedTicker": "GGG", "removedSecurity": "Gam", "reason": "x"},
    {"date": "2013-01-03", "symbol": "FFF", "addedSecurity": "Phi",
     "removedTicker": "EEE", "removedSecurity": "Eps", "reason": "y"},
    {"date": "2014-01-03", "symbol": "EEE", "addedSecurity": "Eps",
     "removedTicker": "CCC", "removedSecurity": "Gamma", "reason": "z"},
]
_CURRENT = [{"symbol": s, "name": s, "cik": "", "dateFirstAdded": None}
            for s in ("AAA", "EEE", "FFF")]


def test_backward_walk_reproduces_membership_at_any_date():
    build = build_intervals("sp500", _CURRENT, _CHANGES, since="2004-01-01")
    on = lambda d: sorted(members_from_spells(build.spells, d))  # noqa: E731
    assert on("2005-01-01") == ["AAA", "DDD", "GGG"]
    assert on("2011-01-01") == ["AAA", "CCC", "GGG"]
    assert on("2012-06-01") == ["AAA", "CCC", "EEE"]
    assert on("2013-06-01") == ["AAA", "CCC", "FFF"]
    assert on("2016-01-01") == ["AAA", "EEE", "FFF"]


def test_a_name_that_left_and_rejoined_gets_two_spells():
    build = build_intervals("sp500", _CURRENT, _CHANGES, since="2004-01-01")
    eee = build.spells[build.spells["ticker"] == "EEE"].sort_values("start_date")
    assert len(eee) == 2
    assert str(eee.iloc[0]["end_date"].date()) == "2013-01-03"
    assert pd.isna(eee.iloc[1]["end_date"])


def test_pre_log_members_are_floored_below_the_window_not_collapsed():
    """A name removed by the very first logged event must keep a real spell."""
    build = build_intervals("sp500", _CURRENT, _CHANGES, since="2004-01-01")
    ddd = build.spells[build.spells["ticker"] == "DDD"].iloc[0]
    assert ddd["start_date"] < pd.Timestamp("2004-01-01")
    assert ddd["end_date"] == pd.Timestamp("2010-06-01")
    assert "DDD" in build.left_censored


def test_current_list_date_first_added_overrides_the_floor():
    current = [{"symbol": "AAA", "name": "Alpha", "cik": "1",
                "dateFirstAdded": "1998-01-05"}] + _CURRENT[1:]
    build = build_intervals("sp500", current, _CHANGES, since="2004-01-01")
    aaa = build.spells[build.spells["ticker"] == "AAA"].iloc[0]
    assert aaa["start_date"] == pd.Timestamp("1998-01-05")
    assert "AAA" not in build.left_censored


def test_a_log_that_starts_after_the_window_is_flagged():
    build = build_intervals("sp500", _CURRENT, _CHANGES, since="2004-01-01")
    assert build.notes and "left-censored" in build.notes[0]


def test_unparseable_change_log_raises_instead_of_writing_an_empty_table():
    with pytest.raises(ValueError, match="parsed to zero rows"):
        build_intervals("sp500", _CURRENT, [{"nonsense": 1}], since="2004-01-01")


def test_change_log_parses_prose_dates():
    rows = parse_changes([{"dateAdded": "October 1, 2024", "symbol": "DELL",
                           "removedTicker": "ETSY"}])
    assert rows.iloc[0]["date"] == pd.Timestamp("2024-10-01")


def test_walk_backward_never_stamps_the_pre_log_set_on_the_first_event_date():
    changes = parse_changes(_CHANGES)
    timeline = walk_backward({"AAA"}, changes, pre_start=pd.Timestamp("2004-01-01"))
    assert timeline[0][0] < timeline[1][0]
    assert len({d for d, _ in timeline}) == len(timeline)


# ── download planning + resume ───────────────────────────────────────────────

def _downloader(tmp_path, patch_session, session, *, rate=100000, **cfg) -> Downloader:
    config = DownloadConfig(start="2020-01-01", end="2021-12-31",
                            groups=("prices",), workers=1, **cfg)
    patch_session(session)
    return Downloader(FMPClient("test-key", per_minute=rate, retries=3),
                      Archive(tmp_path), config,
                      capabilities=Capabilities(price_full_range_ok=True))


def test_plan_multiplies_symbols_periods_and_chunks(tmp_path, patch_session):
    dl = _downloader(tmp_path, patch_session, FakeSession({}), rate=600)
    plan = dl.plan(["AAA", "BBB"])
    assert plan["symbols"] == 2
    assert plan["calls"] == plan["units"]         # one chunk each at full range
    assert plan["estimated_minutes"] > 0

    dl.config.window_years = 1                    # 2020 + 2021 → two chunks
    plan2 = dl.plan(["AAA", "BBB"])
    assert plan2["units"] == plan["units"], "chunking changes calls, never units"
    # Only the windowed endpoints chunk; dividends/splits stay one call each.
    assert plan2["per_endpoint"]["prices_adjusted"]["chunks"] == 2
    assert plan2["per_endpoint"]["dividends"]["chunks"] == 1
    assert plan2["calls"] > plan["calls"]


def test_chunking_covers_the_window_without_gaps_or_overlap(tmp_path, patch_session):
    dl = _downloader(tmp_path, patch_session, FakeSession({}), window_years=1)
    chunks = dl._chunks(ENDPOINTS["prices_adjusted"])
    assert chunks[0][0] == "2020-01-01" and chunks[-1][1] == "2021-12-31"
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(next_start) == pd.Timestamp(prev_end) + pd.Timedelta(days=1)


def test_resolution_falls_back_to_a_variant_and_is_recorded(tmp_path, patch_session):
    def route(n):
        # first candidate (BRK-B) empty, second (BRK.B) serves bars
        return FakeResponse(200, [] if n == 1 else
                            [{"date": "2020-01-02", "adjClose": 1.0}])

    dl = _downloader(tmp_path, patch_session,
                     FakeSession({"/historical-price-eod/dividend-adjusted": route}))
    report = dl.run(["BRK-B"])
    resolution = report.resolutions[0]
    assert resolution.resolved == "BRK.B" and resolution.method == "variant"
    assert "BRK-B>BRK.B" == resolution.attempts
    # The file is keyed by the MEMBERSHIP ticker, not the vendor symbol.
    assert dl.archive.read("prices_adjusted", "BRK-B") is not None


def test_an_unresolvable_name_is_reported_and_does_not_abort_the_run(tmp_path, patch_session):
    session = FakeSession({}, default=FakeResponse(200, []))
    dl = _downloader(tmp_path, patch_session, session)
    report = dl.run(["AAA", "ZZZ"])
    assert len(report.resolutions) == 2
    assert all(r.resolved is None for r in report.resolutions)

    # The miss is recorded, so a resumed run does not re-try every candidate for
    # a name FMP simply does not carry (~100 of those in a 2004+ pull).
    calls_after_first = len(session.calls)
    _downloader(tmp_path, patch_session, session).run(["AAA", "ZZZ"])
    assert len(session.calls) == calls_after_first


def test_a_resumed_run_skips_completed_units(tmp_path, patch_session):
    session = FakeSession({}, default=FakeResponse(
        200, [{"date": "2020-01-02", "adjClose": 1.0}]))
    dl = _downloader(tmp_path, patch_session, session)
    dl.run(["AAA"])
    first_pass = len(session.calls)
    assert first_pass > 0

    dl2 = _downloader(tmp_path, patch_session, session)
    report = dl2.run(["AAA"])
    assert len(session.calls) == first_pass, "a completed unit must not be refetched"
    assert report.units_skipped > 0


# ── capability probe ─────────────────────────────────────────────────────────

def test_probe_reads_the_limit_cap_out_of_the_restriction_text(patch_session, monkeypatch):
    monkeypatch.setattr("quant_fund_agent.data.fmp_ingest.client.time.sleep", lambda s: None)

    def statements(n):
        if n == 1:
            return FakeResponse(
                402, None,
                text="Premium Query Parameter: 'Special Parameters : The values for "
                     "'limit' must be between 0 and 5 based on your current subscription.")
        return FakeResponse(200, [{"date": "2020-03-31", "revenue": 1.0}])

    patch_session(FakeSession(
        {"/income-statement": statements},
        default=FakeResponse(200, [{"date": "2020-01-02", "adjClose": 1.0}]),
    ))
    caps = probe(_client(), start="2020-01-01", end="2020-12-31",
                 names=["income_statement", "prices_adjusted"])
    assert caps.limit_cap == 5
    assert caps.endpoints["income_statement"]["status"] == "ok"
    assert caps.limit_for(ENDPOINTS["income_statement"]) == 5
    assert caps.allows("income_statement")


def test_probe_flags_missing_delisted_access(patch_session):
    patch_session(FakeSession(
        {},
        default=FakeResponse(402, None,
                             text="Special Endpoint : This value set for 'symbol' is "
                                  "not available under your current subscription"),
    ))
    caps = probe(_client(), names=["prices_adjusted"])
    assert not caps.delisted_ok
    assert not caps.allows("prices_adjusted")


def test_capabilities_round_trip_through_json():
    caps = Capabilities(endpoints={"ratios": {"status": "restricted"}}, limit_cap=5,
                        delisted_ok=True, price_earliest="1990-01-02")
    restored = Capabilities.from_dict(json.loads(json.dumps(caps.to_dict())))
    assert restored.limit_cap == 5 and restored.delisted_ok
    assert not restored.allows("ratios") and restored.allows("prices_adjusted")
