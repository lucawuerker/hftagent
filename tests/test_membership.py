"""Point-in-time S&P 500 membership: query API, reconstruction, and artifact audit.

Two layers:
  * **unit** — the query API (``data/membership.py``) on a synthetic fixture and
    the build-script reconstruction primitives on tiny synthetic inputs; and
  * **audit** — invariants on the *shipped* ``sp500.csv`` (count band, no overlaps,
    survivorship spot-checks) so a future rebuild can't silently regress.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from quant_fund_agent.data import membership as M

ROOT = Path(__file__).resolve().parent.parent


def _build_module():
    """Import the standalone build script as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_sp500_membership", ROOT / "scripts" / "build_sp500_membership.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── fixture: a small deterministic membership table ──────────────────────────

@pytest.fixture()
def fixture_index(tmp_path, monkeypatch):
    """Write a tiny membership CSV and point the query API at it.

    AAA: always a member.  BBB: leaves 2015-06-01 then rejoins 2018-01-01 (two
    spells).  CCC: joins 2020-01-01.  ``end_date`` is exclusive.
    """
    rows = [
        ("AAA", "2010-01-01", ""),
        ("BBB", "2010-01-01", "2015-06-01"),
        ("BBB", "2018-01-01", ""),
        ("CCC", "2020-01-01", ""),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    df["name"] = ""; df["add_reason"] = ""; df["remove_reason"] = ""; df["source"] = "test"
    monkeypatch.setattr(M, "MEMBERSHIP_DIR", tmp_path)
    df.to_csv(tmp_path / "testidx.csv", index=False)
    M.load_membership.cache_clear()
    yield "testidx"
    M.load_membership.cache_clear()


# ── query API ────────────────────────────────────────────────────────────────

def test_members_as_of_interval_is_start_inclusive_end_exclusive(fixture_index):
    idx = fixture_index
    assert M.members_as_of("2012-01-01", idx) == ["AAA", "BBB"]
    assert M.members_as_of("2015-05-31", idx) == ["AAA", "BBB"]   # day before end
    assert M.members_as_of("2015-06-01", idx) == ["AAA"]          # end is exclusive
    assert M.members_as_of("2016-01-01", idx) == ["AAA"]          # BBB in its gap
    assert M.members_as_of("2019-01-01", idx) == ["AAA", "BBB"]   # BBB rejoined
    assert M.members_as_of("2021-01-01", idx) == ["AAA", "BBB", "CCC"]


def test_union_members_overlap(fixture_index):
    idx = fixture_index
    assert M.union_members("2010-01-01", "2030-01-01", idx) == ["AAA", "BBB", "CCC"]
    # During BBB's gap and before CCC joins, only AAA was ever a member.
    assert M.union_members("2016-01-01", "2017-01-01", idx) == ["AAA"]
    assert M.union_members(index=idx) == ["AAA", "BBB", "CCC"]    # all history


def test_membership_mask_shape_and_values(fixture_index):
    idx = fixture_index
    dates = pd.date_range("2014-01-01", "2021-01-01", freq="MS")
    mask = M.membership_mask(dates, ["AAA", "BBB", "CCC", "ZZZ"], idx)
    assert mask.shape == (len(dates), 4)
    assert mask["AAA"].all()                       # always a member
    assert not mask["ZZZ"].any()                   # unknown ticker -> never
    assert mask.loc["2014-01-01", "BBB"]           # in first spell
    assert not mask.loc["2016-01-01", "BBB"]       # in the gap
    assert mask.loc["2019-01-01", "BBB"]           # rejoined
    assert not mask.loc["2019-01-01", "CCC"]
    assert mask.loc["2020-06-01", "CCC"]


def test_apply_membership_mask_nans_non_members(fixture_index):
    idx = fixture_index
    dates = pd.date_range("2014-01-01", "2021-01-01", freq="MS")
    panel = {"close": pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB", "CCC"])}
    out = M.apply_membership_mask(panel, idx)["close"]
    assert out["AAA"].notna().all()
    assert pd.isna(out.loc["2016-01-01", "BBB"])   # masked off in the gap
    assert out.loc["2014-01-01", "BBB"] == 1.0
    assert pd.isna(out.loc["2014-01-01", "CCC"])   # before CCC joined


def test_apply_membership_mask_missing_artifact_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "MEMBERSHIP_DIR", tmp_path)
    M.load_membership.cache_clear()
    dates = pd.date_range("2014-01-01", periods=3, freq="MS")
    panel = {"close": pd.DataFrame(1.0, index=dates, columns=["AAA"])}
    out = M.apply_membership_mask(panel, "no_such_index")
    assert out["close"].equals(panel["close"])     # logged no-op, never crashes
    M.load_membership.cache_clear()


# ── build-script reconstruction primitives ───────────────────────────────────

def test_normalize_ticker_to_yfinance_convention():
    b = _build_module()
    assert b.normalize_ticker("brk.b") == "BRK-B"
    assert b.normalize_ticker(" aapl ") == "AAPL"
    assert b.normalize_ticker("BF.B") == "BF-B"


def test_intervals_from_components_runs_and_gaps():
    b = _build_module()
    comp = pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01"]),
        "set": [frozenset({"A", "B"}), frozenset({"A", "B"}),
                frozenset({"A"}), frozenset({"A", "B"})],
    })
    iv = b.intervals_from_components(comp).set_index("ticker")
    # A present through the final row -> open spell.
    assert pd.isna(iv.loc["A", "end_date"])
    # B: present rows 0-1 (end = first absent date), then rejoins row 3 (open).
    bspells = b.intervals_from_components(comp)
    bspells = bspells[bspells.ticker == "B"].sort_values("start_date")
    assert len(bspells) == 2
    assert bspells.iloc[0]["end_date"] == pd.Timestamp("2010-03-01")  # exclusive end
    assert pd.isna(bspells.iloc[1]["end_date"])


def test_apply_renames_coalesces_into_one_continuous_spell():
    b = _build_module()
    raw = pd.DataFrame({
        "ticker": ["FB", "META"],
        "start_date": pd.to_datetime(["2013-12-23", "2022-06-09"]),
        "end_date": [pd.Timestamp("2022-06-09"), pd.NaT],
    })
    out = b.apply_renames(raw, {"FB": "META"})
    assert list(out["ticker"]) == ["META"]
    assert out.iloc[0]["start_date"] == pd.Timestamp("2013-12-23")  # FB's original add
    assert pd.isna(out.iloc[0]["end_date"])                          # one open spell


def test_detect_renames_and_spinoff_classifier():
    b = _build_module()
    assert b._is_spinoff("General Electric Corp. spun off GE Vernova.")
    assert b._is_spinoff("...in a split-off exchange offer.")
    assert not b._is_spinoff("Market capitalization change.")
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2019-04-01"]),
        "add_ticker": ["NEW"], "add_name": ["Acme Corp"],
        "rm_ticker": ["OLD"], "rm_name": ["Acme Corp"],
    })
    assert b.detect_renames(changes) == {"OLD": "NEW"}  # same-day, same company


def test_wiki_backward_walk():
    b = _build_module()
    # Today's set is {B,C,D}; on 2020-01-01 D was added and A removed.
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "add_ticker": ["D"], "rm_ticker": ["A"],
    })
    wiki = b.WikiSnapshots({"B", "C", "D"}, changes)
    assert set(wiki.members_on("2020-06-01")) == {"B", "C", "D"}
    assert set(wiki.members_on("2019-06-01")) == {"A", "B", "C"}  # before the change


# ── integration wiring ───────────────────────────────────────────────────────

def test_resolve_universe_pit_returns_union_and_distinct_scope():
    from quant_fund_agent.config import DataSettings, config_fingerprint
    pit = DataSettings(provider="yfinance", membership="sp500",
                       start="2010-01-01", end="2026-06-01")
    static = DataSettings(provider="yfinance", universe_preset="sp100",
                          start="2010-01-01", end="2026-06-01")
    from quant_fund_agent.data.universe import resolve_universe
    union = resolve_universe(pit)
    assert len(union) > 700 and "AAPL" in union
    assert config_fingerprint(pit) != config_fingerprint(static)


# ── audit of the shipped sp500.csv ───────────────────────────────────────────

@pytest.fixture(scope="module")
def sp500():
    if not M.membership_path("sp500").exists():
        pytest.skip("sp500.csv not built (run scripts/build_sp500_membership.py)")
    return M.load_membership("sp500")


def test_artifact_has_survivorship_free_universe(sp500):
    assert sp500["ticker"].nunique() > 700           # far more than today's ~503
    assert int(sp500["end_date"].isna().sum()) > 490  # ~503 still-active


def test_artifact_no_overlapping_spells_and_valid_intervals(sp500):
    bad = sp500[sp500["end_date"].notna() & (sp500["end_date"] <= sp500["start_date"])]
    assert bad.empty, f"{len(bad)} spells with end_date <= start_date"
    for ticker, sub in sp500.groupby("ticker"):
        sub = sub.sort_values("start_date")
        ends = sub["end_date"].fillna(pd.Timestamp("2999-12-31")).tolist()
        starts = sub["start_date"].tolist()
        for i in range(1, len(starts)):
            assert starts[i] >= ends[i - 1], f"overlapping spells for {ticker}"


def test_artifact_monthly_count_in_band():
    months = pd.date_range("2010-01-31", "2026-05-31", freq="ME")
    counts = [len(M.members_as_of(m, "sp500")) for m in months]
    assert min(counts) >= 475, f"min monthly count {min(counts)} too low"
    assert max(counts) <= 515, f"max monthly count {max(counts)} too high"


def test_artifact_survivorship_spot_checks():
    # Names that left the index must still be present while they were members,
    # and absent afterwards — the whole point of the dataset.
    assert "TSLA" not in M.members_as_of("2015-06-01", "sp500")  # added 2020-12-21
    assert "TSLA" in M.members_as_of("2022-01-03", "sp500")
    assert "AAPL" in M.members_as_of("2010-06-01", "sp500")      # member throughout
    assert "AAPL" in M.members_as_of("2026-01-02", "sp500")
    assert "ATVI" in M.members_as_of("2016-06-01", "sp500")      # before MSFT bought it
    assert "ATVI" not in M.members_as_of("2024-06-03", "sp500")  # removed 2023-10-18
