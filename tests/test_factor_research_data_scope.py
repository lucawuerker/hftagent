"""Data-scope gating for the Factor Researcher.

The researcher must only invent factors the configured feed can serve.  These
tests cover the three seams that enforce that:

  * the LOBSTER order-book level → field partition (``tiers``),
  * the run's usable-field resolution (provider caps + fundamentals opt-out),
  * the data-context prose built for the brainstorm/codegen prompts, and
  * the onboarding wizard writing ``lobster_level`` / ``fundamentals`` to config.
"""

from __future__ import annotations

import argparse
import re

from quant_fund_agent import setup as wizard
from quant_fund_agent.agents.factor_research.prompts import (
    DATA_CONTEXT,
    build_data_context,
)
from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import usable_fields
from quant_fund_agent.data.fields import FUNDAMENTAL_FIELDS, NON_OHLCV_FIELDS
from quant_fund_agent.data.providers.lobster import LobsterProvider
from quant_fund_agent.data.tiers import (
    LOBSTER_L3_MICRO_FIELDS,
    TIERS,
    lobster_fields_for_level,
)

# Fields the user fixed as the L2/L3 boundary ("book + volume").
_L2_PRESENT = {"open", "high", "low", "close", "volume", "spread", "depth",
               "lobImb", "effLobImb", "bidDepth1", "askPrice1"}
_L3_ONLY = {"trade", "orderFlow", "hidden", "auction", "effSpread", "trdLiq",
            "ofLiq", "nbEvents", "nbHidden", "nbTrades"}


# ── tier partition ───────────────────────────────────────────────────────────

def test_lobster_level2_is_book_plus_volume():
    l2 = lobster_fields_for_level(2)
    assert _L2_PRESENT <= l2
    assert not (_L3_ONLY & l2)


def test_lobster_level3_adds_message_stream_fields():
    l3 = lobster_fields_for_level(3)
    assert _L3_ONLY <= l3
    assert LOBSTER_L3_MICRO_FIELDS == frozenset(_L3_ONLY)
    # Level 3 is the historical full field set.
    assert l3 == frozenset(TIERS["standard"] | TIERS["microstructure"])


def test_lobster_level_partition_is_disjoint_and_complete():
    l2, l3 = lobster_fields_for_level(2), lobster_fields_for_level(3)
    assert l2 < l3                                  # strict subset
    assert (l3 - l2) == LOBSTER_L3_MICRO_FIELDS


# ── provider honors the level ────────────────────────────────────────────────

def _lobster(level, data_dir="does_not_exist_dir"):
    # A nonexistent data_dir keeps the header sniff inert so these tests see
    # the static tier advertisement regardless of what CSVs the repo carries.
    return LobsterProvider(Settings(data=DataSettings(
        provider="lobster", lobster_level=level, data_dir=data_dir)))


def test_provider_available_fields_track_level():
    assert _lobster(2).available_fields() == lobster_fields_for_level(2)
    assert _lobster(3).available_fields() == lobster_fields_for_level(3)


def test_provider_defaults_to_full_level3():
    # A config with no lobster_level (legacy) behaves as before — full feed.
    p = LobsterProvider(Settings(data=DataSettings(
        provider="lobster", data_dir="does_not_exist_dir")))
    assert p.available_fields() == lobster_fields_for_level(3)


def test_provider_sniffs_on_disk_book_depth(tmp_path):
    # A feed exporting only 2 book levels must not advertise levels 3-10:
    # factors reading a phantom per-level column would be all-NaN.
    d = tmp_path / "XYZ"
    d.mkdir()
    header = ("date,time,stock,mid,spread,depth,lobImb,"
              "askPrice1,askDepth1,bidPrice1,bidDepth1,"
              "askPrice2,askDepth2,bidPrice2,bidDepth2")
    (d / "bin202401.csv").write_text(header + "\n")
    fields = _lobster(3, data_dir=str(tmp_path)).available_fields()
    assert "askPrice2" in fields and "bidDepth2" in fields
    assert "askPrice3" not in fields and "bidDepth10" not in fields
    # non-per-level advertisement is untouched
    static = lobster_fields_for_level(3)
    non_level = {f for f in static if not f.startswith(("ask", "bid"))}
    assert non_level <= fields


# ── usable_fields: provider caps + fundamentals opt-out ──────────────────────

def test_usable_fields_lobster_level2_excludes_message_stream():
    uf = usable_fields(Settings(data=DataSettings(provider="lobster", lobster_level=2)))
    assert "volume" in uf and "spread" in uf
    assert not (_L3_ONLY & uf)


def test_usable_fields_drops_fundamentals_when_off():
    on = usable_fields(Settings(data=DataSettings(
        provider="fmp", asset_class="equity", fundamentals=True)))
    off = usable_fields(Settings(data=DataSettings(
        provider="fmp", asset_class="equity", fundamentals=False)))
    assert "peRatio" in on
    assert not (off & NON_OHLCV_FIELDS)


def test_usable_fields_non_equity_has_no_fundamentals():
    uf = usable_fields(Settings(data=DataSettings(
        provider="fmp", asset_class="crypto", fundamentals=True)))
    assert not (uf & NON_OHLCV_FIELDS)


# ── data-context prose only lists allowed fields ─────────────────────────────

def _has_entry(ctx: str, name: str) -> bool:
    """True iff ``name`` is offered as its own field entry (not just prose)."""
    return re.search(rf"(?m)^    {re.escape(name)}\b\s*:", ctx) is not None


def test_data_context_hides_level3_fields_at_level2():
    ctx = build_data_context(sorted(lobster_fields_for_level(2)))
    for name in _L3_ONLY:
        assert not _has_entry(ctx, name), f"{name} leaked into the L2 data context"
    for name in ("volume", "spread", "depth", "lobImb", "close"):
        assert _has_entry(ctx, name)
    assert "Per-level order-book" in ctx       # book levels advertised
    assert "Fundamental" not in ctx            # no fundamentals in scope


def test_data_context_shows_level3_fields_at_level3():
    ctx = build_data_context(sorted(lobster_fields_for_level(3)))
    for name in _L3_ONLY:
        assert _has_entry(ctx, name)


def test_data_context_fundamentals_only_uses_generic_intro():
    allowed = {"open", "high", "low", "close", "volume"} | set(FUNDAMENTAL_FIELDS)
    ctx = build_data_context(sorted(allowed))
    assert _has_entry(ctx, "peRatio")
    assert "Fundamental" in ctx and "LOOK-AHEAD" in ctx
    assert "Microstructure" not in ctx          # no micro fields → section dropped
    assert "configured market-data feed" in ctx  # non-LOBSTER intro


def test_data_context_none_is_full_and_ungated():
    assert DATA_CONTEXT == build_data_context(None)
    assert _has_entry(DATA_CONTEXT, "hidden") and _has_entry(DATA_CONTEXT, "peRatio")


# ── onboarding wizard writes the new keys ────────────────────────────────────

def _args(**over):
    base = dict(provider=None, asset_class=None, freq=None, start=None, end=None,
                preset=None, tickers=None, n_tickers=None, data_dir=None,
                cache_dir=None, lobster_level=None, fundamentals=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_build_config_lobster_level_flag():
    cfg = wizard.build_config(
        _args(provider="lobster", lobster_level=2), interactive=False)["data"]
    assert cfg["lobster_level"] == 2


def test_build_config_lobster_default_level3():
    cfg = wizard.build_config(_args(provider="lobster"), interactive=False)["data"]
    assert cfg["lobster_level"] == 3


def test_build_config_fundamentals_flag_for_fmp(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")  # make fmp selectable
    cfg = wizard.build_config(
        _args(provider="fmp", asset_class="equity", fundamentals="no",
              start="2023-01-01", end="2024-01-01"),
        interactive=False)["data"]
    assert cfg["fundamentals"] is False


def test_build_config_no_fundamentals_key_for_yfinance():
    # yfinance serves no fundamentals → the wizard doesn't write the key.
    cfg = wizard.build_config(
        _args(provider="yfinance", asset_class="equity",
              start="2023-01-01", end="2024-01-01"),
        interactive=False)["data"]
    assert "fundamentals" not in cfg
