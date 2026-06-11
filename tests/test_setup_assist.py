"""Offline tests for the onboarding ``--assist`` LLM layer (Phase 5).

No network: the validation/parsing logic is exercised directly, and the one
``propose_config`` path is tested by monkeypatching the LLM factory so we never
hit a real API. The contract under test is that the assist layer can only ever
*narrow* a proposal to legal values and never crashes the wizard.
"""

from __future__ import annotations

import argparse

import pytest

from quant_fund_agent import setup as wizard
from quant_fund_agent import setup_assist as assist

AVAILABLE = ["lobster", "yfinance", "fmp"]
PRESETS = ["demo", "sp100"]


# ── validate_proposal: only legal, in-bounds values survive ──────────────────

def test_validate_keeps_known_fields():
    raw = {
        "provider": "yfinance",
        "asset_class": "equity",
        "frequency": "1d",
        "start": "2023-01-01",
        "end": "2024-06-01",
        "preset": "sp100",
        "n_tickers": 25,
    }
    out = assist.validate_proposal(raw, available=AVAILABLE, presets=PRESETS)
    assert out == raw


def test_validate_drops_unknown_provider_and_preset():
    raw = {"provider": "bloomberg", "preset": "nasdaq100", "asset_class": "equity"}
    out = assist.validate_proposal(raw, available=AVAILABLE, presets=PRESETS)
    assert "provider" not in out          # not a usable provider → dropped
    assert "preset" not in out            # not an existing preset → dropped
    assert out == {"asset_class": "equity"}


def test_validate_rejects_incoherent_dates():
    # end before start → both dropped (we only accept a coherent pair)
    raw = {"start": "2024-01-01", "end": "2023-01-01"}
    assert assist.validate_proposal(raw, available=AVAILABLE, presets=PRESETS) == {}
    # unparseable date → dropped
    raw = {"start": "last tuesday", "end": "2024-01-01"}
    assert assist.validate_proposal(raw, available=AVAILABLE, presets=PRESETS) == {}


def test_validate_tickers_uppercased_capped_and_exclude_preset():
    raw = {"preset": "sp100", "tickers": [f"t{i}" for i in range(20)]}
    out = assist.validate_proposal(raw, available=AVAILABLE, presets=PRESETS)
    assert "preset" not in out                       # tickers win, preset dropped
    assert out["tickers"][0] == "T0" and len(out["tickers"]) == 15  # upper + cap


def test_validate_handles_garbage_input():
    assert assist.validate_proposal("not a dict", available=AVAILABLE, presets=PRESETS) == {}
    assert assist.validate_proposal({}, available=AVAILABLE, presets=PRESETS) == {}
    # n_tickers as a numeric string is coerced; junk is dropped
    assert assist.validate_proposal({"n_tickers": "5"}, available=AVAILABLE, presets=PRESETS) == {"n_tickers": 5}
    assert assist.validate_proposal({"n_tickers": -3}, available=AVAILABLE, presets=PRESETS) == {}


# ── _extract_json: tolerant of fences / surrounding prose ────────────────────

def test_extract_json_from_fenced_and_noisy_text():
    fenced = 'Here you go:\n```json\n{"provider": "fmp"}\n```\nHope that helps!'
    assert assist._extract_json(fenced) == '{"provider": "fmp"}'
    assert assist._extract_json("no json here") is None
    assert assist._extract_json("") is None


# ── propose_config: graceful failure + end-to-end with a stubbed LLM ──────────

def test_propose_config_returns_empty_on_llm_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no API key")

    monkeypatch.setattr("quant_fund_agent.llm.make_chat_llm", boom)
    out = assist.propose_config("anything", available=AVAILABLE, presets=PRESETS)
    assert out == {}                      # never raises; wizard falls back


def test_propose_config_with_stubbed_llm(monkeypatch):
    class FakeResp:
        content = '{"provider": "yfinance", "preset": "demo", "n_tickers": 5}'

    class FakeLLM:
        def invoke(self, messages):
            return FakeResp()

    monkeypatch.setattr("quant_fund_agent.llm.make_chat_llm", lambda **k: FakeLLM())
    out = assist.propose_config("tech, demo universe", available=AVAILABLE, presets=PRESETS)
    assert out == {"provider": "yfinance", "preset": "demo", "n_tickers": 5}


def test_propose_config_empty_description_skips_llm(monkeypatch):
    monkeypatch.setattr(
        "quant_fund_agent.llm.make_chat_llm",
        lambda **k: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    assert assist.propose_config("   ", available=AVAILABLE, presets=PRESETS) == {}


# ── build_config: proposal supplies confirmable defaults, CLI flags still win ─

def _args(**over):
    base = dict(provider=None, asset_class=None, freq=None, start=None, end=None,
                preset=None, tickers=None, n_tickers=None, data_dir=None, cache_dir=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_build_config_uses_proposal_as_default(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "x")  # make fmp "usable" so the proposal sticks
    proposal = {"provider": "yfinance", "start": "2022-01-01", "end": "2023-01-01",
                "tickers": ["AAPL", "MSFT"], "n_tickers": 2}
    cfg = wizard.build_config(_args(), interactive=False, proposal=proposal)["data"]
    assert cfg["provider"] == "yfinance"
    assert cfg["start"] == "2022-01-01" and cfg["end"] == "2023-01-01"
    assert cfg["tickers"] == ["AAPL", "MSFT"] and cfg["n_tickers"] == 2
    assert "universe_preset" not in cfg


def test_build_config_cli_flag_beats_proposal():
    proposal = {"provider": "yfinance", "preset": "sp100"}
    cfg = wizard.build_config(
        _args(provider="yfinance", preset="demo"), interactive=False, proposal=proposal)["data"]
    assert cfg["universe_preset"] == "demo"   # explicit --preset wins over proposal
