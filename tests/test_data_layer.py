"""Tests for the pluggable data layer (Phase 0/2).

Covers:
- byte-identical parity between the routed ``data.load_panel`` and the legacy
  LOBSTER loader (the Phase 0 behaviour-preservation guarantee);
- settings precedence (env overrides config defaults);
- capability-tier helpers (gating + required-tier labelling).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_lobster_dir(root: Path, symbols=("AAA", "BBB"), n: int = 24) -> Path:
    """Write a tiny LOBSTER-schema CSV tree under ``root``."""
    rng = np.random.RandomState(0)
    for sym in symbols:
        d = root / sym
        d.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "date": ["2020-01-02"] * n,
                "time": [f"09:{30 + i // 60:02d}:{i % 60:02d}" for i in range(n)],
                "stock": [sym] * n,
                "trade": np.arange(n, dtype=float),
                "mid": 100 + np.arange(n, dtype=float) * 0.1,
                "midEnd": 100 + np.arange(n, dtype=float) * 0.1 + 0.05,
                "effSpread": np.full(n, 0.02),
                "orderFlow": rng.randn(n),
                "lobImb": rng.randn(n),
            }
        )
        df.to_csv(d / "data.csv", index=False)
    return root


def test_routed_load_panel_matches_legacy(tmp_path):
    from quant_fund_agent.backtesting.data_loader import load_panel as legacy
    from quant_fund_agent.data import load_panel as routed
    from quant_fund_agent.data.panel import SYNTH_DEPS

    root = _write_lobster_dir(tmp_path / "ticker_data")
    a = legacy(str(root))
    b = routed(str(root))
    # routed adds synthesised derived fields on top of the provider's raw fields;
    # every raw field must still be byte-identical to the legacy loader.
    assert set(b) == set(a) | set(SYNTH_DEPS)
    for field in a:
        pd.testing.assert_frame_equal(a[field], b[field])
    pd.testing.assert_frame_equal(b["vwap"], (a["high"] + a["low"] + a["close"]) / 3.0)
    pd.testing.assert_frame_equal(b["returns"], a["close"].pct_change())


def test_targeted_load_synthesises_and_trims(tmp_path):
    """Requesting vwap/returns loads their OHLC deps, derives them, trims output."""
    from quant_fund_agent.data import load_panel as routed

    root = _write_lobster_dir(tmp_path / "ticker_data")
    panel = routed(str(root), fields=["vwap", "returns", "close"])
    assert set(panel) == {"vwap", "returns", "close"}  # high/low loaded as deps, trimmed out
    pd.testing.assert_frame_equal(panel["returns"], panel["close"].pct_change())


def test_settings_env_override(monkeypatch):
    from quant_fund_agent.config import get_settings

    monkeypatch.setenv("DATA_DIR", "/some/where")
    monkeypatch.setenv("ARCHITECT_N_TICKERS", "7")
    s = get_settings()
    assert s.data.provider == "lobster"
    assert s.data.data_dir == "/some/where"
    assert s.data.n_tickers == 7


def test_with_data_overrides_keeps_none_untouched():
    from quant_fund_agent.config import Settings

    s = Settings()
    s.data.n_tickers = 5
    s2 = s.with_data_overrides(data_dir="X", n_tickers=None)
    assert s2.data.data_dir == "X"
    assert s2.data.n_tickers == 5  # None override does not clobber


def test_lobster_provider_capabilities():
    from quant_fund_agent.config import Settings
    from quant_fund_agent.data.providers.lobster import LobsterProvider

    fields = LobsterProvider(Settings()).available_fields()
    assert {"close", "volume", "effSpread", "orderFlow"} <= fields
    assert "sector" not in fields  # LOBSTER has no fundamental classification data


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (["close", "high", "low"], "standard"),
        (["close", "vwap"], "standard"),
        (["close", "cap"], "fundamental"),
        (["orderFlow", "close"], "microstructure"),
    ],
)
def test_required_tier(inputs, expected):
    from quant_fund_agent.data.tiers import required_tier

    assert required_tier(inputs) == expected


def test_is_compatible_synthesizes_vwap_and_returns():
    from quant_fund_agent.data.tiers import TIERS, is_compatible

    std = TIERS["standard"]
    assert is_compatible(["close", "vwap", "returns"], std)
    assert is_compatible(["open", "high", "low", "close", "volume"], std)
    assert not is_compatible(["orderFlow"], std)
    assert not is_compatible(["sector"], std)


# ── frequency-aware annualisation (Phase 1) ─────────────────────────────────

def test_frequency_inference_reproduces_lobster_defaults():
    """10-sec bars must infer the exact legacy 2340/day, 589680/yr values."""
    from quant_fund_agent.data.frequency import (
        DEFAULT_BARS_PER_DAY,
        DEFAULT_BARS_PER_YEAR,
        bars_per_day_from_index,
        periods_per_year_from_index,
    )

    idx = pd.date_range("2020-01-01", periods=5000, freq="10s")
    assert bars_per_day_from_index(idx) == DEFAULT_BARS_PER_DAY == 2340
    assert periods_per_year_from_index(idx) == DEFAULT_BARS_PER_YEAR


def test_frequency_inference_daily_and_minute():
    from quant_fund_agent.data.frequency import (
        bars_per_day_from_index,
        periods_per_year_from_index,
    )

    daily = pd.bdate_range("2020-01-01", periods=300)
    assert bars_per_day_from_index(daily) == 1
    assert periods_per_year_from_index(daily) == 252
    assert periods_per_year_from_index(daily, "crypto") == 365

    minute = pd.date_range("2020-01-01 09:30", periods=2000, freq="1min")
    assert bars_per_day_from_index(minute) == 390


def test_frequency_inference_short_series_falls_back():
    from quant_fund_agent.data.frequency import (
        DEFAULT_BARS_PER_DAY,
        bars_per_day_from_index,
    )

    assert bars_per_day_from_index(pd.DatetimeIndex(["2020-01-01"])) == DEFAULT_BARS_PER_DAY


# ── capability gating (Phase 2) ─────────────────────────────────────────────

def test_compatible_factors_filters_by_provider_fields():
    from quant_fund_agent.data.tiers import TIERS, compatible_factors

    records = [
        {"id": "f_ohlcv", "required_inputs": ["close", "high", "low", "volume"]},
        {"id": "f_micro", "required_inputs": ["orderFlow", "volume"]},
        {"id": "f_vwap", "required_inputs": ["vwap", "close"]},  # synthesised → ok
    ]
    std = TIERS["standard"]
    keep = {r["id"] for r in compatible_factors(records, std)}
    assert keep == {"f_ohlcv", "f_vwap"}  # micro gated out on a standard provider

    full = TIERS["standard"] | TIERS["microstructure"]
    keep_full = {r["id"] for r in compatible_factors(records, full)}
    assert keep_full == {"f_ohlcv", "f_micro", "f_vwap"}  # LOBSTER-like: nothing gated


def _write_factor_db(path, records):
    import json
    path.write_text(json.dumps({"factors": records}))


def test_catalog_gating_lobster_noop_standard_gates(tmp_path, monkeypatch):
    """Catalog: no-op on LOBSTER fields; microstructure gated on standard fields."""
    from quant_fund_agent.data.tiers import TIERS
    from quant_fund_agent.mcp import catalog_service

    db = tmp_path / "factor_db.json"
    _write_factor_db(db, [
        {"id": "a", "name": "A", "required_inputs": ["close", "volume"], "required_tier": "standard"},
        {"id": "b", "name": "B", "required_inputs": ["orderFlow"], "required_tier": "microstructure"},
    ])
    monkeypatch.setenv("FACTOR_DB_PATH", str(db))

    # LOBSTER-like provider (standard ∪ microstructure): both visible — no-op.
    monkeypatch.setattr(catalog_service, "_provider_available_fields",
                        lambda: TIERS["standard"] | TIERS["microstructure"])
    ids = {c["factor_id"] for c in catalog_service.load_factor_catalog()}
    assert ids == {"a", "b"}

    # Standard provider: microstructure factor gated out.
    monkeypatch.setattr(catalog_service, "_provider_available_fields",
                        lambda: TIERS["standard"])
    cat = catalog_service.load_factor_catalog()
    assert {c["factor_id"] for c in cat} == {"a"}
    assert cat[0]["required_tier"] == "standard"  # tier surfaced to the Selector

    # Escape hatch restores the full catalog even on a standard provider.
    monkeypatch.setenv("QF_FACTOR_GATING", "0")
    assert {c["factor_id"] for c in catalog_service.load_factor_catalog()} == {"a", "b"}


def test_resolve_required_inputs_fallback_to_registry(monkeypatch):
    """A record without stored inputs resolves from the live factor class."""
    from quant_fund_agent.data.tiers import resolve_required_inputs
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.factors.registry import get_all_factor_classes

    discover_factors()
    fid, cls = next(iter(get_all_factor_classes().items()))
    expected = list(getattr(cls, "inputs", ["close"]))
    assert resolve_required_inputs({"id": fid}) == expected
