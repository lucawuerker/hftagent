"""Phase 3 verification — yfinance provider chain, fully OFFLINE.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase3_offline.py

Injects a synthetic fetch into YFinanceProvider so NO network is needed, then
proves: universe resolution, parquet cache round-trip (2nd call = cache hit),
panel assembly, vwap/returns synthesis, daily→252/yr annualisation, and that
factor gating drops the 99 microstructure factors (98 visible) for this standard
OHLCV provider.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.data import load_panel
from quant_fund_agent.data.frequency import periods_per_year_from_index
from quant_fund_agent.data.providers.yfinance import YFinanceProvider
from quant_fund_agent.data.universe import available_presets, resolve_universe

FETCH_CALLS = []


def _synthetic(symbol, idx):
    n = len(idx)
    base = 100 + np.arange(n, dtype=float) + hash(symbol) % 7
    return pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1,
         "close": base + 0.5, "volume": np.full(n, 1e6)}, index=idx)


def _fake_fetch(self, symbols):
    # Span exactly the requested window so the cache fully covers it.
    FETCH_CALLS.append(list(symbols))
    idx = pd.bdate_range(self.data.start, self.data.end)
    return {s: _synthetic(s, idx) for s in symbols}


def main():
    YFinanceProvider._fetch = _fake_fetch  # inject — no network

    print("===== 1. universe resolution =====")
    print("available presets:", available_presets())
    print("demo preset      :", resolve_universe(DataSettings(universe_preset="demo")))
    print("custom + cap=2   :", resolve_universe(
        DataSettings(tickers=["aapl", "msft", "nvda"], n_tickers=2)))

    tmp = tempfile.mkdtemp()
    settings = Settings(data=DataSettings(
        provider="yfinance", tickers=["AAA", "BBB", "CCC"],
        start="2023-01-02", end="2023-06-30", frequency="1d", cache_dir=tmp))

    print("\n===== 2. panel load via yfinance provider (synthetic fetch) =====")
    panel = load_panel(settings=settings)
    print("fields:", sorted(panel))
    print("tickers:", list(panel["close"].columns), "| close shape:", panel["close"].shape)
    print("fetch calls so far:", FETCH_CALLS)

    print("\n===== 3. parquet cache round-trip (2nd load must NOT fetch) =====")
    import glob
    files = glob.glob(os.path.join(tmp, "**", "*.parquet"), recursive=True)
    print("cache files written:", [os.path.relpath(f, tmp) for f in files])
    n_before = len(FETCH_CALLS)
    _ = load_panel(settings=settings)
    print(f"fetch calls before 2nd load: {n_before}, after: {len(FETCH_CALLS)} "
          f"(cache hit: {len(FETCH_CALLS) == n_before})")

    print("\n===== 4. derived-field synthesis =====")
    exp_vwap = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    print("vwap == (H+L+C)/3:", bool((panel["vwap"] - exp_vwap).abs().max().max() < 1e-9))
    print("returns == close.pct_change():",
          bool((panel["returns"].fillna(0) - panel["close"].pct_change().fillna(0)).abs().max().max() < 1e-12))

    print("\n===== 5. frequency-aware annualisation (daily) =====")
    ppy = periods_per_year_from_index(panel["close"].index)
    print("periods/year:", ppy, "(expect 252)")

    print("\n===== 6. factor gating for a STANDARD provider =====")
    os.environ["QF_DATA_PROVIDER"] = "yfinance"
    from quant_fund_agent.mcp import catalog_service
    cat = catalog_service.load_factor_catalog()
    tiers = {}
    for c in cat:
        tiers[c["required_tier"]] = tiers.get(c["required_tier"], 0) + 1
    print(f"visible factors: {len(cat)} (by tier: {tiers})")
    del os.environ["QF_DATA_PROVIDER"]

    print("\n===== assertions =====")
    assert {"vwap", "returns"} <= set(panel)
    assert len(FETCH_CALLS) == n_before, "2nd load must hit cache"
    assert ppy == 252
    assert len(cat) == 98, f"expected 98 standard-tier factors, got {len(cat)}"
    assert all(c["required_tier"] == "standard" for c in cat)
    print("ALL PHASE-3 OFFLINE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
