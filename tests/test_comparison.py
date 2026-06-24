"""Offline tests for the research-LLM comparison harness (no LLM, no agents).

Fabricates two tiny preruns from existing seed-factor ids (so the code is already
registered and computable on the current panel), then exercises the IC + brute-force
tracks and the report/figure/notebook writers end to end.  The downstream-agent
track is intentionally not exercised here (it needs an LLM); its plumbing is covered
by the module's importability + the orchestrator's wiring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.factors import discover_factors, preruns
from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdeaCategory

TICKER_DATA = Path("ticker_data")
N_TICKERS = 6


def _seed_ids_with_ic(n: int) -> list[str]:
    db = FactorDatabase()
    db.load_from_json(preruns.BASE_FACTOR_DB)
    ids = [f.id for f in db.list_factors(source=FactorSource.SEED)
           if f.backtest_metrics and f.backtest_metrics.information_coefficient is not None]
    return ids[:n]


def _make_prerun(root: Path, name: str, ids: list[str], horizon: int = 6) -> None:
    db = FactorDatabase()
    for fid in ids:
        db.add_factor(FactorRecord(
            id=fid, name=fid, class_name="",
            category=TradingIdeaCategory.MOMENTUM,
            source=FactorSource.RESEARCHER,
            research_session_id=f"prerun:{name}",
            prediction_horizon=horizon,
            metadata={"llm_model": "test-model"},
        ))
    p = root / name / "factor_db.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    db.save_to_json(p)


def _lobster_tickers(n: int) -> list[str]:
    return sorted(d.name for d in TICKER_DATA.iterdir() if d.is_dir())[:n]


def _synthetic_ohlcv(symbol: str, n: int = 8) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    base = 10.0 + (sum(map(ord, symbol)) % 7)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": [base + i * 0.1 for i in range(n)],
            "volume": 1000,
        },
        index=idx,
    )


@pytest.fixture
def tiny_preruns(tmp_path, monkeypatch):
    """Two fabricated preruns under a temp prerun root."""
    discover_factors()
    root = tmp_path / "preruns"
    monkeypatch.setattr(preruns, "PRERUNS_ROOT", root)
    monkeypatch.setattr(preruns, "PAPER_READ_LOG_ROOT", tmp_path / "readlogs")
    seeds = _seed_ids_with_ic(6)
    if len(seeds) < 4:
        pytest.skip("not enough seed factors with recorded IC in the base DB")
    _make_prerun(root, "modelA", seeds[:3], horizon=6)
    _make_prerun(root, "modelB", seeds[3:6], horizon=60)  # heterogeneous horizons
    return ["modelA", "modelB"]


def test_usable_factor_ids_drops_unregistered(tiny_preruns):
    """A non-existent factor id is dropped (not crashed) with a clear reason."""
    from quant_fund_agent.comparison import factors as cf
    from quant_fund_agent.modeling import service

    service._PANEL_CACHE = None
    service._SIGNAL_CACHE = {}
    if not TICKER_DATA.is_dir():
        pytest.skip("ticker_data not present")
    panel = cf.load_panel_cached(
        "ticker_data", N_TICKERS,
        tickers=_lobster_tickers(N_TICKERS),
        data_overrides={"provider": "lobster"},
    )

    real = cf.prerun_factor_ids("modelA")
    usable, dropped = cf.usable_factor_ids(real + ["__not_a_factor__"], panel)
    assert "__not_a_factor__" in dropped
    assert "no factor class registered" in dropped["__not_a_factor__"]
    assert set(usable) <= set(real)


def test_offline_comparison_end_to_end(tiny_preruns, tmp_path):
    """IC + brute-force tracks + tables/figures/report/notebook, global DB untouched."""
    if not TICKER_DATA.is_dir():
        pytest.skip("ticker_data not present")

    from quant_fund_agent.comparison import bruteforce, factors as cf, ic, report
    from quant_fund_agent.comparison.config import ComparisonConfig
    from quant_fund_agent.modeling import service

    service._PANEL_CACHE = None
    service._SIGNAL_CACHE = {}

    global_db = Path("data/factors/factor_db.json")
    before = hashlib.md5(global_db.read_bytes()).hexdigest() if global_db.exists() else None

    cfg = ComparisonConfig(
        preruns=tiny_preruns, models=["ridge", "lasso"], include_ensemble=True,
        run_downstream=False, target_horizon=6, ic_horizons=(1, 6, 60),
        n_tickers=N_TICKERS, output_root=str(tmp_path / "comparisons"),
        comparison_id="utest", data_provider="lobster",
    )
    panel = cf.load_panel_cached(
        cfg.data_dir, cfg.n_tickers,
        tickers=_lobster_tickers(N_TICKERS),
        data_overrides=cfg.data_overrides(),
    )
    names_map = cf.factor_names(tiny_preruns)

    results: dict = {"usability": {}, "prerun_models": {}}
    usable: dict[str, list[str]] = {}
    for p in tiny_preruns:
        ids = cf.prerun_factor_ids(p)
        ok, dropped = cf.usable_factor_ids(ids, panel)
        usable[p] = ok
        results["usability"][p] = {"n_total": len(ids), "n_usable": len(ok),
                                   "n_dropped": len(dropped), "dropped": dropped}
        results["prerun_models"][p] = "test-model"
    assert any(usable.values()), "expected at least one usable factor"

    horizons_map = cf.prediction_horizons(tiny_preruns)
    assert set(horizons_map.values()) == {6, 60}  # heterogeneous per prerun

    ic_rows: list = []
    for p in tiny_preruns:
        ic_rows += ic.evaluate_prerun_ic(p, usable[p], panel, (1, 6, 60), names_map,
                                         horizons_by_factor=horizons_map)
    results["ic_rows"] = ic_rows
    results["ic_summary"] = ic.summarise_ic(ic_rows, (1, 6, 60))
    assert ic_rows and results["ic_summary"]
    assert all("ic_6" in r for r in ic_rows)
    # each factor also scored at its OWN horizon
    assert all("horizon_own" in r and "ic_own" in r for r in ic_rows)
    assert all("mean_abs_ic_own" in s for s in results["ic_summary"])

    bf_rows: list = []
    for p in tiny_preruns:
        if usable[p]:
            bf_rows += bruteforce.evaluate_prerun_models(p, usable[p], panel, cfg)
    results["bruteforce_rows"] = bf_rows
    assert bf_rows
    models_seen = {r["model"] for r in bf_rows}
    assert "ridge" in models_seen
    assert "ensemble" in models_seen  # 2 base models → ensemble produced

    tables = report.write_tables(cfg, results)
    figs = report.render_figures(cfg, results)
    report_md = report.write_report_md(cfg, results, figs)
    nb = report.build_comparison_notebook(cfg)

    assert (cfg.output_dir / "bruteforce_results.csv").exists()
    assert (cfg.output_dir / "ic_results.csv").exists()
    assert report_md.exists() and report_md.stat().st_size > 0
    assert figs, "expected at least one figure"
    for p in figs.values():
        assert p.exists() and p.stat().st_size > 0

    import nbformat
    parsed = nbformat.read(str(nb), as_version=4)
    assert parsed.cells, "notebook should have cells"

    after = hashlib.md5(global_db.read_bytes()).hexdigest() if global_db.exists() else None
    assert before == after, "global factor DB must not be mutated by a comparison"


def test_comparison_panel_loads_yfinance_provider(monkeypatch, tmp_path):
    """The comparison panel loader must honor API-provider settings, not ticker_data."""
    from quant_fund_agent.comparison import factors as cf
    from quant_fund_agent.data.providers.yfinance import YFinanceProvider
    from quant_fund_agent.modeling import service

    service._PANEL_CACHE = None
    service._SIGNAL_CACHE = {}
    monkeypatch.setattr(
        YFinanceProvider, "_fetch",
        lambda self, syms: {s: _synthetic_ohlcv(s) for s in syms},
    )

    panel = cf.load_panel_cached(
        data_dir="ticker_data",
        tickers=["AAA", "BBB"],
        data_overrides={
            "provider": "yfinance",
            "asset_class": "equity",
            "frequency": "1d",
            "start": "2023-01-02",
            "end": "2023-01-20",
            "cache_dir": str(tmp_path),
        },
    )

    assert list(panel["close"].columns) == ["AAA", "BBB"]
    assert {"open", "high", "low", "close", "volume", "vwap", "returns"} <= set(panel)
    assert "effSpread" not in panel
    assert service._PANEL_CACHE is panel


def test_main_comparison_alias_is_seed_baseline():
    """In comparison reports, 'main' means the seed library, not legacy/main."""
    from quant_fund_agent.comparison import factors as cf

    db = FactorDatabase()
    db.load_from_json(preruns.BASE_FACTOR_DB)
    expected = [f.id for f in db.list_factors(source=FactorSource.SEED)]

    assert cf.is_seed_baseline("main")
    assert cf.prerun_factor_ids("main") == expected
    assert cf.prerun_factor_ids("seed") == expected
    assert cf.factor_names(["main"])[expected[0]]
