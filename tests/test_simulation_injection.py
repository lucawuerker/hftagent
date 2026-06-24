"""Tests for prerun factor injection + the analytics/report additions.

Deterministic and synthetic — no ``ticker_data/``, no LLM, no network — so they
pin down: the draw-without-replacement pool, the config validation, appending
injected factors to the run catalog on disk, the exposure/NAV accumulation in
``BacktestResults``, and that the report renders figures + ``report.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorRecord, FactorSource


# ── helpers ─────────────────────────────────────────────────────────────

def _write_prerun_db(path, ids: list[str]) -> None:
    db = FactorDatabase()
    for i in ids:
        db.add_factor(FactorRecord(
            id=i, name=i, class_name=f"Cls_{i}",
            source=FactorSource.RESEARCHER, research_session_id="prerun",
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    db.save_to_json(path)


# ── PrerunFactorPool ─────────────────────────────────────────────────────

def test_pool_unions_dedups_and_draws_without_replacement(tmp_path, monkeypatch):
    from quant_fund_agent.simulation import factor_injection as fi

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_prerun_db(a, ["f1", "f2", "f3", "shared"])
    _write_prerun_db(b, ["f4", "f5", "shared"])  # 'shared' duplicated across preruns
    mapping = {"A": a, "B": b}
    monkeypatch.setattr(fi, "resolve_prerun_db", lambda name, cfg: mapping[name])

    pool = fi.PrerunFactorPool(["A", "B"], config_name=None, seed=1)
    assert pool.total == 6            # f1..f5 + 'shared' once (deduped across preruns)
    assert pool.remaining == 6

    drawn_ids: list[str] = []
    while not pool.exhausted:
        drawn_ids += [f.id for f in pool.draw(2)]
    assert sorted(drawn_ids) == ["f1", "f2", "f3", "f4", "f5", "shared"]
    assert len(drawn_ids) == len(set(drawn_ids))                # no factor twice
    assert pool.draw(2) == []                                   # exhausted → empty


def test_pool_seed_is_deterministic(tmp_path, monkeypatch):
    from quant_fund_agent.simulation import factor_injection as fi

    a = tmp_path / "a.json"
    _write_prerun_db(a, [f"f{i}" for i in range(10)])
    monkeypatch.setattr(fi, "resolve_prerun_db", lambda name, cfg: a)

    o1 = [f.id for f in fi.PrerunFactorPool(["A"], seed=42).draw(10)]
    o2 = [f.id for f in fi.PrerunFactorPool(["A"], seed=42).draw(10)]
    o3 = [f.id for f in fi.PrerunFactorPool(["A"], seed=7).draw(10)]
    assert o1 == o2          # same seed → same order
    assert o1 != o3          # different seed → (very likely) different order


def test_pool_skips_missing_prerun(tmp_path, monkeypatch):
    from quant_fund_agent.simulation import factor_injection as fi

    a = tmp_path / "a.json"
    _write_prerun_db(a, ["f1", "f2"])
    missing = tmp_path / "nope.json"
    mapping = {"A": a, "MISSING": missing}
    monkeypatch.setattr(fi, "resolve_prerun_db", lambda name, cfg: mapping[name])

    pool = fi.PrerunFactorPool(["A", "MISSING"], seed=1)
    assert pool.total == 2   # missing prerun contributes nothing, no crash


# ── config validation ────────────────────────────────────────────────────

def test_config_rejects_inject_without_preruns():
    from quant_fund_agent.simulation.config import BacktestConfig

    with pytest.raises(ValueError):
        BacktestConfig(start="2020-01-01", end="2020-03-01",
                       factor_source="prerun_inject", inject_preruns=[])


def test_config_accepts_inject_with_preruns():
    from quant_fund_agent.simulation.config import BacktestConfig

    cfg = BacktestConfig(start="2020-01-01", end="2020-03-01",
                         factor_source="prerun_inject", inject_preruns=["x"])
    assert cfg.factor_source == "prerun_inject"
    assert "inject_preruns" in cfg.to_dict()


# ── appending injected factors to the run catalog ─────────────────────────

def test_append_factors_writes_and_dedups(tmp_path):
    from quant_fund_agent.simulation.simulator import _append_factors

    db_path = tmp_path / "factor_db.json"
    seed_db = FactorDatabase()
    seed_db.add_factor(FactorRecord(id="seed1", name="seed1", source=FactorSource.SEED))
    seed_db.save_to_json(db_path)

    recs = [FactorRecord(id=f"r{i}", name=f"r{i}", source=FactorSource.RESEARCHER)
            for i in range(3)]
    _append_factors(db_path, recs)
    _append_factors(db_path, recs)  # idempotent — no duplicates

    out = FactorDatabase()
    out.load_from_json(db_path)
    ids = sorted(f.id for f in out.list_factors())
    assert ids == ["r0", "r1", "r2", "seed1"]


# ── exposure / NAV accumulation + report rendering ────────────────────────

def _synthetic_results(tmp_path):
    from quant_fund_agent.simulation.config import BacktestConfig
    from quant_fund_agent.simulation.execution import ExecutionResult
    from quant_fund_agent.simulation.results import BacktestResults

    cfg = BacktestConfig(start="2020-01-01", end="2020-04-01",
                         output_root=str(tmp_path), run_id="rep", capital=1_000_000.0)
    res = BacktestResults(cfg)

    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-02", periods=60, freq="B")
    ret = pd.Series(rng.standard_normal(60) * 1e-3, index=idx, name="fund_return")
    expo = pd.Series(np.clip(0.8 + rng.standard_normal(60) * 0.05, 0, 1), index=idx)
    npos = pd.Series(rng.integers(5, 20, 60), index=idx, dtype=float)
    res.add_window(ExecutionResult(
        fund_returns=ret, gross_returns=ret, cost=ret * 0,
        turnover=expo * 0, attribution={"stratA": ret * 0.6, "stratB": ret * 0.4},
        gross_exposure=expo, n_positions=npos,
    ))
    res.log_meeting({"cutoff": idx[0].isoformat(), "injected_factor_ids": ["f1", "f2"],
                     "strategies_approved": ["stratA"]})
    res.finalize()
    return cfg, res


def test_results_track_exposure_and_nav(tmp_path):
    cfg, res = _synthetic_results(tmp_path)
    m = res.metrics
    assert "pct_invested_mean" in m and 0 < m["pct_invested_mean"] <= 1.0
    assert "n_positions_mean" in m
    assert "final_nav" in m
    # NAV starts near capital and is a proper path.
    nav = res.nav
    assert abs(nav.iloc[0] - cfg.capital) < cfg.capital * 0.05
    assert len(nav) == len(res.fund_returns)


def test_report_renders_figures_and_markdown(tmp_path):
    cfg, res = _synthetic_results(tmp_path)
    out = res.save()
    report_md = out / "report" / "report.md"
    assert report_md.exists()
    text = report_md.read_text()
    assert "Key metrics" in text and "NAV" in text
    # core figures present
    for fig in ("nav_drawdown.png", "percent_invested.png", "monthly_returns.png"):
        assert (out / "report" / fig).exists(), f"missing {fig}"
    # equity.csv now carries exposure + nav columns
    eq = pd.read_csv(out / "equity.csv")
    for col in ("nav", "gross_exposure", "n_positions", "drawdown"):
        assert col in eq.columns


# ── end-to-end: the simulator injects on the research cadence (no LLM) ─────

def test_simulator_injects_prerun_factors(tmp_path, monkeypatch):
    """Full walk-forward loop in prerun_inject mode with stubbed agents + panel.

    Stubs only the LLM/data boundaries; the injection branch, clock, per-window
    backtest, cost accounting and report all run for real.  Asserts that factors
    are drawn from the pool onto the run catalog on each research-due meeting.
    """
    from quant_fund_agent import pipeline
    from quant_fund_agent.modeling import service
    from quant_fund_agent.schemas import (
        ConstructionMethod,
        PMStatus,
        PortfolioAllocation,
        PortfolioRecord,
        StrategyRecord,
        StrategyStatus,
    )
    from quant_fund_agent.simulation import factor_injection as fi

    # temp prerun pool of 8 factors
    prerun_db = tmp_path / "prerun.json"
    _write_prerun_db(prerun_db, [f"pf{i}" for i in range(8)])
    monkeypatch.setattr(fi, "resolve_prerun_db", lambda name, cfg: prerun_db)

    # synthetic daily panel + factor signal (no ticker_data)
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", "2020-07-01", freq="B")
    cols = ["S0", "S1", "S2", "S3"]
    close = pd.DataFrame(100 + rng.standard_normal((len(idx), 4)).cumsum(0),
                         index=idx, columns=cols).clip(lower=1.0)
    panel = {"close": close,
             "effSpread": pd.DataFrame(0.05, index=idx, columns=cols)}
    signal = pd.DataFrame(rng.standard_normal((len(idx), 4)), index=idx, columns=cols)
    monkeypatch.setattr(service, "_load_panel", lambda: panel)
    monkeypatch.setattr(service, "_factor_signal", lambda fid, data: signal)
    # The pool's panel-computability gate is out of scope here (synthetic pf* have
    # no registered class) — treat every pooled factor as usable.
    import quant_fund_agent.comparison.factors as cmp_factors
    monkeypatch.setattr(cmp_factors, "usable_factor_ids",
                        lambda ids, panel: (list(ids), {}))

    counter = {"n": 0}

    def fake_persist(strategy_db, res, strategy_id=None):
        if counter["n"] >= 2:
            return None
        counter["n"] += 1
        sid = f"strat_{counter['n']}"
        rec = StrategyRecord(
            id=sid, name=sid, class_name="DynamicStrategy",
            factor_ids=["f1"], status=StrategyStatus.APPROVED,
            model_type="static_weights",
            model_params={"weights": {"f1": 1.0}, "holding_period": 1,
                          "max_positions": 4, "equal_weight": False,
                          "min_conviction": 0.0},
            holding_period=1,
        )
        seed = pd.Series(rng.standard_normal(40) * 1e-4,
                         index=pd.date_range("2019-12-01", periods=40, freq="B"))
        strategy_db.register_strategy(rec, portfolio_returns=seed)
        return rec

    def fake_pm(strategy_db, portfolio_db, **kw):
        live = [s for s in strategy_db.list_strategies()
                if s.pm_status != PMStatus.RETIRED]
        if not live:
            return None
        w = 1.0 / len(live)
        for s in live:
            strategy_db.set_pm_status(s.id, PMStatus.LIVE)
        rec = PortfolioRecord(
            id=f"pf_{len(portfolio_db.list_portfolios())}",
            pm_name=kw.get("pm_name", "fund_committee"),
            construction_method=ConstructionMethod.EQUAL_WEIGHT,
            allocations=[PortfolioAllocation(strategy_id=s.id, weight=w, enabled=True)
                         for s in live],
        )
        portfolio_db.add_portfolio(rec)
        return rec

    monkeypatch.setattr(pipeline, "run_strategy_pipeline", lambda **kw: object())
    monkeypatch.setattr(pipeline, "persist_approved_strategy", fake_persist)
    monkeypatch.setattr(pipeline, "run_pm_rebalance", fake_pm)

    from quant_fund_agent.simulation import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        start="2020-01-01", end="2020-07-01", warmup="1M", grid_freq="1M",
        research_every="1M", strategy_every="1M", pm_rebalance_every="1M",
        factor_source="prerun_inject", inject_preruns=["A"], factors_per_inject=2,
        initial_strategies=1, n_strategies_per_meeting=1,
        execution_model="netted", output_root=str(tmp_path), run_id="inject_e2e",
    )
    results = run_backtest(cfg)

    # injected ids were logged across the meetings
    injected = [fid for m in results.meetings for fid in m.get("injected_factor_ids", [])]
    assert injected, "expected factors to be injected on research-due meetings"
    assert len(injected) == len(set(injected)), "no factor injected twice"

    # the run catalog now holds the injected factors (+ seeds), persisted on disk
    db = FactorDatabase()
    db.load_from_json(cfg.factor_db_path)
    catalog_ids = {f.id for f in db.list_factors()}
    assert set(injected) <= catalog_ids
    # report rendered
    assert (cfg.output_dir / "report" / "report.md").exists()
