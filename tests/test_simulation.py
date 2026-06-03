"""Tests for the walk-forward backtest harness (``quant_fund_agent.simulation``).

These are deliberately **synthetic** — no ``ticker_data/`` and no LLM calls — so
they always run and pin down the deterministic core: the meeting schedule, the
spread-aware cost model, position netting, and exact parity with the existing
single-strategy backtester when costs/constraints are off.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.simulation.config import (
    BacktestConfig,
    ExecutionModel,
    cadence_steps,
    offset_to_days,
)
from quant_fund_agent.simulation.clock import TradingClock
from quant_fund_agent.simulation.execution import (
    apply_fund_constraints,
    consolidate_book,
    cost_rate_panel,
    realised_netted,
    realised_pod,
)


def _cfg(**overrides) -> BacktestConfig:
    base = dict(start="2020-01-06", end="2020-03-02")
    base.update(overrides)
    return BacktestConfig(**base)


# ── offset / cadence parsing ────────────────────────────────────────────

def test_offset_parsing():
    assert offset_to_days("1W") == 7
    assert offset_to_days("2W") == 14
    assert offset_to_days("1M") == 30
    assert offset_to_days("10D") == 10
    assert cadence_steps("1W", "1W") == 1
    assert cadence_steps("1M", "1W") == 4    # monthly meeting on a weekly grid
    assert cadence_steps("2W", "1W") == 2


def test_bad_offset_raises():
    with pytest.raises(ValueError):
        offset_to_days("nonsense")


# ── trading clock / schedule ─────────────────────────────────────────────

def _hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="1h", inclusive="left")


def test_clock_warmup_and_first_meeting():
    cfg = _cfg(warmup="2W", grid_freq="1W")
    idx = _hourly_index(cfg.start, cfg.end)
    clock = TradingClock(idx, cfg)
    periods = clock.periods()

    assert periods, "expected at least one period"
    first = periods[0]
    # No meeting before warm-up has elapsed.
    assert (first.cutoff - pd.Timestamp(cfg.start_date)).days >= 14
    # The first strategy meeting must be flagged exactly once, at the first cutoff.
    assert first.is_first_strategy_meeting is True
    assert sum(p.is_first_strategy_meeting for p in periods) == 1


def test_clock_windows_are_contiguous_and_non_overlapping():
    cfg = _cfg(warmup="2W", grid_freq="1W")
    clock = TradingClock(_hourly_index(cfg.start, cfg.end), cfg)
    periods = clock.periods()
    for a, b in zip(periods, periods[1:]):
        assert a.live_end == b.live_start          # contiguous
        assert a.live_start < a.live_end           # non-empty
        assert a.cutoff == a.live_start            # cutoff opens the live window


def test_clock_meeting_cadence():
    cfg = _cfg(warmup="2W", grid_freq="1W", research_every="2W",
               strategy_every="1W", pm_rebalance_every="1W")
    clock = TradingClock(_hourly_index(cfg.start, cfg.end), cfg)
    periods = clock.periods()
    # Dense data ⇒ no skipped grid points ⇒ index 0,1,2,...
    for p in periods:
        assert p.strategy_due is True              # every week
        assert p.pm_due is True
        assert p.research_due == (p.index % 2 == 0)  # every other week


# ── cost model ────────────────────────────────────────────────────────────

def _flat_panel(close=100.0, spread=0.10, n=4, t=4):
    idx = pd.date_range("2020-01-06 09:30", periods=t, freq="10s")
    cols = [f"S{i}" for i in range(n)]
    close_df = pd.DataFrame(close, index=idx, columns=cols)
    spread_df = pd.DataFrame(spread, index=idx, columns=cols)
    return {"close": close_df, "effSpread": spread_df}


def test_cost_rate_spread_plus_commission():
    panel = _flat_panel(close=100.0, spread=0.10)
    cfg = _cfg(commission_bps=10.0, spread_field="effSpread", half_spread=True)
    rate = cost_rate_panel(panel, cfg)
    # half_spread = 0.5 * (0.10/100) = 0.0005 ; commission = 10bps = 0.001
    assert np.allclose(rate.values, 0.0005 + 0.001)


def test_cost_rate_commission_only_when_spread_disabled():
    panel = _flat_panel()
    cfg = _cfg(commission_bps=5.0, use_spread_cost=False)
    rate = cost_rate_panel(panel, cfg)
    assert np.allclose(rate.values, 0.0005)


# ── netting semantics ─────────────────────────────────────────────────────

def _positions(values: dict[str, float], idx) -> pd.DataFrame:
    return pd.DataFrame({k: [v] * len(idx) for k, v in values.items()}, index=idx)


def test_offsetting_signals_net_to_zero():
    idx = pd.date_range("2020-01-06 09:30", periods=3, freq="10s")
    p_a = _positions({"AAPL": 0.5, "MSFT": -0.5}, idx)
    p_b = _positions({"AAPL": -0.5, "MSFT": 0.5}, idx)   # exact opposite of A
    book = consolidate_book({"A": p_a, "B": p_b}, {"A": 0.5, "B": 0.5})
    assert np.allclose(book.values, 0.0)               # they fully net out


def test_agreeing_signals_reinforce():
    idx = pd.date_range("2020-01-06 09:30", periods=3, freq="10s")
    p_a = _positions({"AAPL": 0.5, "MSFT": -0.5}, idx)
    p_b = _positions({"AAPL": 0.5, "MSFT": -0.5}, idx)   # same as A
    book = consolidate_book({"A": p_a, "B": p_b}, {"A": 0.5, "B": 0.5})
    assert np.allclose(book["AAPL"].values, 0.5)        # 0.5*0.5 + 0.5*0.5
    assert np.allclose(book["MSFT"].values, -0.5)


def test_fund_constraints_cap_positions_and_leverage():
    idx = pd.date_range("2020-01-06 09:30", periods=1, freq="10s")
    book = _positions({"A": 0.4, "B": 0.3, "C": -0.2, "D": -0.1}, idx)
    capped = apply_fund_constraints(book, max_positions=2, max_weight_per_name=1.0,
                                    gross_leverage=1.0)
    # only the two largest |weights| (A, B) survive
    nonzero = (capped.abs() > 1e-12).iloc[0]
    assert nonzero["A"] and nonzero["B"]
    assert not nonzero["C"] and not nonzero["D"]
    # gross leverage capped at 1.0
    assert capped.abs().sum(axis=1).iloc[0] <= 1.0 + 1e-9


# ── parity with the single-strategy backtester ───────────────────────────

def _toy_strategy_and_data(seed=0, t=60, n=4):
    from quant_fund_agent.strategies.dynamic import DynamicStrategy

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-06 09:30", periods=t, freq="10s")
    cols = [f"S{i}" for i in range(n)]
    close = pd.DataFrame(100 + rng.standard_normal((t, n)).cumsum(0), index=idx, columns=cols)
    spread = pd.DataFrame(0.05, index=idx, columns=cols)
    signal = pd.DataFrame(rng.standard_normal((t, n)), index=idx, columns=cols)

    data = {"close": close, "effSpread": spread}
    factor_signals = {"f1": signal}
    strat = DynamicStrategy(
        strategy_id="toy", name="toy", description="",
        factor_ids=["f1"], weights={"f1": 1.0},
        holding_period=1, max_positions=n, equal_weight=False, min_conviction=0.0,
    )
    return strat, factor_signals, data


@pytest.mark.parametrize("model", [ExecutionModel.NETTED, ExecutionModel.POD])
def test_zero_cost_single_strategy_matches_backtester(model):
    from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy

    strat, signals, data = _toy_strategy_and_data()
    res = backtest_strategy(strat, signals, data)

    cfg = _cfg(execution_model=model, commission_bps=0.0, use_spread_cost=False,
               gross_leverage=1.0, fund_max_positions=10, max_weight_per_name=1.0)
    positions = {"toy": res.positions}
    weights = {"toy": 1.0}
    out = (realised_netted if model == ExecutionModel.NETTED else realised_pod)(
        positions, weights, data, cfg,
    )

    expected = res.portfolio_returns.reindex(out.fund_returns.index)
    assert np.allclose(out.fund_returns.values, expected.values, atol=1e-12, equal_nan=True)
    assert np.allclose(out.cost.values, 0.0)


# ── cutoff / no-leakage helper ────────────────────────────────────────────

def test_truncate_as_of_is_exclusive():
    from quant_fund_agent.modeling.service import _truncate_as_of

    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    frames = {"close": pd.DataFrame({"A": range(10)}, index=idx)}
    cut = "2020-01-06"
    out = _truncate_as_of(frames, cut)
    assert out["close"].index.max() < pd.Timestamp(cut)   # strictly before cutoff
    # None is a no-op (production behaviour).
    assert len(_truncate_as_of(frames, None)["close"]) == 10


def test_run_strategy_pipeline_accepts_cutoff_date():
    from quant_fund_agent import pipeline

    sig = inspect.signature(pipeline.run_strategy_pipeline)
    assert "cutoff_date" in sig.parameters


# ── end-to-end loop wiring (stub only the LLM meetings + panel loader) ─────

def test_simulator_end_to_end_wiring(monkeypatch, tmp_path):
    """Drive the full walk-forward loop with stubbed agents + a synthetic panel.

    Stubs only the LLM/data boundaries (research/strategy/PM meetings and the
    panel loader); everything else runs for real: the clock, rebuilding a
    strategy from its record, the per-window backtest, position netting, cost
    accounting, live-PnL marking, results aggregation, and persistence.
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

    # synthetic hourly panel + factor signal (no ticker_data, no warm-up issues)
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-06", "2020-02-24", freq="1h", inclusive="left")
    cols = ["S0", "S1", "S2", "S3"]
    close = pd.DataFrame(100 + rng.standard_normal((len(idx), 4)).cumsum(0),
                         index=idx, columns=cols).clip(lower=1.0)
    panel = {
        "close": close,
        "effSpread": pd.DataFrame(0.05, index=idx, columns=cols),
    }
    signal = pd.DataFrame(rng.standard_normal((len(idx), 4)), index=idx, columns=cols)
    monkeypatch.setattr(service, "_load_panel", lambda: panel)
    monkeypatch.setattr(service, "_factor_signal", lambda fid, data: signal)

    # stub the three LLM meetings
    counter = {"n": 0}

    def fake_persist(strategy_db, res, strategy_id=None):
        if counter["n"] >= 3:        # cap the book
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
        seed = pd.Series(rng.standard_normal(50) * 1e-4,
                         index=pd.date_range("2020-01-02", periods=50, freq="10min"))
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
            id=f"pf_{kw.get('pm_name', 'pm')}_{len(portfolio_db.list_portfolios())}",
            pm_name=kw.get("pm_name", "fund_committee"),
            construction_method=ConstructionMethod.EQUAL_WEIGHT,
            allocations=[PortfolioAllocation(strategy_id=s.id, weight=w, enabled=True)
                         for s in live],
        )
        portfolio_db.add_portfolio(rec)
        return rec

    monkeypatch.setattr(pipeline, "run_research_session",
                        lambda **kw: {"kept_factor_ids": []})
    monkeypatch.setattr(pipeline, "run_strategy_pipeline", lambda **kw: object())
    monkeypatch.setattr(pipeline, "persist_approved_strategy", fake_persist)
    monkeypatch.setattr(pipeline, "run_pm_rebalance", fake_pm)

    from quant_fund_agent.simulation import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        start="2020-01-06", end="2020-02-24", warmup="2W", grid_freq="1W",
        initial_strategies=2, n_strategies_per_meeting=1,
        execution_model="netted", commission_bps=0.5,
        output_root=str(tmp_path), run_id="wiring_test",
    )
    results = run_backtest(cfg)

    m = results.metrics
    assert m["n_bars"] > 0
    assert m["n_strategies_deployed"] >= 1
    assert len(results.fund_returns) > 0
    assert results.meetings, "expected meeting log entries"
    # persisted artifacts exist
    out = cfg.output_dir
    assert (out / "equity.csv").exists()
    assert (out / "fund_metrics.json").exists()
    assert (out / "meetings.jsonl").exists()
    # costs were actually charged (commission > 0)
    assert m["total_cost"] > 0.0
