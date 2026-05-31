"""ML strategy persistence happy-path (deterministic, no LLM / no MCP).

The live ``run_fund.py`` run exercises Selector → Architect → Statistician, but
whether a strategy is *accepted* depends on real OOS edge.  This test pins down
the persistence + reload path on its own: an approved ML pipeline result must
become a ``ModelStrategy`` record whose persisted artifact can be reloaded into a
runnable strategy (the basis for the future live backtest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent import pipeline
from quant_fund_agent.agents.architect.state import StrategySpec
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.modeling import fit_and_backtest
from quant_fund_agent.strategies.model_strategy import ModelStrategy

FACTOR_IDS = ["f1", "f2"]


def _panel(n_bars=400, n_tickers=6):
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="10s")
    cols = [f"T{i}" for i in range(n_tickers)]
    signals = {
        fid: pd.DataFrame(rng.normal(size=(n_bars, n_tickers)), index=idx, columns=cols)
        for fid in FACTOR_IDS
    }
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (n_bars, n_tickers)), axis=0),
        index=idx, columns=cols,
    )
    return signals, {"close": close}


def test_ml_strategy_persist_and_reload(tmp_path):
    signals, data = _panel()
    fit = fit_and_backtest(
        factor_signals_is=signals,
        data_is=data,
        model_type="ridge",
        factor_ids=FACTOR_IDS,
        target_horizon=6,
        artifact_dir=tmp_path,
        strategy_id="persist_ridge",
    )

    spec = StrategySpec(
        strategy_name="Ridge Test",
        model_type="ridge",
        model_params={"alpha": 1.0},
        factor_ids=FACTOR_IDS,
        target_horizon=6,
        model_artifact_path=fit["artifact_path"],
        holding_period=6,
        max_positions=5,
        reasoning="test",
    )
    result = pipeline.StrategyPipelineResult(
        selector_result={"hypothesis": "ridge hypothesis"},
        architect_result={
            "strategy_spec": spec,
            "backtest_metrics": fit["metrics"],
            "trial_history": [],
        },
        stat_result=None,
        approved=True,
    )

    record, is_returns = pipeline.build_strategy_record(result)
    assert record.class_name == "ModelStrategy"
    assert record.model_type == "ridge"
    assert record.factor_ids == FACTOR_IDS
    assert record.model_params["model_artifact_path"] == fit["artifact_path"]
    assert is_returns is not None and len(is_returns) > 0

    # The persisted record reconstructs into a runnable ModelStrategy.
    strategy = pipeline.strategy_from_record(record)
    assert isinstance(strategy, ModelStrategy)
    assert strategy.factor_ids == FACTOR_IDS

    bt = backtest_strategy(strategy, signals, data)
    assert bt.metrics.sharpe_ratio is not None
