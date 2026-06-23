"""Out-of-sample backtest test.

Re-runs the approved strategy spec on the held-out OOS slice of the data
that the architect never saw, and compares the OOS performance to the
in-sample performance.

A strategy with genuine edge should generalise: the OOS Sharpe should be
of the same sign as IS, and not catastrophically smaller.
"""

from __future__ import annotations

import uuid

from quant_fund_agent.backtesting.data_split import split_panel, split_signals
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.statistics.base import BaseStatTest, StatTestContext
from quant_fund_agent.statistics.registry import register_test
from quant_fund_agent.statistics.schemas import StatTestResult, StatTestVerdict
from quant_fund_agent.strategies.dynamic import DynamicStrategy


@register_test
class OutOfSampleTest(BaseStatTest):
    test_id = "out_of_sample_backtest"
    name = "Out-of-sample backtest"
    description = (
        "Runs the final strategy spec on the held-out OOS data and checks "
        "that IS edge persists (sign agreement & Sharpe decay)."
    )
    is_mandatory = True

    # Maximum tolerated decay of the OOS Sharpe vs the IS Sharpe (fraction).
    # If OOS Sharpe decays more than this we flag a WARN; positive OOS Sharpe
    # (same sign as IS) is always required to PASS.  For intraday microstructure
    # data a 20% OOS split covering a different market regime makes large decay
    # structurally likely even for genuine strategies, so 0.85 is the bar here.
    max_sharpe_decay: float = 0.85

    def run(self, ctx: StatTestContext) -> StatTestResult:
        if ctx.data_full is None or ctx.factor_signals_full is None:
            return StatTestResult(
                test_id=self.test_id,
                test_name=self.name,
                verdict=StatTestVerdict.FAIL,
                summary="No data / factor signals provided for OOS test.",
            )

        spec = ctx.strategy_spec

        _, data_oos = split_panel(ctx.data_full, ctx.oos_split_ratio)
        _, signals_oos = split_signals(ctx.factor_signals_full, ctx.oos_split_ratio)

        if len(next(iter(data_oos.values()))) < 30:
            return StatTestResult(
                test_id=self.test_id,
                test_name=self.name,
                verdict=StatTestVerdict.FAIL,
                summary="OOS slice is too short for a meaningful test.",
                details={"oos_len": len(next(iter(data_oos.values())))},
            )

        # Rebuild the *exact same* strategy on OOS.  For a fitted ML model we
        # reload the persisted estimator (never refit on OOS data); for the
        # static baseline we reconstruct the DynamicStrategy from its weights.
        if spec.model_type and spec.model_type != "static_weights":
            from quant_fund_agent.strategies.model_strategy import ModelStrategy

            if not spec.model_artifact_path:
                return StatTestResult(
                    test_id=self.test_id,
                    test_name=self.name,
                    verdict=StatTestVerdict.FAIL,
                    summary="ML strategy has no persisted model artifact for the OOS test.",
                )
            strategy = ModelStrategy.from_artifact(
                spec.model_artifact_path,
                strategy_id=f"stat_oos_{uuid.uuid4().hex[:8]}",
                name=spec.strategy_name + " [OOS]",
                description=ctx.hypothesis,
                holding_period=spec.holding_period,
                max_positions=spec.max_positions,
                equal_weight=spec.equal_weight,
                min_conviction=spec.min_conviction,
                position_construction=spec.position_construction,
                position_params=spec.position_params,
            )
        else:
            strategy = DynamicStrategy(
                strategy_id=f"stat_oos_{uuid.uuid4().hex[:8]}",
                name=spec.strategy_name + " [OOS]",
                description=ctx.hypothesis,
                factor_ids=list(spec.weights.keys()),
                weights=spec.weights,
                holding_period=spec.holding_period,
                max_positions=spec.max_positions,
                equal_weight=spec.equal_weight,
                min_conviction=spec.min_conviction,
                position_construction=spec.position_construction,
                position_params=spec.position_params,
            )
        oos_result = backtest_strategy(strategy, signals_oos, data_oos)

        is_sharpe = float(ctx.is_metrics.get("sharpe_ratio") or 0.0)
        oos_sharpe = float(oos_result.metrics.sharpe_ratio or 0.0)
        is_ann_ret = float(ctx.is_metrics.get("annualised_return") or 0.0)
        oos_ann_ret = float(oos_result.metrics.annualised_return or 0.0)
        is_mdd = float(ctx.is_metrics.get("max_drawdown") or 0.0)
        oos_mdd = float(oos_result.metrics.max_drawdown or 0.0)

        # Decision rule
        if is_sharpe <= 0:
            verdict = StatTestVerdict.INFO
            summary = "In-sample Sharpe was non-positive; OOS comparison uninformative."
        elif oos_sharpe <= 0:
            verdict = StatTestVerdict.FAIL
            summary = (
                f"OOS Sharpe is {oos_sharpe:.2f} (IS: {is_sharpe:.2f}). "
                "Edge did not generalise."
            )
        else:
            decay = 1.0 - oos_sharpe / is_sharpe
            if decay <= self.max_sharpe_decay:
                verdict = StatTestVerdict.PASS
                summary = (
                    f"OOS Sharpe {oos_sharpe:.2f} vs IS {is_sharpe:.2f} "
                    f"(decay {decay * 100:.0f}%). Edge persists."
                )
            else:
                verdict = StatTestVerdict.WARN
                summary = (
                    f"OOS Sharpe {oos_sharpe:.2f} vs IS {is_sharpe:.2f} "
                    f"(decay {decay * 100:.0f}% exceeds {self.max_sharpe_decay * 100:.0f}% limit)."
                )

        return StatTestResult(
            test_id=self.test_id,
            test_name=self.name,
            verdict=verdict,
            value=round(oos_sharpe, 4),
            threshold=round(is_sharpe * (1 - self.max_sharpe_decay), 4),
            summary=summary,
            details={
                "is_sharpe": round(is_sharpe, 4),
                "oos_sharpe": round(oos_sharpe, 4),
                "is_annualised_return": round(is_ann_ret, 6),
                "oos_annualised_return": round(oos_ann_ret, 6),
                "is_max_drawdown": round(is_mdd, 6),
                "oos_max_drawdown": round(oos_mdd, 6),
                "oos_metrics": oos_result.metrics.model_dump(),
                "oos_n_bars": int(len(oos_result.portfolio_returns)),
            },
        )
