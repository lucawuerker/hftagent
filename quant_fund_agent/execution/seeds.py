"""The seed executors — today's two hardcoded pipelines as executor programs.

These are genome #0 of the execution evolution (DESIGN §Genome): the exact
signal→positions constructions the deployed fund uses, re-expressed under the
``BaseExecutor`` contract.  They **delegate to the existing shared functions**
(``strategy_backtester._signal_to_positions`` and
``positions.per_underlying_positions``) so byte-equivalence with the current
pipelines holds *by construction* — asserted anyway in
``tests/test_execution_seeds.py``.  The legacy call sites keep calling those
functions directly until E4 flips deployment to resolve executors from the
registry; from then on this module IS the single signal→positions
implementation.

Both seeds are path-independent, so they implement the vectorised
``target_weights`` fast-path; the harness never needs the bar loop for them.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class TopKDollarNeutral(BaseExecutor):
    """The cross-sectional baseline: winsorise-free z-score → top-K → dollar-neutral.

    Byte-identical to ``backtesting.strategy_backtester._signal_to_positions``:
    hold between rebalances (``holding_period``), cross-sectional z-score, zero
    weak signals (``min_conviction``), keep the ``max_positions`` largest |z|
    per bar (sign → magnitude when ``equal_weight``), rescale to Σ|w| = 1
    (dollar-neutral long/short).
    """

    executor_id = "topk_dollar_neutral"
    name = "Top-K dollar-neutral (baseline cross-sectional)"
    description = ("Cross-sectional z-score, top-max_positions by |z|, "
                   "rescaled to a dollar-neutral unit-gross book.")
    regime = "cross_sectional"
    inputs = ["signal"]
    params = {"holding_period": 1, "max_positions": 20,
              "min_conviction": 0.0, "equal_weight": 0}

    def target_weights(self, signal: pd.DataFrame,
                       state: dict[str, pd.DataFrame]) -> pd.DataFrame:
        from quant_fund_agent.backtesting.strategy_backtester import (
            _signal_to_positions,
        )

        p = {**type(self).params, **(getattr(self, "overrides", None) or {})}
        return _signal_to_positions(
            signal,
            holding_period=int(p["holding_period"]),
            max_positions=int(p["max_positions"]),
            equal_weight=bool(p["equal_weight"]),
            min_conviction=float(p["min_conviction"]),
        )


@register_executor
class ZScoreThresholdEqualWeight(BaseExecutor):
    """The per-underlying baseline: own-history z-score → threshold band → 1/n.

    Byte-identical to ``backtesting.positions.per_underlying_positions``:
    hold between rebalances, per-underlying (causal expanding) z-score, map to a
    long/flat/short directional position by the boundary, size each active name
    to ``1 / n_max_positions`` equal weight (directional, net market exposure).
    """

    executor_id = "zscore_threshold_equal_weight"
    name = "Z-threshold equal-weight (baseline per-underlying)"
    description = ("Per-underlying causal z-score, ±threshold directional band, "
                   "1/n equal-weight sizing.")
    regime = "per_underlying"
    inputs = ["signal"]
    params = {"n_max_positions": 6, "holding_period": 1, "threshold": 1.0,
              "zscore_window": 500}
    # non-numeric knobs (mode / basis) stay class config, not jitter params
    mode = "threshold"
    zscore_basis = "expanding"

    def target_weights(self, signal: pd.DataFrame,
                       state: dict[str, pd.DataFrame]) -> pd.DataFrame:
        from quant_fund_agent.backtesting.positions import per_underlying_positions

        p = {**type(self).params, **(getattr(self, "overrides", None) or {})}
        return per_underlying_positions(
            signal,
            n_max_positions=int(p["n_max_positions"]),
            holding_period=int(p["holding_period"]),
            mode=str(getattr(self, "mode", "threshold")),
            threshold=float(p["threshold"]),
            zscore_basis=str(getattr(self, "zscore_basis", "expanding")),
            zscore_window=int(p["zscore_window"]),
        )


SEED_EXECUTOR_IDS = ("topk_dollar_neutral", "zscore_threshold_equal_weight")
