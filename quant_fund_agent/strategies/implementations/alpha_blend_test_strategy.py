"""Test strategy: simple blend of a few Alpha factors.

This is intentionally mechanical and fast:
  - no training
  - no state
  - just a weighted sum of pre-computed factor signals

It is meant as a smoke test for the vectorised strategy backtester.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.strategies.base import BaseStrategy, ModelType
from quant_fund_agent.strategies.registry import register_strategy


@register_strategy
class AlphaBlendTestStrategy(BaseStrategy):
    strategy_id = "alpha_blend_test"
    name = "Alpha Blend Test Strategy"
    description = (
        "Mechanical weighted blend of a few alphas (test for the "
        "vectorised strategy backtester)."
    )

    # pick a few alphas that exist and are cheap to compute
    factor_ids = ["alpha_101", "alpha_012", "alpha_033"]
    model_type = ModelType.STATIC_WEIGHTS

    # position construction knobs (handled by backtester)
    holding_period = 6  # rebalance every minute (10s bars)
    max_positions = 20
    equal_weight = False
    min_conviction = 0.05

    model_params = {
        "weights": {
            "alpha_101": 0.34,
            "alpha_012": 0.33,
            "alpha_033": 0.33,
        }
    }

    def calc(
        self,
        factor_signals: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        weights: dict[str, float] = self.model_params["weights"]
        # factor_signals are already normalised by the backtester
        return sum(weights[fid] * factor_signals[fid] for fid in self.factor_ids)

