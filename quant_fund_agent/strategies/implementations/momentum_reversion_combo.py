"""Example strategy: Momentum + Mean-Reversion Composite.

Demonstrates a simple static-weight blend of two alpha factors.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.strategies.base import BaseStrategy, ModelType
from quant_fund_agent.strategies.registry import register_strategy


@register_strategy
class MomentumReversionCombo(BaseStrategy):
    strategy_id = "momentum_reversion_combo"
    name = "Momentum + Reversion Composite"
    description = (
        "Equal-weight blend of the Three Soldiers momentum signal and "
        "the RSI mean-reversion signal."
    )
    factor_ids = ["three_soldiers", "rsi_mean_reversion"]
    model_type = ModelType.STATIC_WEIGHTS
    holding_period = 6
    model_params = {"weights": {"three_soldiers": 0.5, "rsi_mean_reversion": 0.5}}

    def calc(
        self,
        factor_signals: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        weights = self.model_params["weights"]
        combined = sum(
            w * factor_signals[fid]
            for fid, w in weights.items()
        )
        return combined
