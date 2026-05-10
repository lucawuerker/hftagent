"""A strategy that can be fully configured at runtime.

Used by the Architect Agent to build strategies on the fly without
writing a Python file or registering a class.  Supports static
weighted combinations of factor signals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_fund_agent.strategies.base import BaseStrategy, ModelType


class DynamicStrategy(BaseStrategy):
    """Strategy configured entirely through constructor arguments."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        description: str,
        factor_ids: list[str],
        weights: dict[str, float],
        holding_period: int = 1,
        max_positions: int = 20,
        equal_weight: bool = False,
        min_conviction: float = 0.0,
    ) -> None:
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.factor_ids = list(factor_ids)
        self.model_type = ModelType.STATIC_WEIGHTS
        self.holding_period = holding_period
        self.max_positions = max_positions
        self.equal_weight = equal_weight
        self.min_conviction = min_conviction
        self.model_params = {"weights": dict(weights)}

    def calc(
        self,
        factor_signals: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        weights = self.model_params["weights"]
        return sum(weights[fid] * factor_signals[fid] for fid in self.factor_ids)
