"""Markets often reprice more sharply when trading activity expands faster than the intrabar price range, because new information is being absorbed without immediately exhausting liquidity. When turnover rises but the bar closes near its open, that often reflects latent order-flow imbalance and a fragile equilibrium that can resolve in the next bar; when turnover rises with a large directional close, it more often indicates confirmed participation and continuation. This is in the spirit of the paper’s hierarchical change view: volume can be an early sign of a deeper regime change in the price formation process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, delta, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquidityRegimeShift(BaseFactor):
    """Volume-expansion and directional-efficiency regime shift signal."""

    factor_id = "liquidity_regime_shift"
    name = "Liquidity Regime Shift"
    category = "other"
    description = (
        "Cross-sectional score from recent volume acceleration multiplied by "
        "normalized directional efficiency: high when volume surges with clean "
        "directional bars, low when surges are absorbed into choppy bars."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        vol_recent = adv(volume, 20)
        vol_change = delta(volume, 5)
        vol_accel = rank(vol_change / vol_recent.replace(0.0, np.nan))

        bar_range = (high - low).replace(0.0, np.nan)
        directional_efficiency = (close - open_) / bar_range
        directional_efficiency = directional_efficiency.clip(lower=-3.0, upper=3.0)
        directional_efficiency = directional_efficiency.fillna(0.0)

        surge = (volume / vol_recent.replace(0.0, np.nan)) - 1.0
        surge = surge.clip(lower=-3.0, upper=5.0).fillna(0.0)

        clean_direction = rank(abs_(close - open_) / bar_range)
        clean_direction = (clean_direction * 2.0) - 1.0

        raw = (surge * directional_efficiency) + (0.25 * surge * clean_direction)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return (0.6 * vol_accel + 0.4 * rank(raw)).astype(float)
