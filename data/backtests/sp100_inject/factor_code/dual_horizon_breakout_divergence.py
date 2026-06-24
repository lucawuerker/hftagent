"""Markets often move differently at short and long horizons: intraday breakout pressure can be genuine continuation when it is confirmed by a slower drift, but it is more likely to fade when the move is only a high-frequency burst. Inspired by the paper’s time-scale-dependent correlations, this factor looks for disagreement between fast and slow price components; such cross-scale divergence is a natural setup for temporary mispricing and reversion once the fast impulse exhausts. The signal is most useful when the short-horizon move is strong but the slower trend remains opposed or weak."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, returns, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class DualHorizonBreakoutDivergence(BaseFactor):
    """Fast breakout versus slow trend divergence in close-to-close returns."""

    factor_id = "dual_horizon_breakout_divergence"
    name = "Dual-Horizon Breakout Divergence"
    category = "statistical_arbitrage"
    description = (
        "Compute a fast return momentum term and a slow return momentum term "
        "from close-to-close returns using different rolling windows, then take "
        "their difference after cross-sectional normalization. Positive values "
        "indicate a short-term breakout that is not yet supported by the slower "
        "horizon; negative values indicate the opposite."
    )
    window_length = 40
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]

        ret = returns(data)
        fast = ts_mean(ret, 5)
        slow = ts_mean(ret, 20)

        divergence = fast - slow
        divergence = divergence.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        fast_strength = rank(delta(close, 3).abs())
        slow_strength = rank(delta(close, 15).abs())
        raw_signal = divergence * (fast_strength - slow_strength)

        signal = rank(raw_signal)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
