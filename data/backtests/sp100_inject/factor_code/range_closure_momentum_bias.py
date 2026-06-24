"""Assets that consistently close near the top of their intrabar range after opening weak tend to exhibit continuation, because the tape shows persistent demand overcoming early supply. This is a short-horizon behavioral signal: traders who are forced to chase strength late in the bar can create follow-through, especially when the bar also expands on healthy volume. The idea is complementary to pure gap or close-location signals because it focuses on the *trajectory* from open to close rather than just the endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeClosureMomentumBias(BaseFactor):
    """Ranks bars by close-in-range momentum with a range-expansion emphasis."""

    factor_id = "range_closure_momentum_bias"
    name = "Range Closure Momentum Bias"
    category = "momentum"
    description = (
        "Measure the distance of close from open relative to the full high-low "
        "range, then multiply by a range-expansion term to emphasize strong "
        "directional bars. Cross-sectionally rank the resulting score each bar "
        "so names with strong upward closure within wide ranges rank highest."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0, np.nan)
        close_from_open = (close - open_)
        closure_ratio = close_from_open / bar_range

        range_mid = (high + low) * 0.5
        range_expansion = (bar_range / range_mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        raw_score = closure_ratio * range_expansion
        raw_score = raw_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return rank(raw_score)
