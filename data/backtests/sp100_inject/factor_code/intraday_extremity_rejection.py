"""Prices that spend a lot of the session near the bar extremes but still fail to finish there often reflect failed continuation and temporary liquidity-driven overshoots. When the close is far from the intraday high/low yet the full range is large, it suggests the market explored a direction and got rejected, a pattern that is consistent with short-horizon reversal after order-flow imbalance fades. This complements the multi-feature intraday forecasting view in the provided paper by focusing on the end-of-bar rejection component rather than raw directional return history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayExtremityRejection(BaseFactor):
    """Intraday rejection signal from close location within the bar range."""

    factor_id = "intraday_extremity_rejection"
    name = "Intraday Extremity Rejection"
    category = "mean_reversion"
    description = (
        "Scores names whose close is rejected away from the bar extremes after "
        "a wide intraday excursion, emphasizing short-horizon reversal."
    )
    window_length = 1
    inputs = ["high", "low", "close", "open"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        rng = (high - low).replace(0, np.nan)
        body = close - open_

        # Close location within the bar range: near 1 when closing near the high,
        # near 0 when closing near the low.
        close_loc = (close - low) / rng
        close_loc = close_loc.clip(0.0, 1.0)

        # Rejection is strongest when the close finishes away from the explored
        # extreme after a large range. Use a signed score that favors reversals:
        # - bullish rejection: low close location after a wide range -> positive
        # - bearish rejection: high close location after a wide range -> negative
        rejection = (0.5 - close_loc) * 2.0

        # Amplify by the range relative to the open price movement, but keep the
        # transform robust to sparse/zero open values.
        open_abs = open_.abs().replace(0, np.nan)
        range_to_open = (rng / open_abs).clip(lower=0.0)
        range_to_open = range_to_open.fillna(0.0)

        # A small body penalty ensures the signal emphasizes clear rejection rather
        # than persistent trend continuation.
        body_scale = (body.abs() / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        body_penalty = (1.0 - body_scale).clip(0.0, 1.0)

        signal = rejection * range_to_open * body_penalty
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
