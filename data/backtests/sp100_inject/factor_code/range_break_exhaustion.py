"""A bar that closes near its extreme after already traversing most of its own range often reflects temporary order-flow imbalance rather than durable information. If the move is large relative to the bar's realized range but the close is still not an efficient continuation, the next bar is prone to partial reversal as liquidity replenishes and aggressive traders exhaust themselves. This complements jump/liquidity signals from the cited paper by focusing on post-impulse mean reversion using only OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, rank, returns


@register_factor
class RangeBreakExhaustion(BaseFactor):
    """Range-based exhaustion signal that favors mean reversion after stretched closes."""

    factor_id = "range_break_exhaustion"
    name = "Range Break Exhaustion"
    category = "mean_reversion"
    description = (
        "Measures how strongly the close finishes at one end of the bar after "
        "normalizing by the full high-low range, then penalizes large volume "
        "when the move is stretched. Positive values indicate overextended "
        "up-bars likely to mean-revert; negative values indicate overextended "
        "down-bars likely to bounce."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).replace(0, np.nan)
        close_loc = ((close - low) / bar_range).clip(0.0, 1.0)
        open_loc = ((open_ - low) / bar_range).clip(0.0, 1.0)

        # Signed stretch: positive for closes near the high, negative near the low.
        location_edge = (close_loc - 0.5) * 2.0
        candle_body = close - open_
        body_ratio = (candle_body.abs() / bar_range).replace([np.inf, -np.inf], np.nan)

        # Large range expansion with small body tends to indicate exhaustion.
        stretch = (bar_range / close.abs().replace(0, np.nan)).fillna(0.0)
        inefficiency = (1.0 - body_ratio.fillna(0.0)).clip(0.0, 1.0)

        vol_rank = rank(adv(volume, 20))
        vol_penalty = 1.0 - vol_rank.fillna(0.5)

        # Up bars that close at the high are expected to mean revert downward,
        # while down bars that close at the low are expected to bounce upward.
        direction = np.sign(location_edge * candle_body.fillna(0.0))
        direction = pd.DataFrame(direction, index=close.index, columns=close.columns)

        raw = -location_edge * np.abs(direction) * (0.5 + inefficiency) * (0.5 + stretch)
        adjusted = raw * (0.5 + vol_penalty)

        # Incorporate short-horizon return context to emphasize impulse-and-fade bars.
        ret1 = returns(data).fillna(0.0)
        impulse_bias = (-ret1).where(location_edge.abs() > 0.5, 0.0)

        signal = adjusted + 0.25 * impulse_bias
        signal = signal.where(bar_range.notna(), 0.0)
        return signal.astype(float)
