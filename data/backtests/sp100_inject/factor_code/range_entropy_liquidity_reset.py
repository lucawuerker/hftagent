"""When intraday price action is chaotic but the bar closes back near the open on heavy volume, it often signals that liquidity providers absorbed a transient imbalance and forced a reset in the short-horizon order flow. This effect is especially relevant in systematic portfolios because violent but mean-preserving bars tend to be followed by continuation only if the closing location confirms direction; otherwise the market often reverts as the temporary pressure fades. The idea complements the EPS-style intuition in the paper: shocks that are not “effective” in the sense of changing settled value should be treated as noise rather than trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, rank, returns, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeEntropyLiquidityReset(BaseFactor):
    """Noisy high-volume bars that close back near the open are mean-reverting."""

    factor_id = "range_entropy_liquidity_reset"
    name = "Range Entropy Liquidity Reset"
    category = "other"
    description = (
        "Compute the absolute body-to-range ratio times a normalized volume "
        "surprise term, then convert it into a signed score using the close "
        "location within the bar. High values indicate a noisy, high-"
        "participation bar that settled away from the extremes and is "
        "therefore more likely to mean-revert; low values indicate orderly "
        "trend-confirming bars."
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
        body = abs_(close - open_)
        body_to_range = (body / bar_range).fillna(0.0)

        vol_avg = adv(volume.fillna(0.0), 20)
        vol_surprise = (volume.fillna(0.0) / vol_avg.replace(0, np.nan) - 1.0).fillna(0.0)
        vol_surprise = vol_surprise.clip(lower=0.0)

        close_loc = ((close - low) / bar_range).fillna(0.5)
        close_centered = (close_loc - 0.5) * 2.0
        reset_score = 1.0 - abs_(close_centered)

        noisy_participation = body_to_range * (1.0 + vol_surprise)
        raw = noisy_participation * reset_score

        short_return = returns(data).fillna(0.0)
        drift = ts_mean(short_return, 5).fillna(0.0)
        direction = np.sign((close - open_).fillna(0.0) + drift)

        signal = raw * direction
        return signal.reindex(index=close.index, columns=close.columns)
