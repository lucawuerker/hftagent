"""Zawadowski et al. document that liquid stocks tend to reverse after unusually large intraday moves, with the reversal becoming stronger when the shock is larger. A natural cross-sectional way to exploit that effect is to identify names that have just experienced an outsized directional move relative to their recent intraday range and fade that move over the next bar(s). This should complement a seed ensemble by focusing on short-horizon behavioral overreaction rather than slower trend or valuation effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, correlation, delta, rank, returns, stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayShockReversalPressure(BaseFactor):
    """Fade outsized intraday shocks by scaling latest returns by recent range/volatility."""

    factor_id = "intraday_shock_reversal_pressure"
    name = "Intraday Shock Reversal Pressure"
    category = "mean_reversion"
    description = (
        "Compute the signed distance of the latest close-to-close move from a rolling "
        "intraday volatility scale, then multiply by -1 so extreme up-moves score bearish "
        "and extreme down-moves score bullish."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = returns(data)

        # Recent intraday shock scale: average high-low range and realized close-to-close vol.
        hl_range = (high - low).abs()
        range_scale = ts_mean(hl_range, 10)
        vol_scale = stddev(ret, 10)

        # Normalize latest move by a robust local scale and strengthen extreme shocks.
        scale = (range_scale + vol_scale).replace(0, np.nan)
        shock = ret / scale
        shock = shock.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional emphasis on the most extreme movers, with reversal sign.
        magnitude = abs_(shock)
        relative_extremeness = rank(magnitude)
        direction = np.sign(shock)
        pressure = -1.0 * relative_extremeness * direction * magnitude

        # Mild confirmation from recent shock persistence: stronger after abrupt reversal-prone bursts.
        ret_mom = delta(close, 1)
        short_corr = correlation(ret, hl_range, 5)
        confirmation = rank(abs_(short_corr.fillna(0.0)))

        out = pressure * (1.0 + 0.5 * confirmation)
        out = out.where(np.isfinite(out), 0.0)
        out = out.fillna(0.0)
        return out
