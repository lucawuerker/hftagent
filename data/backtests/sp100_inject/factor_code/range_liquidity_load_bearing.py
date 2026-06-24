"""When a bar traverses a large fraction of its own range on unusually low volume, price discovery is likely happening through thin liquidity rather than broad conviction. That setup often leaves the next bar vulnerable to continuation if the close lands near an extreme, but prone to stalling if the move was mechanically stretched and volume failed to confirm. The idea borrows from the AMM paper's notion that returns depend jointly on price path shape and available liquidity: a move that consumes a lot of range with little volume is effectively "fee-rich but structurally fragile," which should matter cross-sectionally."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeLiquidityLoadBearing(BaseFactor):
    """Range efficiency scaled by low-volume participation pressure."""

    factor_id = "range_liquidity_load_bearing"
    name = "Range Liquidity Load-Bearing"
    category = "other"
    description = (
        "Measures how efficiently price traverses its intrabar range, then "
        "scales that move by a low-volume penalty relative to trailing "
        "average volume. Stronger when the bar closes near an extreme on "
        "weak participation."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        range_ = (high - low).abs()
        range_safe = range_.replace(0, np.nan)

        # Signed range traversal normalized by bar range.
        close_location = (close - open_) / (range_safe + 1e-9)

        # Encourage closes near either edge, with sign retained by direction.
        edge_pressure = ((2.0 * (close - low) / (range_safe + 1e-9)) - 1.0)
        range_efficiency = close_location * edge_pressure

        vol_trend = ts_mean(volume, 20)
        vol_ratio = volume / (vol_trend + 1e-9)
        low_volume_penalty = 1.0 / (1.0 + vol_ratio)

        # Normalize participation pressure relative to recent average to avoid
        # unstable scaling on sparse panels.
        participation = (adv(volume, 20) + volume) / (vol_trend + 1e-9)
        participation = participation.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        factor = range_efficiency * low_volume_penalty * participation
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return factor
