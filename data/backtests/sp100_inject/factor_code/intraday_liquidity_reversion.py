"""Intraday Liquidity Reversion: bet against large stressed intraday moves that are reinforced by unusually high volume, as temporary liquidity pressure often mean reverts once flow imbalance fades."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, log, rank, returns, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayLiquidityReversion(BaseFactor):
    """Cross-sectional mean-reversion signal from stressed intraday price moves."""

    factor_id = "intraday_liquidity_reversion"
    name = "Intraday Liquidity Reversion"
    category = "mean_reversion"
    description = (
        "Compute a cross-sectional score that is high when the close-to-close "
        "move is extreme relative to the bar’s true range and when volume is "
        "elevated versus a recent baseline. The signal is the negative product "
        "of standardized return and standardized volume pressure, so names with "
        "large upward stressed moves are short and large downward stressed moves "
        "are long."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        prev_close = close.shift(1)
        close_move = close - prev_close

        true_range = (high - low).abs()
        gap_up = (high - prev_close).abs()
        gap_down = (low - prev_close).abs()
        range_proxy = pd.concat([true_range, gap_up, gap_down], axis=0).groupby(level=0).max()
        range_proxy = range_proxy.replace(0, np.nan)

        bar_move_stress = close_move / range_proxy
        bar_move_stress = bar_move_stress.replace([np.inf, -np.inf], np.nan)

        volume_baseline = adv(volume, 20)
        volume_baseline = volume_baseline.replace(0, np.nan)
        volume_pressure = volume / volume_baseline
        volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan)

        standardized_move = ts_rank(abs_(bar_move_stress), 20) * rank(close_move)
        standardized_volume = ts_rank(volume_pressure, 20)

        raw_signal = -1.0 * standardized_move * standardized_volume
        raw_signal = raw_signal.replace([np.inf, -np.inf], np.nan)

        recent_direction = ts_mean(returns(data), 3)
        recent_direction = recent_direction.fillna(0.0)
        raw_signal = raw_signal.where(recent_direction.notna(), 0.0)

        return raw_signal.fillna(0.0)
