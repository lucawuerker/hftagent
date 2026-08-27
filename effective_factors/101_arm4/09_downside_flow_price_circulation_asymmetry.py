"""This factor treats signed volume-pressure and returns as a two-dimensional dynamical system: a clockwise downside loop in which negative volume pressure leads subsequent negative returns is interpreted as unresolved forced-sale or information-motivated selling, rather than a completed price discovery event. Such circulation should predict further relative underperformance because liquidity providers and slower fundamental buyers only gradually absorb a sequence of correlated sell programs; the inverse pattern, where price declines lead the volume response, is more consistent with an already-discovered shock and should mean-revert. The hypothesis is falsified if the lag orientation of the loop has no incremental ability to separate continuation from reversal after controlling for recent return, range, and volume level; it can persist because daily-bar investors generally observe the move and volume but do not explicitly estimate directional phase lead-lag or trade the short-lived, capacity-constrained residual."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _downside_circulation(
    flow: pd.DataFrame,
    negative_log_return: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Causal trailing antisymmetric area of the downside flow-price path."""
    lag_flow = flow.shift(1)
    lag_return = negative_log_return.shift(1)
    increments = lag_flow * negative_log_return - lag_return * flow
    return increments.rolling(window, min_periods=window).sum()


@register_factor
class DownsideFlowPriceCirculationAsymmetry(BaseFactor):
    """Ranks the inverse downside flow-price circulation to identify price-led reactions."""

    factor_id = "downside_flow_price_circulation_asymmetry"
    name = "Downside Flow-Price Circulation Asymmetry"
    category = "microstructure"
    description = (
        "Constructs signed flow pressure from close-location value and standardized "
        "log-volume, then ranks the negative trailing 10-bar antisymmetric area "
        "between downside flow and negative log returns. Volume-led downside loops "
        "receive low scores; price-led downside reactions receive high scores."
    )
    window_length = 30
    inputs = ["high", "low", "close", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0).clip(lower=0.0)

        price_range = (high - low).replace(0.0, np.nan)
        close_location = ((2.0 * close - high - low) / price_range).clip(-1.0, 1.0)
        close_location = close_location.fillna(0.0)

        log_volume = np.log1p(volume)
        volume_mean = log_volume.rolling(20, min_periods=10).mean()
        volume_std = log_volume.rolling(20, min_periods=10).std().replace(0.0, np.nan)
        standardized_log_volume = ((log_volume - volume_mean) / volume_std).clip(-5.0, 5.0)
        standardized_log_volume = standardized_log_volume.fillna(0.0)

        signed_flow = close_location * standardized_log_volume

        safe_close = close.where(close > 0.0, np.nan)
        log_return = np.log(safe_close / safe_close.shift(1)).replace(
            [np.inf, -np.inf], np.nan
        )
        negative_log_return = (-log_return).clip(lower=0.0).fillna(0.0)
        downside_flow = (-signed_flow).clip(lower=0.0)

        circulation = _downside_circulation(downside_flow, negative_log_return, 10)
        return rank((-circulation).replace([np.inf, -np.inf], np.nan))
