"""Terminal execution-flow extinction and inventory-release reversal.

Mechanism: elevated, regular directional participation followed by a sharp decay in
that participation is evidence that a metaorder has completed rather than that its
information is still arriving.  When the terminal bar remains range-compressed,
passive liquidity has likely absorbed the program; the removal of one-sided demand
and subsequent dealer inventory redistribution should reverse returns over several
daily bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns
from quant_fund_agent.factors.registry import register_factor


def _flow_extinction_reversal(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    ret: pd.DataFrame,
) -> pd.DataFrame:
    """Causal terminal-flow score based solely on current and trailing bars.

    A lagged directional intensity describes the recently active program.  Its
    decline relative to that lagged state is an extinction clock, while compressed
    range conditions identify inventory absorption rather than a price-discovery
    breakout.
    """
    safe_volume = volume.fillna(0.0).clip(lower=0.0)
    safe_close = close.replace(0.0, np.nan)
    raw_range = (high - low).abs().replace(0.0, np.nan)

    location = ((2.0 * close - high - low) / raw_range).clip(-1.0, 1.0)
    location = location.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volume_base = safe_volume.ewm(
        halflife=20.0, adjust=False, min_periods=12
    ).mean()
    relative_volume = (
        safe_volume / volume_base.replace(0.0, np.nan)
    ).clip(0.0, 8.0)
    marked_flow = location * np.sqrt(relative_volume).fillna(0.0)
    absolute_flow = marked_flow.abs()

    positive = marked_flow.clip(lower=0.0).ewm(
        halflife=3.0, adjust=False, min_periods=8
    ).mean()
    negative = (-marked_flow.clip(upper=0.0)).ewm(
        halflife=3.0, adjust=False, min_periods=8
    ).mean()
    total_intensity = positive + negative
    signed_intensity = positive - negative

    prior_total = total_intensity.shift(3)
    prior_direction = signed_intensity.shift(3) / prior_total.replace(0.0, np.nan)
    slow_activity = absolute_flow.ewm(
        halflife=20.0, adjust=False, min_periods=12
    ).mean()
    prior_elevation = (
        prior_total / slow_activity.shift(3).replace(0.0, np.nan) - 1.0
    ).clip(0.0, 2.0) / 2.0
    extinction = (1.0 - total_intensity / prior_total.replace(0.0, np.nan)).clip(0.0, 1.0)

    regularity_mean = absolute_flow.rolling(10, min_periods=8).mean()
    regularity_rms = np.sqrt(
        (absolute_flow * absolute_flow).rolling(10, min_periods=8).mean()
    )
    regularity = (regularity_mean / regularity_rms.replace(0.0, np.nan)).clip(0.0, 1.0)

    range_fraction = (raw_range / safe_close).replace([np.inf, -np.inf], np.nan)
    normal_range = range_fraction.ewm(
        halflife=15.0, adjust=False, min_periods=10
    ).mean()
    pinning = (1.0 - range_fraction / normal_range.replace(0.0, np.nan)).clip(0.0, 1.0)

    valid = safe_volume.rolling(20, min_periods=12).count() >= 12
    score = -prior_direction * prior_elevation * extinction * regularity * pinning
    return score.where(valid).replace([np.inf, -np.inf], np.nan)


@register_factor
class LatentFlowExtinctionInventoryReversal(BaseFactor):
    factor_id = "latent_flow_extinction_inventory_reversal"
    name = "Flow Extinction Inventory Reversal"
    category = "microstructure"
    description = (
        "Ranks reversals after an elevated regular signed-flow program decays while "
        "the current daily range remains pinned, consistent with absorbed execution "
        "and subsequent inventory release."
    )
    window_length = 45
    inputs = ["close", "high", "low", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].replace(0.0, np.nan)
        high = data["high"].replace(0.0, np.nan)
        low = data["low"].replace(0.0, np.nan)
        volume = data["volume"].fillna(0.0).clip(lower=0.0)
        ret = returns(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        score = _flow_extinction_reversal(close, high, low, volume, ret)
        return rank(score).fillna(0.5).reindex(index=close.index, columns=close.columns)
