"""Markets often trend most reliably when a bar closes near its extreme after a broad, clean excursion; when the path is choppy and indecisive, continuation is much less likely. Inspired by the paper's emphasis on selective memory and the importance of a few large increments in non-Gaussian rough dynamics, this factor rewards directional drift that is supported by a low-complexity intrabar path. It should complement existing range- and volatility-based signals by separating efficient directional moves from noisy, mean-reverting ones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class EntropyWeightedRangeDrift(BaseFactor):
    """Entropy-weighted directional drift from open to close."""

    factor_id = "entropy_weighted_range_drift"
    name = "Entropy-Weighted Range Drift"
    category = "other"
    description = (
        "Compute each bar's signed return from open to close, then weight it "
        "by the inverse of intrabar path entropy: a combination of "
        "normalized bar range, body dominance, and how often the close is "
        "near the high/low. Higher values indicate a strong directional move "
        "made with low intrabar disorder; lower values indicate noisy "
        "movement despite large range."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).abs()
        body = (close - open_).abs()
        signed_drift = (close - open_) / open_.replace(0, np.nan)

        range_ratio = bar_range / close.abs().replace(0, np.nan)
        body_ratio = body / bar_range.replace(0, np.nan)

        upper_wick = (high - close).abs()
        lower_wick = (close - low).abs()
        close_near_extreme = 1.0 - ((upper_wick + lower_wick) / (bar_range.replace(0, np.nan) * 2.0))

        # Low-complexity path proxy: reward large body, near-extreme closes, and compressed wicks.
        complexity = (
            0.45 * rank(range_ratio)
            + 0.35 * rank(1.0 - body_ratio)
            + 0.20 * rank(1.0 - close_near_extreme.fillna(0.0))
        )

        # Inverse entropy-like weight: stronger when the bar is efficient and directional.
        entropy_weight = 1.0 - complexity
        entropy_weight = entropy_weight.clip(lower=0.0, upper=1.0)

        # Slightly reinforce moves aligned with the market's immediate drift using close-to-close returns.
        drift_alignment = rank(returns(data).fillna(0.0))
        weight = (0.7 * entropy_weight) + (0.3 * drift_alignment)

        signal = signed_drift * weight
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
