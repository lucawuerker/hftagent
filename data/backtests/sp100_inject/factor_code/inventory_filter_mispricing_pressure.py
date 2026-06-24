"""The paper’s key insight is that agents infer a hidden demand regime from censored flow, updating beliefs faster when observations are informative and slower when they are ambiguous. In markets, a bar that pushes far through its intraday range but closes near the opposite extreme often signals that the initial move was driven by a transient regime shock or liquidity vacuum rather than persistent informed demand. Combining the size of the reversal with the bar’s expansion relative to recent ranges gives a cross-sectional proxy for “mispriced belief update,” which should be more likely to mean-revert after overreactive moves and continue after genuinely informative ones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, delay, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class InventoryFilterMispricingPressure(BaseFactor):
    """Mispricing pressure from intrabar reversal versus recent range expansion."""

    factor_id = "inventory_filter_mispricing_pressure"
    name = "Inventory Filter Mispricing Pressure"
    category = "other"
    description = (
        "Measures how strongly the bar’s close contradicts its intrabar excursion, "
        "scaled by recent range expansion. High positive values indicate an efficient "
        "close after expansion; low/negative values indicate an overextended move that "
        "was not confirmed into the close."
    )
    inputs = ["high", "low", "close", "volume"]
    window_length = 20

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        prev_close = delay(close, 1)
        bar_range = (high - low).abs()
        range_floor = prev_close.abs().where(prev_close.abs() > 1e-12, 1.0)
        normalized_range = bar_range / range_floor

        open_close_pos = (close - low) / (bar_range.replace(0, np.nan))
        open_close_pos = open_close_pos.fillna(0.5)

        close_contradiction = 1.0 - 2.0 * (open_close_pos - 0.5).abs()
        close_contradiction = close_contradiction * np.sign((close - prev_close).fillna(0.0) + 1e-12)

        recent_range = ts_mean(normalized_range, 5)
        expanded = normalized_range / (recent_range.replace(0, np.nan))
        expanded = expanded.fillna(0.0)

        volume_pressure = rank(volume / (adv(volume, 20).replace(0, np.nan))).fillna(0.5)
        volume_pressure = (volume_pressure - 0.5) * 2.0

        raw = close_contradiction * expanded
        raw = raw * (1.0 + 0.5 * volume_pressure)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        signal = rank(raw) - 0.5
        return signal * 2.0
