"""Large open-to-close moves that are not supported by a persistent intrabar range often reflect temporary inventory imbalances, opening auction overreaction, or forced execution that later gets faded by liquidity provision. When the bar closes back near the open after a wide excursion, the market has effectively rejected the directional information in the move, which tends to mean-revert over the next bars. This complements momentum-style signals by explicitly betting on failed intraday displacement rather than on continuation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseToOpenGapCompression(BaseFactor):
    """Mean-reversion signal from close finishing back near the open after range expansion."""

    factor_id = "close_to_open_gap_compression"
    name = "Close-to-Open Gap Compression"
    category = "mean_reversion"
    description = (
        "Measure how far the close finishes back toward the open after the bar's total "
        "high-low range has expanded. A strong signal occurs when the bar has a large range "
        "but the close is near the open, especially after a directional excursion toward one extreme."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        eps = 1e-12

        bar_range = (high - low).abs()
        body = (close - open_).abs()
        body_to_range = body / (bar_range.replace(0.0, np.nan) + eps)

        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
        wick_imbalance = (upper_wick - lower_wick).abs() / (bar_range.replace(0.0, np.nan) + eps)

        compression_raw = (1.0 - body_to_range) * (1.0 - wick_imbalance)
        range_expansion = rank(ts_mean(bar_range, 5))
        signal = compression_raw * range_expansion

        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
