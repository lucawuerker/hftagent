"""A bar that travels a large distance from open to close while also printing a very wide high-low range often reflects a noisy, unstable intrabar path rather than clean directional information. In such cases, the move is more likely to be partially mean-reverted as liquidity providers fade transient price excursions and execution imbalances unwind. This is conceptually aligned with the paper’s emphasis on extracting robust features from non-stationary, misaligned signals: here we ignore simple endpoint moves and focus on the path structure that distinguishes persistent movement from jittery overshoot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarPathIrregularity(BaseFactor):
    """Intrabar irregularity from OHLC path geometry; higher is more contrarian."""

    factor_id = "intrabar_path_irregularity"
    name = "Intrabar Path Irregularity"
    category = "microstructure"
    description = (
        "Measures how wide the intrabar high-low excursion is relative to the "
        "net open-close displacement, with an added body-to-range penalty. "
        "Higher values indicate a noisier, more unstable path."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-9
        true_range = (high - low).abs()
        body = (close - open_).abs()

        irregularity = true_range / (body + eps)
        body_share = body / (true_range + eps)

        score = irregularity * (1.0 - body_share)
        score = score.replace([np.inf, -np.inf], np.nan)
        return score.fillna(0.0)
