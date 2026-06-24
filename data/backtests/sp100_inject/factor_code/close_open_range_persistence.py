"""When a bar closes near one extreme and repeatedly holds that position relative to its own open-high-low range, it often reflects persistent informed order flow rather than a one-off price spike. This is a short-horizon continuation signal: strong directional closes that do not mean-revert intrabar tend to persist into the next bars as the market digests the same imbalance. The hf0 paper is relevant in spirit because it uses a coarse-to-fine decomposition: first identify the dominant regime, then decode the exact signal; here we similarly isolate regime persistence before translating it into a tradable directional score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseOpenRangePersistence(BaseFactor):
    """Persistence of close location within the bar range, weighted by recent range expansion."""

    factor_id = "close_open_range_persistence"
    name = "Close-Open Range Persistence"
    category = "momentum"
    description = (
        "Compute the normalized close location within each bar's range, then smooth it over a short "
        "lookback and multiply by recent range expansion to emphasize persistent directional closes "
        "under active trading. The output is cross-sectional by ticker, with higher values indicating "
        "stronger continuation pressure."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).replace(0.0, np.nan)
        close_loc = (close - low) / bar_range
        close_loc = close_loc.clip(0.0, 1.0)
        directional_pressure = (close_loc - 0.5) * 2.0

        open_close_range = (close - open_).abs() / bar_range
        persistence = ts_mean(directional_pressure, 5)
        persistence_strength = ts_mean(open_close_range, 5)
        recent_range_expansion = bar_range / adv(bar_range.fillna(0.0), 20)
        recent_range_expansion = recent_range_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        activity = ts_rank(volume, 10).fillna(0.5)

        raw = persistence * (0.5 + persistence_strength) * recent_range_expansion * (0.5 + activity)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return raw.reindex(index=close.index, columns=close.columns)
