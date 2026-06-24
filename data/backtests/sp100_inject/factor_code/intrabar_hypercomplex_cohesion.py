"""Prices often trend more reliably when the bar’s directional move is internally coherent rather than built from a wide, noisy excursion. Inspired by the paper’s finding that multi-channel, structure-preserving representations outperform naive real-valued views, this factor treats open, high, low, close, and volume as a single coupled object and rewards bars where directional progress is confirmed by tight range usage and efficient turnover. Cross-sectionally, stocks with the strongest cohesion tend to be those where informed order flow is dominating short-term price discovery rather than being diluted by mean-reverting noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarHypercomplexCohesion(BaseFactor):
    """Intrabar cohesion from directional consistency, range placement, and volume-normalized efficiency."""

    factor_id = "intrabar_hypercomplex_cohesion"
    name = "Intrabar Hypercomplex Cohesion"
    category = "microstructure"
    description = (
        "Compute a bar-level cohesion score from the consistency of open-to-close direction, "
        "close location within the range, and volume-normalized range efficiency. Aggregate it "
        "over a short rolling window and rank stocks cross-sectionally; higher values indicate "
        "more structurally coherent intrabar demand/supply imbalance."
    )
    window_length = 8
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        oc_move = close - open_
        bar_range = (high - low).replace(0.0, np.nan)
        close_pos = ((close - low) / bar_range).clip(lower=0.0, upper=1.0).fillna(0.5)

        direction_consistency = (oc_move / bar_range).clip(lower=-1.0, upper=1.0).fillna(0.0)
        location_alignment = (2.0 * close_pos - 1.0) * np.sign(oc_move.fillna(0.0))
        location_alignment = location_alignment.fillna(0.0)

        range_efficiency = abs_(oc_move) / (bar_range + 1e-12)
        range_efficiency = range_efficiency.clip(lower=0.0, upper=1.0).fillna(0.0)

        vol_norm = volume / (ts_mean(volume.replace(0.0, np.nan), 8) + 1e-12)
        vol_norm = vol_norm.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        turnover_efficiency = 1.0 / (1.0 + np.log1p(vol_norm.clip(lower=0.0)))

        bar_score = (
            0.40 * direction_consistency
            + 0.30 * location_alignment
            + 0.20 * range_efficiency
            + 0.10 * turnover_efficiency
        )

        short_horizon = ts_sum(bar_score, 3) + 0.5 * ts_mean(bar_score, 5)
        short_horizon = short_horizon.fillna(0.0)

        cross_sectional = rank(short_horizon)
        return cross_sectional.where(np.isfinite(cross_sectional), 0.0)
