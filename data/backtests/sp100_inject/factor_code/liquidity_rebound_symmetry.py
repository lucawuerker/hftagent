"""When a bar shows a large intrabar excursion away from the open but the close returns near the open, it often reflects temporary liquidity imbalance rather than durable information. Markets with strong mean-reverting order-flow absorption tend to snap back after such "false moves," especially when the bar's path is efficient but the terminal location is neutral. This is consistent with contraction-style robustness intuition from the source paper: disturbances may perturb the state, but systems with strong restorative dynamics contract back toward equilibrium."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquidityReboundSymmetry(BaseFactor):
    """Liquidity rebound signal from intrabar excursion, close reversion, and volume confirmation."""

    factor_id = "liquidity_rebound_symmetry"
    name = "Liquidity Rebound Symmetry"
    category = "microstructure"
    description = (
        "Measure how far price traveled intrabar versus how much of that move was recovered by the close, "
        "normalized by range. The factor is strongest when high/low range is large, close is near open, "
        "and volume does not confirm a directional break."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        range_ = (high - low).replace(0, np.nan)
        body = close - open_

        # Intrabar excursion relative to open, signed by the direction of the initial move away from open.
        up_excursion = (high - open_).clip(lower=0.0)
        down_excursion = (open_ - low).clip(lower=0.0)
        excursion = up_excursion + down_excursion

        # How much of the intrabar excursion was recovered by the close, centered so that full recovery is positive.
        recovery = 1.0 - (body.abs() / range_)

        # Symmetry favors bars where the path was wide but terminal location is near the open.
        path_efficiency = excursion / range_
        neutral_close = 1.0 - (body.abs() / range_)

        # Directional false-move component: if the bar swept away from open and reverted, this is positive.
        false_move = ((high - close).abs() + (close - low).abs()) / range_
        false_move = 1.0 - false_move

        # Volume confirmation dampener: stronger volume reduces the signal when it looks like a confirmed breakout.
        vol_avg = adv(volume, 20)
        vol_ratio = volume / vol_avg.replace(0, np.nan)
        vol_dampen = 1.0 / (1.0 + vol_ratio.fillna(0.0))

        # Recent contraction: bars with unusually large range versus their own recent average are more interesting.
        recent_range = ts_mean(range_, 5)
        expansion = range_ / recent_range.replace(0, np.nan)
        expansion = expansion.fillna(0.0)

        core = 0.4 * path_efficiency + 0.4 * neutral_close + 0.2 * false_move
        signal = core * recovery * vol_dampen * expansion

        # Cross-sectional emphasis on relative strength of the rebound behavior.
        signal = rank(signal)

        # Preserve the original shape and keep output numeric.
        return signal.fillna(0.0)
