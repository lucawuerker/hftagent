"""When a bar closes near its high or low on elevated volume, part of the move is often an information-driven impulse rather than pure noise. But if the intrabar path is already “spent” — large range relative to body and limited follow-through from open to close — continuation is weaker because the move has exhausted local liquidity and becomes vulnerable to short-horizon reversal. This factor tries to separate true directional conviction from high-energy but inefficient price excursions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, rank, sign, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class SignedCloseRangeEnergyDecay(BaseFactor):
    """Signed close-range energy decay momentum factor."""

    factor_id = "signed_close_range_energy_decay"
    name = "Signed Close-Range Energy Decay"
    category = "momentum"
    description = (
        "Computes a signed strength score from close-to-open direction, the "
        "close location within the bar range, and a penalty for inefficient "
        "range usage. Higher values indicate bullish bars with strong "
        "settlement near the high and efficient directional travel; lower "
        "values indicate the opposite."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).replace(0, np.nan)
        body = close - open_
        direction = sign(body)

        # Close location in range: near 1 at the high, near 0 at the low.
        close_location = ((close - low) / bar_range).clip(lower=0.0, upper=1.0)
        close_location = close_location.fillna(0.5)

        # Efficiency proxy: body relative to total range.
        body_efficiency = (abs_(body) / bar_range).clip(lower=0.0, upper=1.0)
        body_efficiency = body_efficiency.fillna(0.0)

        # Penalize large-range, weak-body bars by downweighting low-efficiency moves.
        inefficiency_penalty = 1.0 - body_efficiency

        # Volume confirmation relative to recent average volume.
        vol_ratio = volume / adv(volume, 20)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_score = rank(vol_ratio)

        # Persistence of directional settlement over a short horizon.
        settle_persistence = ts_mean(close_location - 0.5, 5)
        settle_persistence = settle_persistence.fillna(0.0)

        raw = direction * (close_location - 0.5)
        raw = raw * (1.0 + vol_score)
        raw = raw * (1.0 - 0.5 * inefficiency_penalty)
        raw = raw + 0.5 * settle_persistence

        signal = raw / (1.0 + abs_(raw))
        return signal.replace([np.inf, -np.inf], 0.0).fillna(0.0)
