"""When price trades through a narrow intrabar range on unusually high volume, it often signals that latent demand/supply has finally absorbed the resting liquidity at a congested level. This is a cleaner continuation setup than raw momentum because it requires both price displacement and participation, reducing false breakouts from thin prints. The graph-based literature in the provided paper emphasizes that local relationships and propagation matter; here we exploit the market's own micro-level accumulation phase before the next directional move."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumePriceCongestionBreakout(BaseFactor):
    """Volume-confirmed breakout from intrabar congestion."""

    factor_id = "volume_price_congestion_breakout"
    name = "Volume-Price Congestion Breakout"
    category = "momentum"
    inputs = ["open", "high", "low", "close", "volume"]

    window_length = 30

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12

        intrabar_range = (high - low).abs()
        range_safe = intrabar_range.replace(0.0, np.nan)
        close_pos = (close - low) / (range_safe + eps)
        mid_pos = (close - (high + low) * 0.5) / (range_safe + eps)

        # Congestion: narrow range versus recent history and limited displacement from open.
        range_ratio = intrabar_range / (ts_mean(intrabar_range, 10) + eps)
        congestion_strength = (1.0 - rank(range_ratio)).clip(lower=0.0, upper=1.0)

        body = (close - open_).abs()
        body_ratio = body / (range_safe + eps)
        tight_body = (1.0 - rank(body_ratio)).clip(lower=0.0, upper=1.0)

        # Volume confirmation versus recent activity.
        vol_ratio = volume / (adv(volume, 20) + eps)
        volume_confirmation = (rank(vol_ratio) - 0.5) * 2.0
        volume_confirmation = volume_confirmation.clip(lower=-1.0, upper=1.0)

        # Directional breakout pressure: close away from midrange, emphasized when near extremes.
        direction = (close_pos - 0.5) * 2.0
        extremity = (mid_pos.abs() * 2.0).clip(lower=0.0, upper=1.0)
        directional_push = direction * extremity

        # Persistence of the move within the short term.
        short_trend = ts_rank(close - open_, 5) * 2.0 - 1.0

        raw = (
            0.40 * congestion_strength
            + 0.20 * tight_body
            + 0.25 * volume_confirmation
            + 0.10 * directional_push
            + 0.05 * short_trend
        )

        # Require some participation and breakout character; otherwise suppress toward neutral.
        participation_gate = (rank(vol_ratio) > 0.6) & (rank(range_ratio) < 0.6)
        signal = raw.where(participation_gate, 0.0)

        return signal.fillna(0.0)
