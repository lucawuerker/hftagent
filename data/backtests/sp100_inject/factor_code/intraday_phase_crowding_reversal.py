"""When an intraday move runs far and fast, it often reflects temporary crowding by liquidity takers rather than durable information, especially if the bar closes back near the open after a large excursion. This factor targets that "burst then quiescence" structure by fading names whose current bar has a strong directional impulse but weak close commitment relative to the full high-low range. The intuition is analogous to the intermittent low-variance events in the NLSA paper: transient bursts can look important locally, but often do not persist once the flow imbalance is exhausted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayPhaseCrowdingReversal(BaseFactor):
    """Fade transient intraday crowding when impulse is large but close commitment is weak."""

    factor_id = "intraday_phase_crowding_reversal"
    name = "Intraday Phase Crowding Reversal"
    category = "mean_reversion"
    description = (
        "Compute a cross-sectional reversal score from the gap between directional intrabar impulse "
        "and close commitment, then apply a short rolling normalization so the factor is comparable "
        "across tickers and time."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        bar_range = (high - low).abs().replace(0.0, np.nan)
        oc_move = close - open_

        # Directional intrabar impulse scaled by realized range traversal.
        impulse = oc_move / (bar_range + eps)

        # Close commitment relative to the full bar: near 1 when closing near the bar extreme,
        # near 0 when the close sits near the open / mid despite large excursion.
        close_position = (close - low) / (bar_range + eps)
        commitment = (close_position - 0.5) * 2.0

        # Penalty for bars that move far but fail to hold that move into the close.
        reversal_raw = -(impulse * (1.0 - abs_(commitment)))

        # Cross-sectional exposure and short rolling normalization.
        cs_signal = rank(reversal_raw)
        local_mean = ts_mean(cs_signal, 5)
        local_disp = ts_rank(abs_(reversal_raw), 5)

        signal = (cs_signal - 0.5) - 0.5 * local_mean
        signal = signal * (1.0 + local_disp)

        return signal.where(np.isfinite(signal), 0.0)
