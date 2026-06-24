"""Markets often overreact when short-horizon realized volatility shifts abruptly: a sudden expansion after a quiet regime can trigger crowding and temporary price dislocation, while volatility compression after turbulence can leave prices overshooting relative to near-term fair value. Inspired by the paper’s emphasis on denoising and time-frequency structure, this factor looks for regime changes in intrabar volatility and fades the move when the expansion looks climactic rather than trend-confirming. It should complement trend and breakout signals by isolating cases where the market’s recent path has become noisy enough that continuation is less reliable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, correlation, delta, rank, returns, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolatilityRegimeMispricingReversal(BaseFactor):
    """Cross-sectional reversal signal driven by abrupt volatility regime shifts."""

    factor_id = "volatility_regime_mispricing_reversal"
    name = "Volatility Regime Mispricing Reversal"
    category = "volatility"
    description = (
        "Detect abrupt changes in short-horizon realized volatility relative to a slower baseline "
        "and fade the move when volatility expansion or compression appears regime-driven rather than trend-confirming."
    )
    window_length = 60
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        close_ret = returns(data).fillna(0.0)
        intrabar_range = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        short_close_vol = ts_mean(abs_(close_ret), 5)
        long_close_vol = ts_mean(abs_(close_ret), 20)
        short_range_vol = ts_mean(intrabar_range, 5)
        long_range_vol = ts_mean(intrabar_range, 20)

        vol_regime_shift = (short_close_vol - long_close_vol) + 0.5 * (short_range_vol - long_range_vol)
        vol_regime_accel = delta(vol_regime_shift, 3)

        recent_price_move = delta(close, 3)
        move_confirm = correlation(recent_price_move, vol_regime_shift, 10)

        climactic_component = -rank(vol_regime_accel) * rank(abs_(recent_price_move))
        confirm_penalty = rank(move_confirm) - 0.5
        raw_signal = climactic_component - confirm_penalty

        quiet_regime = rank(long_close_vol + long_range_vol)
        compression_tilt = 0.5 - quiet_regime
        compression_adjustment = ts_rank(vol_regime_shift, 10) - 0.5

        signal = raw_signal + 0.5 * compression_tilt * compression_adjustment
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
