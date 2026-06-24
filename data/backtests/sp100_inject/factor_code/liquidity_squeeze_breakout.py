"""When price compresses into a narrow intrabar range but volume remains elevated, the market is often building pressure rather than reverting. The Malyshkin/Bakhramov paper emphasizes that spikes in execution flow carry more information than stationary statistics; here, the combination of tight range and strong flow suggests a latent imbalance that is likely to resolve with a directional break. This should complement mean-reversion seeds by favoring continuation only when the market has already absorbed substantial activity without much price progress."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean, returns
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquiditySqueezeBreakout(BaseFactor):
    """Breakout pressure from high volume and tight intrabar compression."""

    factor_id = "liquidity_squeeze_breakout"
    name = "Liquidity Squeeze Breakout"
    category = "microstructure"
    description = (
        "Compute a breakout bias from high recent volume normalized by small "
        "intrabar range, then sign it by the candle direction relative to the "
        "bar open/close. A positive value indicates upward breakout pressure "
        "from a high-activity compression regime; a negative value indicates "
        "downward pressure."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        intrabar_range = (high - low).abs()
        price_level = close.abs().where(close.abs() > 0.0, np.nan)
        compression = 1.0 - (intrabar_range / price_level)
        compression = compression.clip(lower=0.0, upper=1.0).fillna(0.0)

        vol20 = adv(volume, 20)
        vol_rank = rank(vol20)
        vol_pressure = ts_mean(returns({"close": close}), 3).fillna(0.0)

        squeeze_intensity = compression * vol_rank.fillna(0.0)
        candle_direction = np.sign(close - open_)
        candle_direction = pd.DataFrame(candle_direction, index=close.index, columns=close.columns).fillna(0.0)

        breakout_bias = squeeze_intensity * (1.0 + vol_pressure.abs())
        signal = breakout_bias * candle_direction

        return signal.reindex(index=close.index, columns=close.columns).fillna(0.0)
