"""Research indicates that significant price jumps are often followed by reversals, particularly in high-frequency trading environments where market participants overreact to information (Scaillet et al., 2020). By monitoring the volume accompanying these jumps, we can identify potential exhaustion points where prices are likely to revert to mean levels after extreme movements. This aligns with behavioral finance principles, suggesting that traders may overextend in response to sudden price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, abs_, sign, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class JumpVolumeReversal(BaseFactor):
    factor_id = "jump_volume_reversal"
    name = "Jump Volume Reversal Signal"
    category = "mean_reversion"
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        adv5 = adv(volume, 5)
        jump = delta(close, 1).where(delta(close, 1) > 0).fillna(0.0)
        jump_volume = volume.where(jump > 0, 0.0)
        jump_volume_ratio = jump_volume / adv5.replace(0, np.nan)
        return jump_volume_ratio