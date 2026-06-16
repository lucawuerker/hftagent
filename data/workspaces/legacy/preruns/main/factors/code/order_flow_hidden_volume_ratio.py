"""This factor leverages the insight that hidden volume is often driven by informed traders seeking to minimize market impact. By comparing the ratio of order flow to hidden volume, we can identify moments where the market may be underestimating the true demand or supply pressures due to the presence of hidden orders, potentially leading to price adjustments in the near future."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, ts_sum, ts_mean, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowHiddenVolumeRatio(BaseFactor):
    factor_id = "order_flow_hidden_volume_ratio"
    name = "Order Flow to Hidden Volume Ratio"
    category = "microstructure"
    inputs = ["orderFlow", "hidden"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        hidden_volume = data["hidden"].replace(0, np.nan)

        # Calculate the ratio of absolute signed order flow to hidden volume
        ratio = abs_(order_flow) / hidden_volume

        # Handle NaN values in the ratio
        ratio = ratio.fillna(0.0)

        return ratio