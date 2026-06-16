"""Understanding lead-lag relationships through mutual information can uncover non-linear dependencies between stocks. By analyzing the mutual information between the order flows of different stocks, we can identify potential predictive relationships where changes in one stock's order flow may lead price movements in another. This is based on the premise that informed trading activity can cluster and exhibit predictable patterns in a non-linear market environment."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, correlation, delay

@register_factor
class MutualInformationBasedFlows(BaseFactor):
    factor_id = "mutual_information_based_flows"
    name = "Mutual Information Based Order Flows"
    category = "microstructure"
    description = "This factor computes the mutual information between the order flows of different stocks over time, highlighting stocks whose order flows are statistically informative about each other."
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        mutual_info = pd.DataFrame(index=order_flow.index, columns=order_flow.columns)

        for stock in order_flow.columns:
            for other_stock in order_flow.columns:
                if stock != other_stock:
                    # Calculate mutual information between order flows
                    joint_distribution = pd.concat([order_flow[stock], order_flow[other_stock]], axis=1).dropna()
                    if joint_distribution.empty:
                        continue
                    # Calculate probabilities
                    p_xy = joint_distribution.value_counts(normalize=True)
                    p_x = order_flow[stock].value_counts(normalize=True)
                    p_y = order_flow[other_stock].value_counts(normalize=True)
                    # Calculate mutual information
                    mi = 0.0
                    for (x, y), p in p_xy.items():
                        if p > 0:
                            mi += p * np.log(p / (p_x[x] * p_y[y]))
                    mutual_info.loc[:, stock] = mutual_info.get(stock, 0) + mi

        return mutual_info.fillna(0.0)