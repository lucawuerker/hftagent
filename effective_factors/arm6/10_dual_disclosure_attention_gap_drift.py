from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _fresh_report(actual: pd.DataFrame) -> pd.DataFrame:
    previous = actual.shift(1)
    changed = actual.ne(previous) | previous.isna()
    return actual.notna() & changed


@register_factor
class DualDisclosureAttentionGapDrift(BaseFactor):
    factor_id = "dual_disclosure_attention_gap_drift"
    name = "Dual Disclosure Attention Gap Drift"
    category = "sentiment"
    description = (
        "Ranks fresh reports with concordant EPS and revenue surprises by their "
        "fundamental-to-price reaction gap, emphasizing disclosures that received "
        "unusually little trading attention."
    )
    window_length = 60
    inputs = [
        "close", "volume", "epsActual", "revenueActual",
        "epsSurprise", "revenueSurprise"
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0.0)
        volume = data["volume"].clip(lower=0.0)
        eps_surprise = data["epsSurprise"]
        revenue_surprise = data["revenueSurprise"]

        report = _fresh_report(data["epsActual"]) | _fresh_report(data["revenueActual"])
        eps_signal = 2.0 * rank(eps_surprise) - 1.0
        revenue_signal = 2.0 * rank(revenue_surprise) - 1.0
        valid = eps_signal.notna() & revenue_signal.notna()
        concordant = valid & (eps_signal * revenue_signal > 0.0)
        fundamental_direction = (eps_signal + revenue_signal) / 2.0

        reaction = close.pct_change(2, fill_method=None)
        reaction_signal = 2.0 * rank(reaction) - 1.0
        reaction_gap = (1.0 - fundamental_direction * reaction_signal).clip(0.0, 2.0)

        volume_base = volume.rolling(60, min_periods=20).mean().replace(0.0, np.nan)
        abnormal_turnover = volume.rolling(3, min_periods=3).mean() / volume_base
        low_attention = 1.0 - rank(abnormal_turnover)

        score = fundamental_direction * reaction_gap * low_attention
        score = score.where(report & concordant, 0.0)
        return rank(score).replace([np.inf, -np.inf], np.nan).fillna(0.5)
