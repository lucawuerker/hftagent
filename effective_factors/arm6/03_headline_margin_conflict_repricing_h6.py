"""Headline-margin conflict repricing after earnings releases.

Mechanism: a revenue beat alongside an EPS miss can receive an initially positive
headline interpretation even though it reveals weak incremental profitability.  The
contradiction should be resolved negatively over subsequent sessions, particularly
for expensive growth firms whose valuation requires sustained margin delivery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HeadlineMarginConflictRepricingH6(BaseFactor):
    factor_id = "headline_margin_conflict_repricing_h6"
    name = "Headline-Margin Conflict Repricing"
    category = "sentiment"
    description = (
        "Fresh revenue-beat versus EPS-miss disagreement, neutralised within "
        "industry and gated to relatively expensive sales-multiple firms."
    )
    window_length = 2
    inputs = ["close", "revenueSurprise", "epsSurprise", "psRatio", "industry"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        template = data["close"]
        revenue_surprise = data["revenueSurprise"].replace([np.inf, -np.inf], np.nan)
        eps_surprise = data["epsSurprise"].replace([np.inf, -np.inf], np.nan)
        ps_ratio = data["psRatio"].replace([np.inf, -np.inf], np.nan)

        prior_revenue = revenue_surprise.shift(1)
        prior_eps = eps_surprise.shift(1)
        revenue_update = revenue_surprise.notna() & (
            revenue_surprise.ne(prior_revenue) | prior_revenue.isna()
        )
        eps_update = eps_surprise.notna() & (
            eps_surprise.ne(prior_eps) | prior_eps.isna()
        )
        fresh_report = revenue_update | eps_update

        revenue_rank = rank(revenue_surprise).fillna(0.5)
        eps_rank = rank(eps_surprise).fillna(0.5)
        conflict = revenue_rank - eps_rank
        industry_conflict = indneutralize(conflict, data["industry"])

        expensive_growth = rank(ps_ratio).fillna(0.5) >= 0.5
        valid_conflict = revenue_surprise.notna() & eps_surprise.notna()
        active = fresh_report & valid_conflict & expensive_growth

        raw = industry_conflict.where(active, 0.0)
        return rank(raw).reindex_like(template).fillna(0.5)
