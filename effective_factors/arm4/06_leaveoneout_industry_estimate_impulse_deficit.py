"""Estimate changes are often incorporated first into highly visible firms, while economically similar companies adjust with delay as analysts, discretionary investors, and sector funds update their relative valuation frameworks. The signal identifies an industry-level consensus impulse from peers' recent EPS-estimate changes, excluding the target company, and buys firms whose own estimate revision and price response have lagged that peer impulse. It is a falsifiable lead-lag hypothesis: the factor should fail when leave-one-out peer estimate shocks do not predict target returns after controlling for the target's own revision, industry return, and size. The delay can persist because quarterly estimate updates are sparse, analyst coverage is uneven, and portfolio managers often rebalance peer exposures gradually."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _latest_normalized_estimate_revision(
    estimate: pd.DataFrame,
    max_age: int,
) -> pd.DataFrame:
    """Carry the latest genuine consensus-estimate step change forward.

    EPS estimates are point-in-time step functions.  Treating each unchanged
    daily observation as a fresh revision would spuriously manufacture an
    impulse, so only estimate transitions create events.  The event is carried
    for a bounded age to avoid using very stale quarterly information.
    """
    previous = estimate.shift(1)
    denominator = previous.abs().clip(lower=0.05)
    event_revision = ((estimate - previous) / denominator).clip(-3.0, 3.0)
    changed = estimate.notna() & previous.notna() & estimate.ne(previous)
    return event_revision.where(changed).ffill(limit=max_age)


def _leave_one_out_industry_median(
    values: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """Return each stock's same-date industry median excluding itself.

    Industry labels are static in this feed.  The calculation is explicitly
    leave-one-out, rather than a group median containing the target, so a
    large revision by a single visible company cannot mechanically become its
    own peer signal.  Singleton industries remain missing and are neutralized
    by the caller.
    """
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    if values.empty or industry.empty:
        return result

    labels = industry.iloc[0].reindex(values.columns)
    valid_labels = labels.dropna()
    for label in pd.unique(valid_labels):
        members = list(valid_labels.index[valid_labels.eq(label)])
        if len(members) < 2:
            continue
        group_values = values.loc[:, members]
        for ticker in members:
            peers = [peer for peer in members if peer != ticker]
            result.loc[:, ticker] = group_values.loc[:, peers].median(axis=1)

    return result


@register_factor
class LeaveOneOutIndustryEstimateImpulseDeficit(BaseFactor):
    """Ranks firms lagging a directionally matched leave-one-out industry impulse."""

    factor_id = "leaveoneout_industry_estimate_impulse_deficit"
    name = "Industry Estimate-Impulse Catch-Up"
    category = "momentum"
    description = (
        "Within each industry, forms a leave-one-out median of the latest "
        "normalized EPS-consensus revision and favors names whose own revision "
        "and recent price response lag that directionally matched peer impulse."
    )
    window_length = 63
    inputs = ["close", "epsEstimate", "industry"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        estimate = data["epsEstimate"].reindex(index=close.index, columns=close.columns)
        industry = data["industry"].reindex(index=close.index, columns=close.columns)

        # A bounded carry converts sparse reporting-step updates into the most
        # recently public consensus impulse without inventing daily revisions.
        own_impulse = _latest_normalized_estimate_revision(estimate, 63)
        peer_impulse = _leave_one_out_industry_median(own_impulse, industry)

        # Use the same leave-one-out construction for the short price-response
        # benchmark.  Positive return_lag means the stock has underperformed its
        # peers over the response window.
        five_bar_return = close / close.shift(5) - 1.0
        peer_return = _leave_one_out_industry_median(five_bar_return, industry)
        return_lag = peer_return - five_bar_return

        revision_lag = peer_impulse - own_impulse
        direction = np.sign(peer_impulse)

        # A positive peer revision is attractive only when the target has both
        # revised and traded less favorably; for negative impulses the signs
        # reverse, naturally producing a short-side signal.
        directional_revision_lag = direction * revision_lag
        directional_return_lag = direction * return_lag

        composite = (
            rank(peer_impulse)
            + rank(directional_revision_lag)
            + rank(directional_return_lag)
        )
        signal = 2.0 * rank(composite) - 1.0
        return signal.reindex(index=close.index, columns=close.columns).fillna(0.0)
