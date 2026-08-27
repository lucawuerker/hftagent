"""Fresh analyst-revision information with incomplete price assimilation.

Mechanism: a newly published consensus EPS revision is informative, but its
price effect is delayed when the immediate reaction is small relative to the
name's normal daily volatility.  The factor isolates this event-conditioned
reaction gap rather than extrapolating ordinary price momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns
from quant_fund_agent.factors.registry import register_factor


def _fresh_eps_revision(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return causal fresh relative estimate changes and their event mask.

    Consensus estimates are point-in-time step functions.  A change between
    today and yesterday is therefore a newly available update, not a revision
    inferred using future observations.
    """
    previous = estimates.shift(1)
    fresh = estimates.notna() & previous.notna() & estimates.ne(previous)
    relative_change = (estimates - previous) / (previous.abs() + 0.05)
    return relative_change.clip(lower=-2.0, upper=2.0).where(fresh), fresh


@register_factor
class FreshRevisionReactionInertia(BaseFactor):
    factor_id = "fresh_revision_reaction_inertia"
    name = "Fresh Revision Reaction Inertia"
    category = "sentiment"
    description = (
        "Fresh cross-sectional EPS-consensus revisions weighted by how little "
        "the stock's same-day volatility-normalized return has incorporated "
        "the revision direction."
    )
    window_length = 21
    inputs = ["close", "epsEstimate"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        revision, fresh = _fresh_eps_revision(data["epsEstimate"])

        # Ranking only the fresh revision innovation is robust to differing EPS
        # scales while retaining the economically meaningful revision sign.
        revision_score = (rank(revision.fillna(0.0)) - 0.5).where(fresh, 0.0)

        ret = returns(data)
        volatility = ret.rolling(20, min_periods=10).std().replace(0.0, np.nan)
        signed_reaction = np.sign(revision_score) * (ret / volatility)

        # A response of one own-volatility unit in the revision direction is
        # treated as substantially assimilated.  Muted or opposite reactions
        # receive more weight, without turning this into an unconditional fade.
        assimilation_gap = 1.0 - signed_reaction.clip(lower=-1.0, upper=1.0)
        raw = (revision_score * assimilation_gap).where(fresh, 0.0)

        return raw.reindex_like(close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
