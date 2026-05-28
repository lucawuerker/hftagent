"""Portfolio construction & analytics utilities used by the PM agent.

The PM agent does *not* implement optimisation logic itself — it picks
which method to use from the catalog in ``construction.py`` and calls
it.  This keeps the agent thin and the maths testable in isolation.

Public surface
--------------
* ``construction``  – ``equal_weight``, ``inverse_volatility``,
  ``min_variance``, ``mean_variance``, ``max_sharpe``, ``risk_parity``,
  ``hierarchical_risk_parity`` + dispatcher ``construct_portfolio``.
* ``correlations``  – ``compute_correlation_matrix`` over a dict of
  per-strategy return series.
* ``personalities`` – pre-baked PM risk profiles (defensive / balanced
  / aggressive) used as default config for the agent.
"""

from quant_fund_agent.portfolio import construction, correlations, personalities
from quant_fund_agent.portfolio.construction import (
    PortfolioWeights,
    construct_portfolio,
    equal_weight,
    expected_portfolio_metrics,
    hierarchical_risk_parity,
    inverse_volatility,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_parity,
)
from quant_fund_agent.portfolio.correlations import (
    compute_correlation_matrix,
    covariance_from_vols_and_corr,
)
from quant_fund_agent.portfolio.personalities import (
    PersonalityProfile,
    get_personality_profile,
)

__all__ = [
    # modules
    "construction",
    "correlations",
    "personalities",
    # construction
    "PortfolioWeights",
    "construct_portfolio",
    "equal_weight",
    "inverse_volatility",
    "min_variance",
    "mean_variance",
    "max_sharpe",
    "risk_parity",
    "hierarchical_risk_parity",
    "expected_portfolio_metrics",
    # correlations
    "compute_correlation_matrix",
    "covariance_from_vols_and_corr",
    # personalities
    "PersonalityProfile",
    "get_personality_profile",
]
