"""Balance-sheet-buffered rebound after systemic deleveraging.

Mechanism: broad downside-volatility shocks induce correlated VaR reductions and
liquidity-demanding sales.  When a financially resilient company also suffers a
clustered idiosyncratic sell shock, the shock is more likely to reflect temporary
inventory transfer than a lasting solvency repricing; its relative return should
recover as forced execution and dealer hedging pressure dissipate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


def _stress_buffered_sell_score(
    close: pd.DataFrame,
    cash_ratio: pd.DataFrame,
    debt_to_assets: pd.DataFrame,
    sector: pd.DataFrame,
) -> pd.DataFrame:
    """Causal systemic-stress and idiosyncratic-sale interaction.

    The exponentially decayed downside innovations are a fixed-parameter
    Hawkes-style proxy for clustered selling.  All normalizers are shifted
    trailing estimates, so the current observation is never used to establish
    its own baseline.
    """
    clean_close = close.replace([np.inf, -np.inf], np.nan)
    ret = clean_close.pct_change().replace([np.inf, -np.inf], np.nan)
    ret = ret.clip(lower=-0.90, upper=2.00)

    market_ret = ret.median(axis=1)
    idio_ret = ret.sub(market_ret, axis=0)

    prior_idio_var = idio_ret.pow(2).rolling(60, min_periods=30).mean().shift(1)
    idio_vol = np.sqrt(prior_idio_var.replace(0.0, np.nan))
    downside_shock = (-idio_ret / idio_vol).clip(lower=0.0, upper=8.0)
    downside_shock = downside_shock.fillna(0.0)

    cluster = downside_shock.ewm(halflife=4.0, adjust=False, min_periods=6).mean()
    cluster_base = cluster.rolling(80, min_periods=40).median().shift(1)
    cluster_excess = (cluster / cluster_base.replace(0.0, np.nan) - 1.0)
    cluster_excess = cluster_excess.clip(lower=0.0, upper=8.0).fillna(0.0)

    market_downside_var = market_ret.clip(upper=0.0).pow(2)
    recent_stress = market_downside_var.rolling(15, min_periods=10).mean()
    normal_stress = market_downside_var.rolling(90, min_periods=45).median().shift(1)
    stress_excess = (recent_stress / normal_stress.replace(0.0, np.nan) - 1.0)
    stress_excess = stress_excess.clip(lower=0.0, upper=6.0).fillna(0.0)

    safe_cash = cash_ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=10.0)
    safe_debt = debt_to_assets.replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=5.0)
    raw_buffer = 0.5 * (rank(safe_cash) + rank(-safe_debt))
    raw_buffer = raw_buffer.fillna(0.5)

    # Financial institutions and capital-intensive industries have structurally
    # different ratio levels; retain only relative resilience within sector.
    buffer = rank(indneutralize(raw_buffer, sector)).fillna(0.5)

    systemic_pressure = pd.DataFrame(
        np.repeat(stress_excess.to_numpy()[:, None], close.shape[1], axis=1),
        index=close.index,
        columns=close.columns,
    )
    score = cluster_excess * systemic_pressure * (0.25 + buffer)
    active = (idio_ret < 0.0) & (cluster_excess > 0.0) & (systemic_pressure > 0.0)
    return score.where(active, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


@register_factor
class BalanceSheetBufferedMarketStressRebound(BaseFactor):
    factor_id = "balance_sheet_buffered_market_stress_rebound"
    name = "Balance-Sheet Buffered Stress Rebound"
    category = "microstructure"
    description = (
        "Long financially resilient names with clustered idiosyncratic downside "
        "pressure during an elevated broad-market downside-volatility regime."
    )
    window_length = 150
    inputs = ["close", "cashRatio", "debtToAssets", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        score = _stress_buffered_sell_score(
            data["close"],
            data["cashRatio"],
            data["debtToAssets"],
            data["sector"],
        )
        active = score > 0.0
        signal = (2.0 * rank(score) - 1.0).where(active, 0.0)
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
