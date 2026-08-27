"""Valuation-conditioned drift after a silent consensus confirmation.

Mechanism: when a sparse estimate revision confirms an already established
sector-relative price move, the revision may certify a gradual information
process rather than mark its endpoint.  Continuation is strongest when the
revision also validates the stock's relative valuation regime: costly names
upgraded by consensus and cheap names downgraded by consensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _sector_demean(value: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    """Remove contemporaneous sector effects using static sector membership."""
    labels = sector.ffill().iloc[-1].fillna("Unknown").astype(str)
    sector_mean = value.T.groupby(labels, sort=False).transform("mean").T
    return value - sector_mean


@register_factor
class SilentConfirmationValuationRegimeDrift(BaseFactor):
    factor_id = "silent_confirmation_valuation_regime_drift"
    name = "Silent Confirmation Valuation Drift"
    category = "sentiment"
    description = (
        "Directional consensus-confirmation drift after an extended update "
        "silence, amplified when relative PE is coherent with the revision."
    )
    window_length = 60
    inputs = ["close", "epsEstimate", "revenueEstimate", "peRatio", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float)
        eps = data["epsEstimate"].astype(float)
        revenue = data["revenueEstimate"].astype(float)
        pe = data["peRatio"].astype(float)
        sector = data["sector"]

        prior_eps = eps.shift(1)
        prior_revenue = revenue.shift(1)
        eps_update = eps.notna() & prior_eps.notna() & eps.ne(prior_eps)
        revenue_update = (
            revenue.notna() & prior_revenue.notna() & revenue.ne(prior_revenue)
        )
        update = eps_update | revenue_update

        # A true silent update requires observed estimate history and no update
        # in the preceding month; all observations used here precede today.
        observed_before = (eps.notna() | revenue.notna()).shift(1)
        sufficient_history = observed_before.rolling(21, min_periods=21).sum().ge(21.0)
        previous_updates = update.shift(1).rolling(20, min_periods=20).sum()
        silent = update & sufficient_history & previous_updates.eq(0.0)

        eps_revision = ((eps - prior_eps) / (prior_eps.abs() + 0.10)).clip(-2.0, 2.0)
        revenue_revision = (
            (revenue - prior_revenue) / (prior_revenue.abs() + 1.0)
        ).clip(-2.0, 2.0)
        revision = (
            0.7 * eps_revision.where(eps_update, 0.0).fillna(0.0)
            + 0.3 * revenue_revision.where(revenue_update, 0.0).fillna(0.0)
        )

        ret = close.replace(0.0, np.nan).pct_change(fill_method=None).clip(-0.30, 0.30)
        idio_ret = _sector_demean(ret, sector)
        prior_move = idio_ret.rolling(15, min_periods=12).sum()
        prior_vol = idio_ret.rolling(20, min_periods=12).std()
        move_strength = (
            prior_move.abs() / (prior_vol * np.sqrt(15.0)).replace(0.0, np.nan)
        ).clip(0.0, 3.0)
        aligned = (revision * prior_move).gt(0.0)

        # Relative valuation is a state variable, not a standalone value trade.
        valid_pe = (pe > 0.0) & np.isfinite(pe)
        log_pe = np.log(pe.where(valid_pe).clip(lower=0.1, upper=500.0))
        valuation_state = rank(_sector_demean(log_pe, sector)).sub(0.5).fillna(0.0)

        signed_confirmation = (revision * move_strength).where(
            silent & aligned & valid_pe, 0.0
        ).fillna(0.0)
        coherence = (
            1.0 + 1.2 * np.sign(signed_confirmation) * valuation_state
        ).clip(lower=0.4, upper=1.6)
        event = signed_confirmation * coherence

        # Consensus dissemination is gradual; retain a short causal event memory.
        drift = event.ewm(halflife=2.5, adjust=False, min_periods=1).mean()
        signal = _sector_demean(drift, sector)
        return rank(signal).fillna(0.0)
