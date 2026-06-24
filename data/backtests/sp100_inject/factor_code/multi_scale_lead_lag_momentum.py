"""Multi-Scale Lead-Lag Momentum.

The wavelet lead-lag paper argues that different market participants react on different time scales, so a price move can propagate from fast to slow horizons with a stable lag structure. This factor looks for tickers whose short-horizon returns are consistently preceded by the market's own earlier returns while the lagged response remains coherent across several intraday horizons; such names should continue to underreact in the near term. It complements plain momentum by explicitly requiring a cross-horizon lag pattern rather than just persistent drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, rank, returns, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MultiScaleLeadLagMomentum(BaseFactor):
    """Cross-horizon lead-lag momentum score from lagged return alignment."""

    factor_id = "multi_scale_lead_lag_momentum"
    name = "Multi-Scale Lead-Lag Momentum"
    category = "statistical_arbitrage"
    description = (
        "Compute multi-horizon returns using close-to-close changes over several rolling windows, "
        "then measure how strongly each horizon aligns with lagged versions of shorter-horizon returns "
        "within the same ticker. Aggregate the lagged alignment score across horizons, rewarding stable "
        "positive self-lead and penalizing inconsistent or noisy lag structure."
    )
    window_length = 48
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        close_filled = close.replace(0, np.nan)

        # Multi-horizon close-to-close returns.
        r1 = returns({"close": close_filled})
        r2 = delta(close_filled, 2) / close_filled.shift(2)
        r4 = delta(close_filled, 4) / close_filled.shift(4)
        r8 = delta(close_filled, 8) / close_filled.shift(8)
        r16 = delta(close_filled, 16) / close_filled.shift(16)

        # Lead-lag alignment: longer-horizon return should be preceded by shorter-horizon return.
        s2 = correlation(r2, delta(r1, 1), 12)
        s4 = correlation(r4, delta(r2, 1), 12)
        s8 = correlation(r8, delta(r4, 1), 12)
        s16 = correlation(r16, delta(r8, 1), 12)

        # Stability terms: sign agreement between current and lagged horizon returns.
        a2 = rank((r2 * ts_mean(r1, 3)).where(np.isfinite(r2), np.nan))
        a4 = rank((r4 * ts_mean(r2, 3)).where(np.isfinite(r4), np.nan))
        a8 = rank((r8 * ts_mean(r4, 3)).where(np.isfinite(r8), np.nan))
        a16 = rank((r16 * ts_mean(r8, 3)).where(np.isfinite(r16), np.nan))

        # Aggregate across horizons; favor strong positive lead-lag coherence.
        score = (s2 + s4 + s8 + s16) / 4.0
        stability = (a2 + a4 + a8 + a16) / 4.0
        signal = score * (2.0 * stability - 1.0)

        signal = signal.replace([np.inf, -np.inf], np.nan)
        signal = signal.fillna(0.0)
        return signal
