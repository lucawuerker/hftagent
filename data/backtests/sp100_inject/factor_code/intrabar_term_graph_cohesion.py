"""Inspired by the paper’s core claim that non-local context and topology reveal hidden structure in short texts, this factor treats each bar as a small "sentence" made of OHLCV states. When open, high, low, close, and volume move in a mutually coherent way across a short rolling window, the price path tends to reflect cleaner information assimilation rather than noisy churn, which should be more persistent cross-sectionally. Conversely, bars whose intrabar structure is topologically inconsistent relative to recent history often indicate transient dislocations that mean-revert."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, correlation, delta, rank, ts_mean, ts_sum, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarTermGraphCohesion(BaseFactor):
    """Rolling OHLCV coherence score built from intrabar structural agreement."""

    factor_id = "intrabar_term_graph_cohesion"
    name = "Intrabar Term-Graph Cohesion"
    category = "other"
    description = (
        "Compute a rolling coherence score from standardized OHLCV features within each ticker, "
        "using the joint agreement of return direction, range expansion, close location, and volume "
        "surprise across recent bars. The factor is higher when the current bar's feature pattern "
        "is similar to a locally consistent historical regime and lower when the bar looks like a "
        "structural outlier."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12

        # Intrabar structural features.
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_ = (high - low).div(close.abs().replace(0, np.nan) + eps).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        close_loc = ((close - low).div((high - low).replace(0, np.nan) + eps) * 2.0 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        body = (close - open_).div((high - low).replace(0, np.nan) + eps).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol_log = np.log1p(volume.clip(lower=0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Standardize locally to capture regime-relative structure.
        ret_z = ret.div(stddev(ret, 10).replace(0, np.nan)).fillna(0.0)
        range_z = (range_ - ts_mean(range_, 10)).div(stddev(range_, 10).replace(0, np.nan)).fillna(0.0)
        close_loc_z = (close_loc - ts_mean(close_loc, 10)).div(stddev(close_loc, 10).replace(0, np.nan)).fillna(0.0)
        body_z = (body - ts_mean(body, 10)).div(stddev(body, 10).replace(0, np.nan)).fillna(0.0)
        vol_z = (vol_log - ts_mean(vol_log, 10)).div(stddev(vol_log, 10).replace(0, np.nan)).fillna(0.0)

        # Local agreement: current bar should align with recent multi-feature structure.
        signed_agreement = (
            rank(ret_z * body_z)
            + rank(close_loc_z * range_z)
            + rank(ret_z * close_loc_z)
            + rank(range_z * vol_z)
        ) / 4.0

        # Cohesion of the feature graph: strong positive correlations imply a cleaner regime.
        c1 = correlation(ret_z, body_z, 8)
        c2 = correlation(close_loc_z, range_z, 8)
        c3 = correlation(range_z, vol_z, 8)
        c4 = correlation(ret_z, close_loc_z, 8)
        graph_cohesion = (c1 + c2 + c3 + c4) / 4.0

        # Outlier penalty: large recent deviations in any standardized feature reduce cohesion.
        deviation = (
            abs_(delta(ret_z, 1))
            + abs_(delta(range_z, 1))
            + abs_(delta(close_loc_z, 1))
            + abs_(delta(body_z, 1))
            + abs_(delta(vol_z, 1))
        ) / 5.0
        deviation_score = 1.0 / (1.0 + ts_sum(deviation, 5).fillna(0.0))

        # Final score: higher when the bar is both internally coherent and aligned with the local regime.
        score = (0.55 * signed_agreement) + (0.35 * graph_cohesion) + (0.10 * deviation_score)
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return score
