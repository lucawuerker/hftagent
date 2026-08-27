'''Sector commonality inventory debt release.

Mechanism: persistent signed abnormal participation creates dealer inventory debt when
price impact is absorbed. A sector-wide liquidity-cost shock identifies a common
warehouse constraint; once both activity and impact cool, dealers unwind that debt.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _sector_mean(values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for _, members in labels.groupby(labels).groups.items():
        tickers = list(members)
        group = values.loc[:, tickers]
        average = group.mean(axis=1)
        out.loc[:, tickers] = pd.DataFrame(
            np.repeat(average.to_numpy()[:, None], len(tickers), axis=1),
            index=values.index,
            columns=tickers,
        )
    return out


def _trailing_zscore(values: pd.DataFrame, window: int, minimum: int) -> pd.DataFrame:
    average = values.rolling(window, min_periods=minimum).mean()
    deviation = values.rolling(window, min_periods=minimum).std()
    return (values - average) / deviation.replace(0.0, np.nan)


@register_factor
class SectorCommonalityInventoryDebtRelease(BaseFactor):
    factor_id = 'sector_commonality_inventory_debt_release'
    name = 'Sector Commonality Inventory Debt Release'
    category = 'microstructure'
    description = (
        'Fades accumulated signed abnormal participation after activity and impact '
        'cool, with a smooth sector-wide liquidity-cost stress condition.'
    )
    window_length = 60
    inputs = ['open', 'high', 'low', 'close', 'volume', 'sector']
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data['close'].replace(0.0, np.nan)
        open_price = data['open'].reindex(index=close.index, columns=close.columns).replace(0.0, np.nan)
        high = data['high'].reindex(index=close.index, columns=close.columns)
        low = data['low'].reindex(index=close.index, columns=close.columns)
        volume = data['volume'].reindex(index=close.index, columns=close.columns).fillna(0.0).clip(lower=0.0)
        labels = data['sector'].iloc[0].reindex(close.columns).fillna('Unknown')

        bar_range = (high - low).clip(lower=0.0)
        location = ((2.0 * close - high - low) / bar_range.replace(0.0, np.nan)).clip(-1.0, 1.0).fillna(0.0)
        impact = ((close - open_price).abs() / bar_range.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)

        volume_base = volume.shift(1).rolling(20, min_periods=12).mean()
        activity = np.log1p((volume / volume_base.replace(0.0, np.nan)).clip(0.0, 30.0)).fillna(0.0)
        signed_pressure = location * activity
        inventory_debt = signed_pressure.ewm(halflife=6, adjust=False, min_periods=6).mean().shift(1)

        prior_activity = activity.shift(3).rolling(5, min_periods=4).mean()
        activity_cooling = (prior_activity - activity) / prior_activity.replace(0.0, np.nan)
        activity_cooling = activity_cooling.clip(lower=0.0, upper=2.0).fillna(0.0)

        prior_impact = impact.shift(3).rolling(5, min_periods=4).mean()
        impact_cooling = (prior_impact - impact) / prior_impact.replace(0.0, np.nan)
        impact_cooling = impact_cooling.clip(lower=0.0, upper=2.0).fillna(0.0)

        dollar_volume = (close.abs() * volume).replace(0.0, np.nan)
        liquidity_cost = bar_range / dollar_volume
        log_cost = np.log(liquidity_cost.where(liquidity_cost > 0.0))
        sector_cost = _sector_mean(log_cost, labels)
        sector_stress = _trailing_zscore(sector_cost, 60, 35).clip(-5.0, 5.0)
        common_constraint = sector_stress.clip(lower=0.0, upper=4.0).fillna(0.0)

        score = -inventory_debt * activity_cooling * impact_cooling * common_constraint
        score = score.replace([np.inf, -np.inf], np.nan)
        return rank(score).fillna(0.5)
