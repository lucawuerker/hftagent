"""Distributed sector inventory transfer and delayed liquidity relaxation.

Mechanism: mixed-sign sector-wide range and turnover shocks can reflect basket
risk transfer rather than information.  When a constituent's relative move is
created through persistent participation and closes away from a directional
range endpoint, the displacement is an inventory overhang expected to repair
more slowly than a one-bar forced-flow reversal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _sector_broadcast(values: pd.DataFrame, labels: pd.Series, how: str) -> pd.DataFrame:
    """Broadcast a contemporaneous sector aggregation back to constituents."""
    return values.T.groupby(labels, sort=False).transform(how).T


@register_factor
class DistributedSectorInventoryRelaxationH6(BaseFactor):
    factor_id = "distributed_sector_inventory_relaxation_h6"
    name = "Distributed Sector Inventory Relaxation"
    category = "microstructure"
    description = (
        "Fades a persistent sector-relative displacement accumulated during "
        "mixed-sign synchronized range-and-turnover shocks, while excluding "
        "endpoint-confirmed directional moves."
    )
    window_length = 35
    inputs = ["close", "high", "low", "volume", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].reindex(index=close.index, columns=close.columns).astype(float)
        low = data["low"].reindex(index=close.index, columns=close.columns).astype(float)
        volume = data["volume"].reindex(index=close.index, columns=close.columns).astype(float)

        sector_frame = data["sector"].reindex(index=close.index, columns=close.columns)
        fallback = pd.Series(close.columns, index=close.columns, dtype=object)
        labels = sector_frame.iloc[0].where(sector_frame.iloc[0].notna(), fallback).astype(str)

        safe_close = close.abs().replace(0.0, np.nan)
        raw_return = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        bounded_return = raw_return.clip(-1.0, 1.0)
        valid = bounded_return.notna().astype(float)
        filled_return = bounded_return.fillna(0.0)

        sector_sum = _sector_broadcast(filled_return, labels, "sum")
        sector_count = _sector_broadcast(valid, labels, "sum")
        leave_one_out = (sector_sum - filled_return) / (sector_count - 1.0).replace(0.0, np.nan)
        relative_return = (bounded_return - leave_one_out).where(sector_count > 1.0).fillna(0.0)

        bar_range = ((high - low).abs() / safe_close).replace([np.inf, -np.inf], np.nan)
        log_range = np.log(bar_range.replace(0.0, np.nan))
        range_abnormal = (
            log_range - log_range.rolling(30, min_periods=12).mean()
        ).clip(-3.0, 3.0).fillna(0.0)

        log_volume = np.log(volume.where(volume > 0.0))
        volume_abnormal = (
            log_volume - log_volume.rolling(30, min_periods=12).mean()
        ).clip(-3.0, 3.0).fillna(0.0)

        sector_range = _sector_broadcast(range_abnormal, labels, "mean")
        sector_volume = _sector_broadcast(volume_abnormal, labels, "mean")
        range_dispersion = np.sqrt(
            _sector_broadcast((range_abnormal - sector_range).pow(2.0), labels, "mean").clip(lower=0.0)
        )
        volume_dispersion = np.sqrt(
            _sector_broadcast((volume_abnormal - sector_volume).pow(2.0), labels, "mean").clip(lower=0.0)
        )

        sign_sum = _sector_broadcast(np.sign(filled_return), labels, "sum")
        sign_coherence = (
            sign_sum / sector_count.replace(0.0, np.nan)
        ).abs().clip(0.0, 1.0).fillna(1.0)

        common_liquidity_shock = np.minimum(sector_range, sector_volume).clip(lower=0.0)
        synchronized_shock = common_liquidity_shock / (
            1.0 + range_dispersion + volume_dispersion
        )
        forced_transfer = (
            synchronized_shock / (synchronized_shock + 0.15)
        ) * (1.0 - sign_coherence)

        raw_range = (high - low).abs()
        close_location = (
            (2.0 * close - high - low) / raw_range.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0.0)
        non_endpoint = 1.0 - close_location.abs()

        persistent_participation = volume_abnormal.clip(lower=0.0, upper=3.0).rolling(
            3, min_periods=2
        ).mean().fillna(0.0)
        transfer_weight = forced_transfer * (1.0 + persistent_participation) * non_endpoint

        # A five-bar accumulation is deliberately slower than the parent's
        # one-bar polarity decision: it measures inventory still distributed
        # through an execution program rather than a completed impulse.
        inventory_overhang = (relative_return * transfer_weight).rolling(
            5, min_periods=2
        ).sum().fillna(0.0)
        active_transfer = transfer_weight.rolling(5, min_periods=2).sum().fillna(0.0)
        confidence = active_transfer / (active_transfer + 0.20)

        fade_score = rank(-inventory_overhang).fillna(0.5) - 0.5
        signal = fade_score * confidence
        return signal.where(raw_return.notna(), 0.0).fillna(0.0)
