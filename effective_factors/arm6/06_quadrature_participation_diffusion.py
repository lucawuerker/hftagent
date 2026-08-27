"""Quadrature participation diffusion: price discovery can lead trading participation.

Mechanism: a coherent directional return component accompanied by a lagging volume
component suggests that price discovery precedes the broader execution and attention
response.  The signal isolates this causal phase relationship rather than applying
unconditional trend following, expecting the delayed participation to extend the
move over the following week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns
from quant_fund_agent.factors.registry import register_factor


def _quadrature_diffusion(ret: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Causal return-volume cross-spectrum at an eight-bar frequency.

    All weighted terms use current or positively lagged observations.  A positive
    imaginary cross-spectrum under this coefficient convention means volume is
    phase-delayed relative to returns at the selected frequency.
    """
    window = 24
    period = 8.0
    omega = 2.0 * np.pi / period

    log_volume = np.log(volume.where(volume > 0.0))
    vol_center = log_volume - log_volume.rolling(window, min_periods=window).mean()
    ret_center = ret - ret.rolling(window, min_periods=window).mean()

    r_real = ret * 0.0
    r_imag = ret * 0.0
    v_real = ret * 0.0
    v_imag = ret * 0.0
    for lag in range(window):
        cosine = float(np.cos(omega * lag))
        sine = float(np.sin(omega * lag))
        r_lag = ret_center.shift(lag)
        v_lag = vol_center.shift(lag)
        r_real = r_real + cosine * r_lag
        r_imag = r_imag - sine * r_lag
        v_real = v_real + cosine * v_lag
        v_imag = v_imag - sine * v_lag

    r_power = r_real.pow(2) + r_imag.pow(2)
    v_power = v_real.pow(2) + v_imag.pow(2)
    cross_real = r_real * v_real + r_imag * v_imag
    cross_imag = r_imag * v_real - r_real * v_imag
    coherence = (cross_real.pow(2) + cross_imag.pow(2)) / (
        r_power * v_power
    ).replace(0.0, np.nan)
    coherence = coherence.clip(lower=0.0, upper=1.0)

    phase_lag = cross_imag / np.sqrt(
        (cross_real.pow(2) + cross_imag.pow(2)).replace(0.0, np.nan)
    )
    delayed_participation = phase_lag.clip(lower=0.0, upper=1.0)

    ret_scale = ret.rolling(window, min_periods=window).std() * np.sqrt(float(window))
    directional_component = r_real / ret_scale.replace(0.0, np.nan)
    valid = ret.notna().rolling(window, min_periods=window).sum().eq(window)
    valid = valid & log_volume.notna().rolling(window, min_periods=window).sum().eq(window)

    return (directional_component * coherence * delayed_participation).where(valid)


@register_factor
class QuadratureParticipationDiffusion(BaseFactor):
    factor_id = "quadrature_participation_diffusion"
    name = "Quadrature Participation Diffusion"
    category = "microstructure"
    description = (
        "Ranks directional return-frequency components when log-volume innovations "
        "are spectrally coherent but phase-delayed, indicating delayed participation."
    )
    window_length = 24
    inputs = ["close", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = returns(data)
        raw = _quadrature_diffusion(ret, data["volume"])
        return (rank(raw) - 0.5).reindex_like(close).fillna(0.0)
