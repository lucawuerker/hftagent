"""Investors often exhibit a tendency to seek outperformance against a benchmark, especially under conditions of risk aversion. This factor leverages the insights from the game-theoretic approach to highlight stocks that are expected to outperform a risk-adjusted benchmark, thus capitalizing on the behavioral bias of investors who adjust their strategies according to perceived risk. Such a strategy aligns with findings in the literature that underscore the importance of risk sensitivity in asset management."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_mean, stddev, returns)

@register_factor
class RiskSensitiveBenchmarkOutperformance(BaseFactor):
    factor_id = "risk_sensitive_benchmark_outperformance"
    name = "Risk-Sensitive Benchmark Outperformance"
    category = "other"
    description = "This factor computes the expected outperformance of each stock relative to a risk-sensitive benchmark, taking into account historical volatility and recent performance metrics."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        returns_series = returns(data)
        historical_volatility = stddev(returns_series, 20)
        recent_performance = ts_mean(returns_series, 20)
        expected_outperformance = recent_performance / historical_volatility
        return rank(expected_outperformance.fillna(0.0))