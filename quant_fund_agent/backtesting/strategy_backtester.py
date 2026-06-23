"""Vectorised strategy backtester.

Takes a strategy's raw signal DataFrame (positive = long, negative = short)
and converts it into a dollar-neutral long/short portfolio, then computes
PnL and summary statistics.

Pipeline
--------
0. factor signals are cross-sectionally z-scored so every alpha is on a
   comparable scale before the strategy combines them
1. strategy.calc(factor_signals, data) → raw signal  (timestamps × tickers)
2. signal → normalised positions:
   a. cross-sectional z-score
   b. apply min_conviction threshold  (zero out weak signals)
   c. keep only top max_positions by |z|  (zero out the rest)
   d. if equal_weight, replace surviving magnitudes with ±1
   e. rescale so |weights| sum to 1, weights sum to ~0 (dollar-neutral)
3. positions × forward bar returns → per-bar portfolio return
4. portfolio return series → aggregate statistics

All operations are fully vectorised — no row-level Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns
from quant_fund_agent.backtesting.engine import _ic_series
from quant_fund_agent.data.frequency import (
    DEFAULT_BARS_PER_DAY,
    bars_per_day_from_index,
    trading_days_per_year_from_index,
)
from quant_fund_agent.schemas import StrategyBacktestMetrics
from quant_fund_agent.strategies.base import BaseStrategy


# Legacy default (10-sec LOBSTER bars).  Metrics now infer bars-per-day from the
# data's own index; this remains as the fallback for too-short series.
BARS_PER_DAY = DEFAULT_BARS_PER_DAY


@dataclass
class BacktestResult:
    """Full backtest output (metrics + intermediate series)."""
    metrics: StrategyBacktestMetrics
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    drawdown: pd.Series


# ── factor signal pre-normalisation ────────────────────────────────────

def normalise_factor_signals(
    factor_signals: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Cross-sectionally z-score each factor signal independently.

    This puts every alpha on the same scale (mean 0, std 1 across tickers
    at each timestamp) so that strategies combining multiple alphas are not
    dominated by whichever alpha has the largest raw magnitude.
    """
    # Vectorised cross-sectional z-score in NumPy.  Equivalent to the pandas
    # ``sig.mean/std(axis=1)`` form (skipna, std ddof=1, σ==0 → NaN) but avoids
    # pandas' per-row NaN-mask machinery (``_isna_array``), which dominated the
    # backtest hot path when this runs thousands of times across the comparison.
    out: dict[str, pd.DataFrame] = {}
    for fid, sig in factor_signals.items():
        arr = sig.to_numpy(dtype=float)
        mask = ~np.isnan(arr)
        cnt = mask.sum(axis=1)
        ssum = np.where(mask, arr, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(cnt > 0, ssum / cnt, np.nan)
            diff = arr - mu[:, None]
            sq = np.where(mask, diff * diff, 0.0).sum(axis=1)
            std = np.sqrt(np.where(cnt > 1, sq / np.where(cnt > 1, cnt - 1, 1), np.nan))
            std = np.where(std == 0.0, np.nan, std)
            normed = (arr - mu[:, None]) / std[:, None]
        out[fid] = pd.DataFrame(normed, index=sig.index, columns=sig.columns)
    return out


# ── position construction ──────────────────────────────────────────────

def _signal_to_positions(
    signal: pd.DataFrame,
    holding_period: int = 1,
    max_positions: int = 20,
    equal_weight: bool = False,
    min_conviction: float = 0.0,
) -> pd.DataFrame:
    """Convert a raw signal matrix into dollar-neutral portfolio weights.

    Steps (all vectorised):
        1. Resample signal for *holding_period* (hold between rebalances).
        2. Cross-sectional z-score.
        3. Zero out positions where ``|z| < min_conviction``.
        4. Keep only the *max_positions* tickers with the largest ``|z|``
           per row; zero out the rest.
        5. If *equal_weight*, replace surviving z-scores with their sign.
        6. Rescale so ``sum(|w|) = 1`` (dollar-neutral: ``sum(w) ≈ 0``).
    """
    # 1 — holding period resampling
    if holding_period > 1:
        mask = np.arange(len(signal)) % holding_period == 0
        signal = signal.where(pd.Series(mask, index=signal.index), other=np.nan)
        signal = signal.ffill()

    # 2 — cross-sectional z-score
    mu = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0, np.nan)
    z = signal.sub(mu, axis=0).div(std, axis=0)

    # 3 — minimum conviction filter
    if min_conviction > 0:
        z = z.where(z.abs() >= min_conviction, other=0.0)

    # 4 — keep only top max_positions by |z| per row
    abs_z = z.abs()
    # rank descending: rank 1 = largest absolute z-score
    rk = abs_z.rank(axis=1, ascending=False, method="first")
    z = z.where(rk <= max_positions, other=0.0)

    # 5 — equal weight: replace magnitudes with sign
    if equal_weight:
        z = np.sign(z)

    # 6 — rescale to |weights| sum to 1
    abs_sum = z.abs().sum(axis=1).replace(0, np.nan)
    positions = z.div(abs_sum, axis=0)
    return positions


# ── return computation ─────────────────────────────────────────────────

def _portfolio_returns(
    positions: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.Series:
    """Per-bar portfolio return from positions and single-bar forward returns.

    ``port_ret[t] = sum_i( position_i[t] * (close_i[t+1]/close_i[t] - 1) )``
    """
    fwd = forward_returns(close, horizon=1)

    common_idx = positions.index.intersection(fwd.index)
    common_cols = positions.columns.intersection(fwd.columns)
    # Trade with a one-bar delay to avoid same-bar leakage:
    # signals/positions formed at the end of bar t are only tradable from bar t+1.
    pos = positions.shift(1).loc[common_idx, common_cols]
    ret = fwd.loc[common_idx, common_cols]

    port_ret = (pos * ret).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret


# ── statistics helpers ─────────────────────────────────────────────────

def _drawdown_series(cum_ret: pd.Series) -> pd.Series:
    wealth = 1.0 + cum_ret
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    dd.name = "drawdown"
    return dd


def _max_drawdown_duration(dd: pd.Series) -> int:
    is_in_dd = dd < 0
    groups = (~is_in_dd).cumsum()
    if is_in_dd.sum() == 0:
        return 0
    return int(is_in_dd.groupby(groups).sum().max())


def _compute_metrics(
    port_ret: pd.Series,
    positions: pd.DataFrame,
    signal: pd.DataFrame,
    close: pd.DataFrame,
    bars_per_day: int | None = None,
    ic_horizon: int = 6,
) -> StrategyBacktestMetrics:
    """Derive all summary statistics from the portfolio return series.

    ``bars_per_day=None`` (the default) infers the annualisation from the
    return series' own ``DatetimeIndex`` — 2340 for 10-sec LOBSTER bars, 1 for
    daily — so the same code annualises correctly at any frequency.
    """
    n = len(port_ret)
    if n < 2:
        return StrategyBacktestMetrics()

    if bars_per_day is None:
        bars_per_day = bars_per_day_from_index(port_ret.index)
    # Trading-days/year inferred from the index (365 if it trades weekends, e.g.
    # crypto; 252 otherwise) → correct annualisation across asset classes.
    bars_per_year = bars_per_day * trading_days_per_year_from_index(port_ret.index)

    # annualised return & vol
    mean_bar = float(port_ret.mean())
    std_bar = float(port_ret.std())
    ann_ret = mean_bar * bars_per_year
    ann_vol = std_bar * np.sqrt(bars_per_year) if std_bar > 0 else 0.0

    # Sharpe
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside = port_ret[port_ret < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    ann_downside = downside_std * np.sqrt(bars_per_year)
    sortino = (ann_ret / ann_downside) if ann_downside > 0 else 0.0

    # drawdown
    cum_ret = port_ret.cumsum()
    dd = _drawdown_series(cum_ret)
    max_dd = float(dd.min())
    max_dd_dur = _max_drawdown_duration(dd)

    # Calmar
    calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else 0.0

    # win / loss
    wins = port_ret[port_ret > 0]
    losses = port_ret[port_ret < 0]
    hit_rate = float(len(wins) / n) if n > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(losses.abs().sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    # turnover = mean absolute change in positions per bar
    pos_diff = positions.diff().abs().sum(axis=1)
    avg_daily_turnover = float(pos_diff.mean()) * bars_per_day

    # average number of non-zero positions held
    avg_positions = float((positions.abs() > 1e-9).sum(axis=1).mean())

    # IC of the composite signal at the strategy's prediction horizon.
    fwd_ret_h = forward_returns(close, horizon=ic_horizon)
    # Align IC to the same "tradeable next bar" convention:
    # signal at t is evaluated against returns starting at t+1.
    ic = _ic_series(signal, fwd_ret_h.shift(-1))
    ic_mean = float(ic.mean()) if len(ic) > 0 else None
    ic_std = float(ic.std()) if len(ic) > 0 else None
    ic_ir = float(ic_mean / ic_std) if ic_std and ic_std > 0 else None

    return StrategyBacktestMetrics(
        annualised_return=round(ann_ret, 6),
        annualised_volatility=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        max_drawdown=round(max_dd, 6),
        max_drawdown_duration_bars=max_dd_dur,
        hit_rate=round(hit_rate, 4),
        profit_factor=round(profit_factor, 4),
        avg_win=round(avg_win, 8),
        avg_loss=round(avg_loss, 8),
        avg_daily_turnover=round(avg_daily_turnover, 4),
        avg_positions_held=round(avg_positions, 2),
        ic_mean=round(ic_mean, 6) if ic_mean is not None else None,
        ic_ir=round(ic_ir, 6) if ic_ir is not None else None,
        extra={
            "n_bars": n,
            "bars_per_day": bars_per_day,
        },
    )


# ── public API ─────────────────────────────────────────────────────────

def backtest_strategy(
    strategy: BaseStrategy,
    factor_signals: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
) -> BacktestResult:
    """Run a full vectorised backtest for a strategy.

    Args:
        strategy: instantiated strategy with a ``.calc()`` method.
        factor_signals: ``{factor_id: signal_df}`` pre-computed.
        data: OHLCV panel dict.

    Returns:
        BacktestResult containing metrics and all intermediate series.
    """
    normed_signals = normalise_factor_signals(factor_signals)
    signal = strategy.calc(normed_signals, data)
    construction = getattr(strategy, "position_construction", "cross_sectional")
    if construction == "per_underlying":
        # Directional per-name book: standardise each underlying over its own
        # history, go long/flat/short by a boundary, size each active name to
        # 1/max_positions.  Not dollar-neutral (has net market exposure).
        from quant_fund_agent.backtesting.positions import (
            DEFAULT_POSITION_MODE,
            DEFAULT_POSITION_THRESHOLD,
            DEFAULT_ZSCORE_BASIS,
            DEFAULT_ZSCORE_WINDOW,
            per_underlying_positions,
        )

        pp = getattr(strategy, "position_params", None) or {}
        positions = per_underlying_positions(
            signal,
            n_max_positions=strategy.max_positions,
            holding_period=strategy.holding_period,
            mode=pp.get("mode", DEFAULT_POSITION_MODE),
            threshold=pp.get("threshold", DEFAULT_POSITION_THRESHOLD),
            zscore_basis=pp.get("zscore_basis", DEFAULT_ZSCORE_BASIS),
            zscore_window=pp.get("zscore_window", DEFAULT_ZSCORE_WINDOW),
        )
    else:
        positions = _signal_to_positions(
            signal,
            holding_period=strategy.holding_period,
            max_positions=strategy.max_positions,
            equal_weight=strategy.equal_weight,
            min_conviction=strategy.min_conviction,
        )
    port_ret = _portfolio_returns(positions, data["close"])
    cum_ret = port_ret.cumsum()
    dd = _drawdown_series(cum_ret)

    metrics = _compute_metrics(
        port_ret, positions, signal, data["close"],
        ic_horizon=getattr(strategy, "target_horizon", 6),
    )

    return BacktestResult(
        metrics=metrics,
        portfolio_returns=port_ret,
        cumulative_returns=cum_ret,
        positions=positions,
        drawdown=dd,
    )
