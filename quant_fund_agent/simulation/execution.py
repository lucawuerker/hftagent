"""Fund-level execution: how per-strategy positions become a traded book + PnL.

This is where the trade-execution semantics the project needed to pin down live.
Two models, chosen by ``BacktestConfig.execution_model``:

**Consolidated netted book (default).**  Every live strategy *s* has a PM capital
weight ``w_s`` and a dollar-neutral position matrix ``P_s`` (time × ticker, from
``backtest_strategy(...).positions``).  We build ONE fund book::

    B(t, ticker) = Σ_s  w_s · P_s(t, ticker)      # offsetting signals NET here

then apply **fund-level** constraints to ``B`` (top-N ``max_positions`` by |weight|,
per-name cap, gross-leverage cap), and charge costs on the **netted** turnover.
So if strategy A is long AAPL and B is short AAPL, only the residual is traded
and charged once.

**Independent pod book.**  Each strategy trades its own book and pays its own
costs; the fund return is ``Σ_s w_s · r_s^net``.  No netting — both legs of a
conflicting trade execute and are charged.

Costs are **spread-aware + commission, no market impact**: for each unit of
turnover in a name we charge ``0.5 · spread/close`` (one side of the quoted
spread, from the panel) plus a fixed ``commission_bps``.

Everything reuses the existing vectorised engine
(:func:`quant_fund_agent.backtesting.strategy_backtester._portfolio_returns`) so a
single-strategy, zero-cost run reproduces ``backtest_strategy`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_fund_agent.simulation.config import BacktestConfig, ExecutionModel

PositionsByStrategy = dict[str, pd.DataFrame]
Weights = dict[str, float]


@dataclass
class ExecutionResult:
    """Per-bar fund PnL for one trading window, plus attribution."""

    fund_returns: pd.Series                 # net-of-cost fund return per bar
    gross_returns: pd.Series                # before costs
    cost: pd.Series                         # transaction cost per bar
    turnover: pd.Series                     # traded turnover per bar
    attribution: dict[str, pd.Series]       # strategy_id → weighted net contribution
    strategy_returns: dict[str, pd.Series] = field(default_factory=dict)  # standalone net r_s (unweighted)
    book: pd.DataFrame | None = None        # the netted fund book (netted model only)
    netting_benefit: pd.Series | None = None  # netted_fund − pod_fund (cum. interpretable)
    extra: dict[str, float] = field(default_factory=dict)


# ── cost model ─────────────────────────────────────────────────────────

def cost_rate_panel(panel: dict[str, pd.DataFrame], cfg: BacktestConfig) -> pd.DataFrame:
    """Per-unit-turnover fractional cost for each (bar, ticker).

    ``rate = (0.5 if half_spread else 1.0) · spread/close + commission``.
    Falls back to commission-only where the spread field is missing/NaN.
    """
    close = panel["close"]
    commission_frac = cfg.commission_bps * 1e-4
    rate = pd.DataFrame(commission_frac, index=close.index, columns=close.columns)

    if cfg.use_spread_cost and cfg.spread_field in panel:
        spread = panel[cfg.spread_field].reindex(index=close.index, columns=close.columns)
        spread_frac = (spread / close.replace(0, np.nan)).abs().fillna(0.0)
        mult = 0.5 if cfg.half_spread else 1.0
        rate = rate + mult * spread_frac
    return rate


# ── per-strategy net return (shared by both models) ─────────────────────

def _net_from_positions(
    positions: pd.DataFrame,
    close: pd.DataFrame,
    cost_rate: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return ``(gross, cost, net, turnover)`` per bar for one position book.

    ``gross`` uses the exact same engine as the per-strategy backtester (one-bar
    execution delay, no same-bar leakage).  The initial entry into the book is
    counted as turnover, so costs are charged from the first trade.
    """
    from quant_fund_agent.backtesting.strategy_backtester import _portfolio_returns

    gross = _portfolio_returns(positions, close)

    pos0 = positions.fillna(0.0)
    dpos = pos0.diff()
    if len(pos0) > 0:
        dpos.iloc[0] = pos0.iloc[0]          # entering the book is a trade
    cr = cost_rate.reindex(index=pos0.index, columns=pos0.columns).fillna(0.0)
    cost = (dpos.abs() * cr).sum(axis=1)
    turnover = dpos.abs().sum(axis=1)

    cost = cost.reindex(gross.index).fillna(0.0)
    turnover = turnover.reindex(gross.index).fillna(0.0)
    net = gross - cost
    return gross, cost, net, turnover


# ── netted-book construction ─────────────────────────────────────────────

def consolidate_book(positions: PositionsByStrategy, weights: Weights) -> pd.DataFrame:
    """Capital-weighted sum of every strategy's positions → one fund book.

    Offsetting positions on the same ticker net out here.
    """
    frames = [
        (weights.get(sid, 0.0), P.fillna(0.0))
        for sid, P in positions.items()
        if weights.get(sid, 0.0) != 0.0 and P is not None and not P.empty
    ]
    if not frames:
        return pd.DataFrame()

    idx = sorted(set().union(*[P.index for _, P in frames]))
    cols = sorted(set().union(*[P.columns for _, P in frames]))
    book = pd.DataFrame(0.0, index=pd.DatetimeIndex(idx), columns=cols)
    for w, P in frames:
        book = book + P.reindex(index=book.index, columns=book.columns).fillna(0.0) * w
    return book


def apply_fund_constraints(
    book: pd.DataFrame,
    max_positions: int,
    max_weight_per_name: float | None,
    gross_leverage: float,
) -> pd.DataFrame:
    """Apply fund-level risk caps to the consolidated book (row-wise).

    1. keep the ``max_positions`` names with the largest |weight|;
    2. clip each name to ``±max_weight_per_name`` (``None``/``>=1`` disables);
    3. scale any row whose gross exposure exceeds ``gross_leverage`` down to it.
    """
    if book.empty:
        return book
    z = book.copy()

    # 1. top-N by |weight| per bar
    rk = z.abs().rank(axis=1, ascending=False, method="first")
    z = z.where(rk <= max_positions, other=0.0)

    # 2. per-name exposure cap
    if max_weight_per_name is not None and max_weight_per_name < 1.0:
        z = z.clip(lower=-max_weight_per_name, upper=max_weight_per_name)

    # 3. gross-leverage cap
    gross = z.abs().sum(axis=1)
    scale = pd.Series(1.0, index=z.index)
    over = gross > gross_leverage
    scale[over] = gross_leverage / gross[over]
    z = z.mul(scale, axis=0)
    return z


# ── public execution entry points ────────────────────────────────────────

def realised_pod(
    positions: PositionsByStrategy,
    weights: Weights,
    panel: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> ExecutionResult:
    """Independent-pod model: fund return = Σ w_s · r_s^net (no netting)."""
    close = panel["close"]
    cost_rate = cost_rate_panel(panel, cfg)

    attribution: dict[str, pd.Series] = {}
    strategy_returns: dict[str, pd.Series] = {}
    gross_parts, cost_parts, turn_parts = [], [], []
    for sid, P in positions.items():
        w = weights.get(sid, 0.0)
        if w == 0.0 or P is None or P.empty:
            continue
        g, c, n, t = _net_from_positions(P, close, cost_rate)
        attribution[sid] = w * n
        strategy_returns[sid] = n          # standalone (unweighted) net return
        gross_parts.append(w * g)
        cost_parts.append(w * c)
        turn_parts.append(w * t)

    fund = _sum_series(list(attribution.values()))
    return ExecutionResult(
        fund_returns=fund,
        gross_returns=_sum_series(gross_parts),
        cost=_sum_series(cost_parts),
        turnover=_sum_series(turn_parts),
        attribution=attribution,
        strategy_returns=strategy_returns,
    )


def realised_netted(
    positions: PositionsByStrategy,
    weights: Weights,
    panel: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> ExecutionResult:
    """Consolidated netted book model (default)."""
    close = panel["close"]
    cost_rate = cost_rate_panel(panel, cfg)

    book = consolidate_book(positions, weights)
    book = apply_fund_constraints(
        book, cfg.fund_max_positions, cfg.max_weight_per_name, cfg.gross_leverage,
    )
    if book.empty:
        empty = pd.Series(dtype=float, name="fund_return")
        return ExecutionResult(empty, empty, empty, empty, {}, book=book)

    gross, cost, net, turnover = _net_from_positions(book, close, cost_rate)

    # Per-strategy attribution + the netting benefit vs. the pod model.
    pod = realised_pod(positions, weights, panel, cfg)
    netting_benefit = net.subtract(pod.fund_returns.reindex(net.index).fillna(0.0))

    net.name = "fund_return"
    return ExecutionResult(
        fund_returns=net,
        gross_returns=gross,
        cost=cost,
        turnover=turnover,
        attribution=pod.attribution,   # standalone weighted contributions (additive)
        strategy_returns=pod.strategy_returns,
        book=book,
        netting_benefit=netting_benefit,
        extra={
            "gross_exposure_mean": float(book.abs().sum(axis=1).mean()),
            "net_exposure_mean": float(book.sum(axis=1).mean()),
            "n_names_mean": float((book.abs() > 1e-9).sum(axis=1).mean()),
        },
    )


def run_execution(
    positions: PositionsByStrategy,
    weights: Weights,
    panel: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> ExecutionResult:
    """Dispatch to the configured execution model."""
    if cfg.execution_model == ExecutionModel.POD:
        return realised_pod(positions, weights, panel, cfg)
    return realised_netted(positions, weights, panel, cfg)


# ── helpers ──────────────────────────────────────────────────────────────

def _sum_series(parts: list[pd.Series]) -> pd.Series:
    """Sum a list of per-bar Series aligned on the union of their indices."""
    if not parts:
        return pd.Series(dtype=float, name="fund_return")
    out = parts[0].copy()
    for s in parts[1:]:
        out = out.add(s, fill_value=0.0)
    out.name = "fund_return"
    return out
