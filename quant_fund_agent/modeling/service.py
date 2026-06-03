"""Data orchestration shared by the MCP modeling server and its fallback.

The MCP server and the in-process fallback must compute *identical* numbers, so
the work that surrounds :func:`quant_fund_agent.modeling.train.fit_and_backtest`
— loading the panel, computing factor signals, holding out the OOS slice — lives
here in one place.  The server is then a thin protocol wrapper over these
functions, and the in-process client calls them directly.

The panel and per-factor signals are cached at module level so a long-lived
server process loads the (heavy) data once and reuses it across every architect
iteration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from quant_fund_agent.backtesting.data_split import split_panel, split_signals
from quant_fund_agent.modeling.catalog import STATIC_WEIGHTS, list_model_menu
from quant_fund_agent.modeling.train import fit_and_backtest

log = logging.getLogger("modeling.service")

DATA_DIR = os.getenv("DATA_DIR", "ticker_data")

_PANEL_CACHE: dict[str, Any] | None = None
_SIGNAL_CACHE: dict[str, Any] = {}


def _truncate_as_of(
    frames: dict[str, Any],
    as_of: str | None,
) -> dict[str, Any]:
    """Keep only rows strictly **before** ``as_of`` (exclusive upper bound).

    Used by the walk-forward backtest so the Architect / Statistician only see
    data available at a given point in time.  ``as_of`` is an ISO date/datetime
    string; ``None`` is a no-op (the full panel, current production behaviour).

    Slicing the (full) cached frames per call is cheap and keeps the heavy panel
    + factor signals computed once on the *whole* history — so warm-up bars at
    the start are correct and nothing after the cutoff can leak in.
    """
    if as_of is None:
        return frames
    import pandas as pd

    cut = pd.Timestamp(as_of)
    return {k: df.loc[df.index < cut] for k, df in frames.items()}


def _resolve_n_tickers() -> int | None:
    """Universe cap from ``ARCHITECT_N_TICKERS`` (shared with the architect)."""
    raw = os.getenv("ARCHITECT_N_TICKERS")
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


def _load_panel() -> dict[str, Any]:
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        from quant_fund_agent.backtesting.data_loader import load_panel
        n_tickers = _resolve_n_tickers()
        log.info("modeling.service loading panel from %s%s …", DATA_DIR,
                 f" (capped at {n_tickers} tickers)" if n_tickers else "")
        t0 = time.time()
        _PANEL_CACHE = load_panel(DATA_DIR, n_tickers=n_tickers)
        log.info("panel loaded in %.1fs (universe size: %d)", time.time() - t0,
                 next(iter(_PANEL_CACHE.values())).shape[1] if _PANEL_CACHE else 0)
    return _PANEL_CACHE


def _factor_signal(factor_id: str, data: dict[str, Any]):
    from quant_fund_agent.factors import instantiate_factor
    if factor_id not in _SIGNAL_CACHE:
        _SIGNAL_CACHE[factor_id] = instantiate_factor(factor_id).calc(data)
    return _SIGNAL_CACHE[factor_id]


def model_menu() -> list[dict[str, Any]]:
    """The model catalog shown to the Architect LLM."""
    return list_model_menu()


def run_fit_and_backtest(
    *,
    model_type: str,
    factor_ids: list[str] | None = None,
    model_params: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    target_horizon: int = 6,
    holding_period: int = 6,
    max_positions: int = 20,
    equal_weight: bool = False,
    min_conviction: float = 0.0,
    oos_split_ratio: float = 0.2,
    valid_ratio: float = 0.3,
    strategy_id: str | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Compute IS signals for the requested factors and fit + backtest one model.

    The held-out OOS slice (``oos_split_ratio``) is never passed to the model —
    it is reserved for the Statistician.

    ``as_of`` (walk-forward backtest only): an ISO date/datetime string.  When
    set, the panel + factor signals are truncated to rows strictly before it
    *before* the IS/OOS split, so the Architect never sees future data.
    """
    from quant_fund_agent.factors import discover_factors

    discover_factors()
    data_full = _load_panel()

    # Which factors do we need signals for?
    if model_type == STATIC_WEIGHTS:
        needed = list((weights or {}).keys())
    else:
        needed = list(factor_ids or [])
    if not needed:
        raise ValueError(f"No factors supplied for model_type={model_type!r}.")

    signals_full = {fid: _factor_signal(fid, data_full) for fid in needed}

    # Walk-forward cutoff: hide everything from the cutoff date onward.
    data_full = _truncate_as_of(data_full, as_of)
    signals_full = _truncate_as_of(signals_full, as_of)

    data_is, _ = split_panel(data_full, oos_split_ratio)
    signals_is, _ = split_signals(signals_full, oos_split_ratio)

    return fit_and_backtest(
        factor_signals_is=signals_is,
        data_is=data_is,
        model_type=model_type,
        model_params=model_params,
        factor_ids=factor_ids,
        weights=weights,
        target_horizon=target_horizon,
        holding_period=holding_period,
        max_positions=max_positions,
        equal_weight=equal_weight,
        min_conviction=min_conviction,
        valid_ratio=valid_ratio,
        strategy_id=strategy_id,
    )


def evaluate_config(
    *,
    factor_ids: list[str],
    model_type: str = "ridge",
    model_params: dict[str, Any] | None = None,
    target_horizon: int = 6,
    holding_period: int | None = None,
    max_positions: int = 20,
    equal_weight: bool = False,
    min_conviction: float = 0.0,
    oos_split_ratio: float = 0.2,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    """Fit one (model, horizon) config on IS and evaluate IS **and** OOS.

    Deterministic and LLM-free — used by the notebook's horizon sweep.  Holds the
    position over the forecast horizon (``holding_period`` defaults to
    ``target_horizon``) and reloads the persisted full-IS model for the OOS leg
    (never refits on OOS).
    """
    from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.strategies.model_strategy import ModelStrategy

    discover_factors()
    data_full = _load_panel()
    signals_full = {fid: _factor_signal(fid, data_full) for fid in factor_ids}
    data_is, data_oos = split_panel(data_full, oos_split_ratio)
    signals_is, signals_oos = split_signals(signals_full, oos_split_ratio)

    hp = holding_period or target_horizon
    fit = fit_and_backtest(
        factor_signals_is=signals_is, data_is=data_is,
        model_type=model_type, model_params=model_params, factor_ids=factor_ids,
        target_horizon=target_horizon, holding_period=hp,
        max_positions=max_positions, equal_weight=equal_weight,
        min_conviction=min_conviction,
        strategy_id=strategy_id or f"sweep_{model_type}_h{target_horizon}",
    )

    strat = ModelStrategy.from_artifact(
        fit["artifact_path"],
        strategy_id=f"{strategy_id or model_type}_h{target_horizon}_oos",
        holding_period=hp, max_positions=max_positions,
        equal_weight=equal_weight, min_conviction=min_conviction,
    )
    oos = backtest_strategy(strat, signals_oos, data_oos).metrics
    m, diag = fit["metrics"], fit["diagnostics"]
    return {
        "target_horizon": target_horizon,
        "model_type": model_type,
        "valid_ic": diag.get("valid_ic"),
        "is_ic": m.get("ic_mean"),
        "oos_ic": oos.ic_mean,
        "is_sharpe": m.get("sharpe_ratio"),
        "oos_sharpe": oos.sharpe_ratio,
        "is_ann_return": m.get("annualised_return"),
        "oos_ann_return": oos.annualised_return,
        "oos_max_drawdown": oos.max_drawdown,
        "artifact_path": fit["artifact_path"],
    }


def sweep_horizons(
    *,
    factor_ids: list[str],
    model_type: str = "ridge",
    model_params: dict[str, Any] | None = None,
    horizons: tuple[int, ...] = (1, 6, 60),
    oos_split_ratio: float = 0.2,
) -> list[dict[str, Any]]:
    """Evaluate a fixed model/feature set across forecast horizons (10s/1m/10m).

    Returns one KPI row per horizon — the input to the notebook's regime
    comparison that shows the IC-vs-Sharpe tradeoff.
    """
    return [
        evaluate_config(
            factor_ids=factor_ids, model_type=model_type,
            model_params=model_params, target_horizon=h, oos_split_ratio=oos_split_ratio,
        )
        for h in horizons
    ]
