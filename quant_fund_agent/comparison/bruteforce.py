"""Track 2 — brute-force ML on the researched factors (no agents).

Feed *only* a prerun's researched factors straight into the model catalog and an
equal-weight ensemble, fit on the in-sample window and evaluate on the held-out
tail.  This isolates "are these factors good raw features?" from whether the
downstream agents happen to exploit them well.

Per-model rows reuse :func:`modeling.service.evaluate_config` (the same IS-fit /
OOS-reload path the notebook's horizon sweep uses).  The ensemble fits each base
model, reloads it, averages the base models' cross-sectionally-standardised
predicted signals, and backtests that averaged signal — all on the identical
IS/OOS split, so every number is comparable across preruns and models.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from quant_fund_agent.strategies.base import BaseStrategy

log = logging.getLogger("comparison.bruteforce")


class _PrecomputedSignalStrategy(BaseStrategy):
    """Backtest a signal frame that's already been computed (the ensemble mean)."""

    def __init__(self, signal: pd.DataFrame, holding_period: int, max_positions: int) -> None:
        self.strategy_id = "ensemble"
        self.name = "ensemble"
        self.description = ""
        self.factor_ids = []
        self.holding_period = holding_period
        self.max_positions = max_positions
        self.equal_weight = False
        self.min_conviction = 0.0
        self._signal = signal

    def calc(self, factor_signals, data):  # noqa: D401 — returns the stored signal
        close = data["close"]
        return self._signal.reindex(index=close.index, columns=close.columns)


def evaluate_prerun_models(
    prerun: str, factor_ids: list[str], cfg,
) -> list[dict[str, Any]]:
    """One brute-force KPI row per catalog model (+ the ensemble)."""
    from quant_fund_agent.modeling.service import evaluate_config

    rows: list[dict[str, Any]] = []
    models = cfg.resolved_models()
    log.info("brute-force: prerun '%s' — %d factors × %d models", prerun, len(factor_ids), len(models))
    for model in models:
        log.info("brute-force: prerun '%s' fitting '%s' ...", prerun, model)
        try:
            r = evaluate_config(
                factor_ids=factor_ids,
                model_type=model,
                model_params=cfg.fast_model_params(model),
                target_horizon=cfg.target_horizon,
                holding_period=cfg.holding_period,
                max_positions=cfg.max_positions,
                oos_split_ratio=cfg.oos_split_ratio,
                strategy_id=f"cmp_{prerun}_{model}",
                train_sample_frac=cfg.train_sample_frac,
            )
        except Exception as e:  # noqa: BLE001 — one model must not abort the sweep
            log.warning("brute-force %s/%s failed: %s", prerun, model, e)
            rows.append({"prerun": prerun, "model": model,
                         "n_factors_used": len(factor_ids), "error": str(e)[:200]})
            continue
        rows.append({
            "prerun": prerun, "model": model, "n_factors_used": len(factor_ids),
            "valid_ic": r.get("valid_ic"),
            "is_ic": r.get("is_ic"), "oos_ic": r.get("oos_ic"),
            "is_sharpe": r.get("is_sharpe"), "oos_sharpe": r.get("oos_sharpe"),
            "is_ann_return": r.get("is_ann_return"), "oos_ann_return": r.get("oos_ann_return"),
            "oos_max_drawdown": r.get("oos_max_drawdown"),
        })

    if cfg.include_ensemble:
        log.info("brute-force: prerun '%s' building ensemble ...", prerun)
        ens = _evaluate_ensemble(prerun, factor_ids, cfg)
        if ens is not None:
            rows.append(ens)
    return rows


def _evaluate_ensemble(
    prerun: str, factor_ids: list[str], cfg, base_models: list[str] | None = None,
) -> dict[str, Any] | None:
    """Equal-weight ensemble of the base models' predicted OOS signals."""
    from quant_fund_agent.backtesting.data_split import split_panel, split_signals
    from quant_fund_agent.backtesting.strategy_backtester import (
        backtest_strategy,
        normalise_factor_signals,
    )
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.modeling.service import _factor_signal, _load_panel
    from quant_fund_agent.modeling.train import fit_and_backtest
    from quant_fund_agent.strategies.model_strategy import ModelStrategy

    discover_factors()
    base_models = base_models or [m for m in cfg.resolved_models() if m != "ensemble"]
    if len(base_models) < 2:
        return None

    data_full = _load_panel()
    signals_full = {fid: _factor_signal(fid, data_full) for fid in factor_ids}
    data_is, data_oos = split_panel(data_full, cfg.oos_split_ratio)
    signals_is, signals_oos = split_signals(signals_full, cfg.oos_split_ratio)
    hp = cfg.holding_period or cfg.target_horizon

    from pathlib import Path

    from quant_fund_agent.modeling.train import DEFAULT_ARTIFACT_DIR

    strats: list[ModelStrategy] = []
    for model in base_models:
        # Reuse the cmp_ artifact just saved by evaluate_prerun_models — same data,
        # same split, same factors — avoids refitting the same model a second time.
        cmp_path = Path(DEFAULT_ARTIFACT_DIR) / f"cmp_{prerun}_{model}.joblib"
        if cmp_path.exists():
            try:
                strats.append(ModelStrategy.from_artifact(
                    cmp_path, strategy_id=f"ens_{prerun}_{model}",
                    holding_period=hp, max_positions=cfg.max_positions,
                ))
                log.info("ensemble: reusing cmp_%s_%s artifact", prerun, model)
                continue
            except Exception as e:  # noqa: BLE001
                log.debug("could not reuse cmp_ artifact for %s/%s: %s", model, prerun, e)
        try:
            fit = fit_and_backtest(
                factor_signals_is=signals_is, data_is=data_is, model_type=model,
                model_params=cfg.fast_model_params(model),
                factor_ids=factor_ids, target_horizon=cfg.target_horizon,
                holding_period=hp, max_positions=cfg.max_positions,
                strategy_id=f"ens_{prerun}_{model}",
                train_sample_frac=cfg.train_sample_frac,
            )
            strats.append(ModelStrategy.from_artifact(
                fit["artifact_path"], strategy_id=f"ens_{prerun}_{model}",
                holding_period=hp, max_positions=cfg.max_positions,
            ))
        except Exception as e:  # noqa: BLE001
            log.warning("ensemble base %s failed for %s: %s", model, prerun, e)
    if len(strats) < 2:
        return None

    def _ensemble_signal(signals, data) -> pd.DataFrame:
        normed = normalise_factor_signals(signals)
        preds = []
        for s in strats:
            p = s.calc(normed, data)
            # cross-sectionally standardise per timestamp so no single model
            # (with larger-variance predictions) dominates the average.
            p = p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, pd.NA), axis=0)
            preds.append(p)
        return sum(preds) / len(preds)

    try:
        oos = backtest_strategy(
            _PrecomputedSignalStrategy(_ensemble_signal(signals_oos, data_oos), hp, cfg.max_positions),
            signals_oos, data_oos,
        ).metrics
        is_ = backtest_strategy(
            _PrecomputedSignalStrategy(_ensemble_signal(signals_is, data_is), hp, cfg.max_positions),
            signals_is, data_is,
        ).metrics
    except Exception as e:  # noqa: BLE001
        log.warning("ensemble backtest failed for %s: %s", prerun, e)
        return None

    return {
        "prerun": prerun, "model": "ensemble", "n_factors_used": len(factor_ids),
        "is_ic": is_.ic_mean, "oos_ic": oos.ic_mean,
        "is_sharpe": is_.sharpe_ratio, "oos_sharpe": oos.sharpe_ratio,
        "is_ann_return": is_.annualised_return, "oos_ann_return": oos.annualised_return,
        "oos_max_drawdown": oos.max_drawdown,
        "ensemble_members": ",".join(s.model_params["model_type"] for s in strats),
    }
