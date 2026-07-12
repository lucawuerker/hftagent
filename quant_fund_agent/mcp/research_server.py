"""MCP server exposing the Factor Researcher's deterministic toolbox.

Run as a stdio server::

    python -m quant_fund_agent.mcp.research_server

It owns the (heavy) factor panel server-side and caches it, so materialising a
candidate and IC-backtesting it reuse a single loaded panel.  Only small JSON
crosses the MCP boundary.

All real work is delegated to :mod:`quant_fund_agent.mcp.research_service`, so
the in-process fallback in ``research_client.py`` computes identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.mcp import research_service as svc

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.research_server")

mcp = FastMCP("quant-research")


@mcp.tool()
def load_papers(
    n: int = 2,
    cutoff_date: str | None = None,
    strategy: str = "unread_first",
    max_chars: int = 200_000,
) -> str:
    """Select up to ``n`` papers (published before ``cutoff_date`` if given) and
    extract their text.  Returns a JSON array of snippet objects.
    """
    return json.dumps(svc.load_papers(
        n=n, cutoff_date=cutoff_date, strategy=strategy, max_chars=max_chars,
    ))


@mcp.tool()
def existing_factor_ids(scope: str = "package") -> str:
    """Return the factor ids a new brainstorm should avoid (JSON array).

    ``scope="package"`` → registered classes ∪ active DB; ``"prerun"`` → active DB only.
    """
    return json.dumps(svc.existing_factor_ids(scope=scope))


@mcp.tool()
def materialise_factor(
    factor_id: str,
    code: str,
    expected_prediction_horizon: int | None = None,
) -> str:
    """Validate / write / import / smoke-test generated factor code.

    Returns JSON ``{"ok": bool, "code_path"?: str, "error"?: str}``.
    """
    return json.dumps(svc.materialise_factor(
        factor_id, code, expected_prediction_horizon))


@mcp.tool()
def backtest_factors(
    factor_ids: list[str],
    horizon: int = 6,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
) -> str:
    """Run the single-factor IC backtest for each factor (JSON keyed by id)."""
    return json.dumps(svc.backtest_factors(
        factor_ids=factor_ids, horizon=horizon, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers,
    ))


@mcp.tool()
def evaluate_fitness(
    candidate: dict,
    book: list[dict] | None = None,
    jitter: list[dict] | None = None,
    reference: list[dict] | None = None,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    cpcv_groups: int = 6,
    cpcv_k: int = 2,
    embargo: int = 0,
    cpcv_model: str | None = None,
    cpcv_fast: bool = True,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    independence_metric: str = "residual_ic",
    regime_kind: str = "drawdown",
    regime_quantile: float = 0.2,
    marginal_model: str = "gradient_boosting",
    gate_turnover: float | None = None,
    cost_rate: float = 5e-4,
    perturbation_weight: float = 0.0,
    perturbation_sigma: float = 0.5,
    cost_executor: dict | None = None,
) -> str:
    """Deterministically score one candidate factor program against the book
    (in-memory compile → research_eval harness).  Returns JSON
    ``{"ok": bool, "fitness"?: {...}, "error"?: str}``.
    """
    return json.dumps(svc.evaluate_fitness(
        candidate, book, jitter, reference,
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        n_trials=n_trials, cpcv_groups=cpcv_groups, cpcv_k=cpcv_k,
        embargo=embargo, cpcv_model=cpcv_model, cpcv_fast=cpcv_fast,
        cutoff_date=cutoff_date, data_dir=data_dir, n_tickers=n_tickers, fields=fields,
        independence_metric=independence_metric, regime_kind=regime_kind,
        regime_quantile=regime_quantile, marginal_model=marginal_model,
        gate_turnover=gate_turnover, cost_rate=cost_rate,
        perturbation_weight=perturbation_weight, perturbation_sigma=perturbation_sigma,
        cost_executor=cost_executor,
    ))


@mcp.tool()
def evaluate_set_fitness(
    programs: list[dict],
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    cpcv_groups: int = 6,
    cpcv_k: int = 2,
    embargo: int = 0,
    cpcv_model: str | None = None,
    cpcv_fast: bool = True,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    candidate_id: str = "set",
    regime_kind: str = "drawdown",
    regime_quantile: float = 0.2,
    marginal_model: str = "gradient_boosting",
    gate_turnover: float | None = None,
    cost_rate: float = 5e-4,
    perturbation_weight: float = 0.0,
    perturbation_sigma: float = 0.5,
    cost_executor: dict | None = None,
) -> str:
    """SET mode: score a whole factor set jointly (its own combined-model OOS
    IC is the primary axis).  Returns JSON ``{"ok": bool, "fitness"?: {...}}``.
    """
    return json.dumps(svc.evaluate_set_fitness(
        programs, target_horizon=target_horizon, is_frac=is_frac,
        val_frac=val_frac, n_trials=n_trials, cpcv_groups=cpcv_groups,
        cpcv_k=cpcv_k, embargo=embargo, cpcv_model=cpcv_model,
        cpcv_fast=cpcv_fast, cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields,
        candidate_id=candidate_id, regime_kind=regime_kind,
        regime_quantile=regime_quantile, marginal_model=marginal_model,
        gate_turnover=gate_turnover, cost_rate=cost_rate,
        perturbation_weight=perturbation_weight, perturbation_sigma=perturbation_sigma,
        cost_executor=cost_executor,
    ))


@mcp.tool()
def score_book_oos(
    book: list[dict],
    start: str,
    end: str | None = None,
    target_horizon: int = 6,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
) -> str:
    """Touch-once OOS scoring of a factor book on [start, end): combined-model
    OOS IC + per-factor ICs + CSCV PBO.  Returns JSON."""
    return json.dumps(svc.score_book_oos(
        book, start=start, end=end, target_horizon=target_horizon,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields,
    ))


@mcp.tool()
def curate_book(
    book: list[dict],
    mode: str = "greedy",
    n_keep: int | None = None,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    marginal_model: str = "gradient_boosting",
    en_threshold: float = 0.5,
    en_l1_ratio: float = 0.5,
    seed: int = 0,
) -> str:
    """Curate a pool of gate-passing factors into the final book (two-stage
    Lever 2): ``greedy`` forward-selection or ``elastic_net`` stability
    selection, ``n_keep`` optional.  Returns JSON
    ``{"ok": bool, "kept_factor_ids": [...], ...}``."""
    return json.dumps(svc.curate_book(
        book, mode=mode, n_keep=n_keep, target_horizon=target_horizon,
        is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields,
        marginal_model=marginal_model, en_threshold=en_threshold,
        en_l1_ratio=en_l1_ratio, seed=seed,
    ))


@mcp.tool()
def publish_book(
    book: list[dict],
    n_trials: int = 1,
    mode: str = "on",
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    marginal_model: str = "gradient_boosting",
) -> str:
    """Selection-time deflation publish filter (WS1): deflate the book's combined
    OOS IC for ``n_trials`` and, in ``mode='on'``, narrow the book (pruning by
    marginal contribution) to what beats selection luck.  Returns JSON
    ``{"ok": bool, "kept_factor_ids": [...], "passed": bool, "deflated": {...}}``."""
    return json.dumps(svc.publish_book(
        book, n_trials=n_trials, mode=mode, target_horizon=target_horizon,
        is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields,
        marginal_model=marginal_model,
    ))


@mcp.tool()
def persist_results(kept_records: list[dict], rejected: list[dict]) -> str:
    """Persist survivor FactorRecords and purge rejects.  Returns JSON
    ``{"kept_factor_ids": [...], "rejected_factor_ids": [...]}``.
    """
    return json.dumps(svc.persist_results(kept_records, rejected))


@mcp.tool()
def freeze_signals(
    book: list[dict],
    out_dir: str,
    version: int = 1,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    specs: list[dict] | None = None,
) -> str:
    """Freeze the K evaluation signals from a factor book (the execution arm's
    interface artifact).  Returns JSON
    ``{"ok": bool, "manifest_path"?: str, "manifest"?: {...}, "error"?: str}``.
    """
    return json.dumps(svc.freeze_signals(
        book, out_dir=out_dir, version=version, target_horizon=target_horizon,
        is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields, specs=specs,
    ))


@mcp.tool()
def evaluate_executor_fitness(
    candidate: dict,
    signals_manifest: str,
    jitter: list[dict] | None = None,
    archive: list[dict] | None = None,
    n_trials: int = 1,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    cost_rate: float = 5e-4,
    lambda_dispersion: float = 0.5,
    gate_turnover: float | None = None,
    gate_degradation: float = 0.5,
    min_activity: float = 0.05,
    selection_deflation: str = "off",
) -> str:
    """Deterministically score one executor program against the frozen
    evaluation signals (in-memory compile → exec harness).  Returns JSON
    ``{"ok": bool, "fitness"?: {...}, "error"?: str}``.
    """
    return json.dumps(svc.evaluate_executor_fitness(
        candidate, signals_manifest, jitter, archive,
        n_trials=n_trials, is_frac=is_frac, val_frac=val_frac,
        cutoff_date=cutoff_date, data_dir=data_dir, n_tickers=n_tickers,
        fields=fields, cost_rate=cost_rate, lambda_dispersion=lambda_dispersion,
        gate_turnover=gate_turnover, gate_degradation=gate_degradation,
        min_activity=min_activity, selection_deflation=selection_deflation,
    ))


@mcp.tool()
def score_joint_state(
    book: list[dict],
    executor: dict | None = None,
    n_joint_looks: int = 1,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    model: str = "ridge",
    cost_rate: float = 5e-4,
    baseline_executor_id: str = "zscore_threshold_equal_weight",
) -> str:
    """The joint objective J: deflated net VAL Sharpe of book → SOTA executor →
    cost layer (block-boundary scoring for the joint outer layer).  Returns JSON
    ``{"ok": bool, "J"?: float, ...}``.
    """
    return json.dumps(svc.score_joint_state(
        book, executor, n_joint_looks=n_joint_looks,
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        cutoff_date=cutoff_date, data_dir=data_dir, n_tickers=n_tickers,
        fields=fields, model=model, cost_rate=cost_rate,
        baseline_executor_id=baseline_executor_id,
    ))


@mcp.tool()
def score_joint_oos(
    book: list[dict],
    executor: dict | None = None,
    start: str = "",
    end: str | None = None,
    target_horizon: int = 6,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    model: str = "ridge",
    cost_rate: float = 5e-4,
    baseline_executor_id: str = "zscore_threshold_equal_weight",
) -> str:
    """Touch-once OOS scoring of a (book, executor) pair on [start, end)
    (the J4 walk-forward's per-fold scorer).  Returns JSON."""
    return json.dumps(svc.score_joint_oos(
        book, executor, start=start, end=end, target_horizon=target_horizon,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields, model=model,
        cost_rate=cost_rate, baseline_executor_id=baseline_executor_id,
    ))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
