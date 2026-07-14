"""Synchronous client facade for the quant-research MCP server.

Set ``RESEARCH_USE_MCP=0`` (or the global ``QF_USE_MCP=0``) to call the
:mod:`quant_fund_agent.mcp.research_service` functions in-process instead.
"""

from __future__ import annotations

from typing import Any

from quant_fund_agent.mcp._bridge import MCPBridge, use_mcp

_SERVER_MODULE = "quant_fund_agent.mcp.research_server"


def _use_mcp() -> bool:
    return use_mcp("RESEARCH_USE_MCP")


def _bridge() -> MCPBridge:
    return MCPBridge.instance(_SERVER_MODULE)


def load_papers(
    n: int = 2,
    cutoff_date: str | None = None,
    strategy: str = "unread_first",
    max_chars: int = 200_000,
) -> list[dict[str, Any]]:
    args = dict(n=n, cutoff_date=cutoff_date, strategy=strategy, max_chars=max_chars)
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.load_papers(**args)
    return _bridge().call("load_papers", {k: v for k, v in args.items() if v is not None})


def existing_factor_ids(scope: str = "package") -> list[str]:
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.existing_factor_ids(scope=scope)
    return _bridge().call("existing_factor_ids", {"scope": scope})


def materialise_factor(
    factor_id: str,
    code: str,
    expected_prediction_horizon: int | None = None,
) -> dict[str, Any]:
    args = {
        "factor_id": factor_id,
        "code": code,
        "expected_prediction_horizon": expected_prediction_horizon,
    }
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.materialise_factor(**args)
    return _bridge().call(
        "materialise_factor", {k: v for k, v in args.items() if v is not None})


def backtest_factors(
    factor_ids: list[str],
    horizon: int = 6,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
) -> dict[str, Any]:
    args: dict[str, Any] = dict(
        factor_ids=factor_ids, horizon=horizon, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.backtest_factors(**args)
    return _bridge().call("backtest_factors", {k: v for k, v in args.items() if v is not None})


def persist_results(
    kept_records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, list[str]]:
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.persist_results(kept_records, rejected)
    return _bridge().call(
        "persist_results", {"kept_records": kept_records, "rejected": rejected},
    )


def evaluate_fitness(
    candidate: dict[str, Any],
    book: list[dict[str, Any]] | None = None,
    jitter: list[dict[str, Any]] | None = None,
    reference: list[dict[str, Any]] | None = None,
    *,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    stability_blocks: int = 4,
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
) -> dict[str, Any]:
    """Deterministic evolutionary fitness of one candidate program vs the book.

    See :func:`quant_fund_agent.mcp.research_service.evaluate_fitness`.
    """
    kwargs: dict[str, Any] = dict(
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        n_trials=n_trials, stability_blocks=stability_blocks,
        cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields,
        independence_metric=independence_metric, regime_kind=regime_kind,
        regime_quantile=regime_quantile, marginal_model=marginal_model,
        gate_turnover=gate_turnover, cost_rate=cost_rate,
        perturbation_weight=perturbation_weight, perturbation_sigma=perturbation_sigma,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.evaluate_fitness(candidate, book, jitter, reference, **kwargs)
    args = {"candidate": candidate, "book": book or [], "jitter": jitter or [],
            "reference": reference or []}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("evaluate_fitness", args)


def evaluate_set_fitness(
    programs: list[dict[str, Any]],
    *,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    stability_blocks: int = 4,
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
) -> dict[str, Any]:
    """SET-mode fitness of a whole factor set (one genome).

    See :func:`quant_fund_agent.mcp.research_service.evaluate_set_fitness`.
    """
    kwargs: dict[str, Any] = dict(
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        n_trials=n_trials, stability_blocks=stability_blocks,
        cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields, candidate_id=candidate_id,
        regime_kind=regime_kind, regime_quantile=regime_quantile,
        marginal_model=marginal_model,
        gate_turnover=gate_turnover, cost_rate=cost_rate,
        perturbation_weight=perturbation_weight, perturbation_sigma=perturbation_sigma,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.evaluate_set_fitness(programs, **kwargs)
    args: dict[str, Any] = {"programs": programs}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("evaluate_set_fitness", args)


def score_book_oos(
    book: list[dict[str, Any]],
    *,
    start: str,
    end: str | None = None,
    target_horizon: int = 6,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Touch-once OOS scoring of a factor book on ``[start, end)``.

    See :func:`quant_fund_agent.mcp.research_service.score_book_oos`.
    """
    kwargs: dict[str, Any] = dict(
        start=start, end=end, target_horizon=target_horizon,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.score_book_oos(book, **kwargs)
    args: dict[str, Any] = {"book": book}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("score_book_oos", args)


def curate_book(
    book: list[dict[str, Any]],
    *,
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
) -> dict[str, Any]:
    """Curate a pool of gate-passing factors into the final book (Lever 2).

    See :func:`quant_fund_agent.mcp.research_service.curate_book`.
    """
    kwargs: dict[str, Any] = dict(
        mode=mode, n_keep=n_keep, target_horizon=target_horizon, is_frac=is_frac,
        val_frac=val_frac, cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields, marginal_model=marginal_model,
        en_threshold=en_threshold, en_l1_ratio=en_l1_ratio, seed=seed,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.curate_book(book, **kwargs)
    args: dict[str, Any] = {"book": book}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("curate_book", args)


def publish_book(
    book: list[dict[str, Any]],
    *,
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
) -> dict[str, Any]:
    """Apply the selection-time deflation *publish* filter to a final book (WS1).

    See :func:`quant_fund_agent.mcp.research_service.publish_book`.
    """
    kwargs: dict[str, Any] = dict(
        n_trials=n_trials, mode=mode, target_horizon=target_horizon, is_frac=is_frac,
        val_frac=val_frac, cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields, marginal_model=marginal_model,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.publish_book(book, **kwargs)
    args: dict[str, Any] = {"book": book}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("publish_book", args)
