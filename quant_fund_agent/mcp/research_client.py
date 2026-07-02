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


def materialise_factor(factor_id: str, code: str) -> dict[str, Any]:
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.materialise_factor(factor_id, code)
    return _bridge().call("materialise_factor", {"factor_id": factor_id, "code": code})


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
    *,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    cpcv_groups: int = 6,
    cpcv_k: int = 2,
    embargo: int = 0,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Deterministic evolutionary fitness of one candidate program vs the book.

    See :func:`quant_fund_agent.mcp.research_service.evaluate_fitness`.
    """
    kwargs: dict[str, Any] = dict(
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        n_trials=n_trials, cpcv_groups=cpcv_groups, cpcv_k=cpcv_k,
        embargo=embargo, cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields,
    )
    if not _use_mcp():
        from quant_fund_agent.mcp import research_service as svc
        return svc.evaluate_fitness(candidate, book, jitter, **kwargs)
    args = {"candidate": candidate, "book": book or [], "jitter": jitter or []}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return _bridge().call("evaluate_fitness", args)


def evaluate_set_fitness(
    programs: list[dict[str, Any]],
    *,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    n_trials: int = 1,
    cpcv_groups: int = 6,
    cpcv_k: int = 2,
    embargo: int = 0,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    candidate_id: str = "set",
) -> dict[str, Any]:
    """SET-mode fitness of a whole factor set (one genome).

    See :func:`quant_fund_agent.mcp.research_service.evaluate_set_fitness`.
    """
    kwargs: dict[str, Any] = dict(
        target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
        n_trials=n_trials, cpcv_groups=cpcv_groups, cpcv_k=cpcv_k,
        embargo=embargo, cutoff_date=cutoff_date, data_dir=data_dir,
        n_tickers=n_tickers, fields=fields, candidate_id=candidate_id,
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
