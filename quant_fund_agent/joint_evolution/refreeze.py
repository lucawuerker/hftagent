"""The block-boundary re-freeze protocol (J1).

* **After a factor block** — the factor arm's accepted book changed, so the
  interface artifact the exec arm scores against must be re-materialised:
  new book view → ``freeze_eval_signals`` version k+1 (IS-only fits,
  poison-audited — a failed audit ABORTS the run rather than silently
  laundering look-ahead into every executor score) → the exec archive is
  deterministically **re-scored** against the new set (billing the joint
  ledger's look count, never ``n_exec``).
* **After an exec block** — the SOTA-executor view is refreshed for the
  factor arm's coupling seam (J3) and the joint objective.
"""

from __future__ import annotations

import logging
from typing import Any

from quant_fund_agent.joint_evolution.ledger import TrialsLedger
from quant_fund_agent.joint_evolution.state import SOTAState

log = logging.getLogger("joint_evolution.refreeze")


class FrozenSignalAuditError(RuntimeError):
    """A re-frozen evaluation signal failed its poison audit — hard abort."""


def refreeze_after_factor_block(
    sota: SOTAState,
    factor_controller: Any,
    exec_loop: Any | None,
    ledger: TrialsLedger,
    *,
    out_dir: str,
    target_horizon: int = 6,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
    fields: list[str] | None = None,
    specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Book view → freeze v(k+1) → re-score the exec archive.  Returns a report.

    The book view = the factor controller's **accepted book** (gate-passing
    Pareto archive).  An empty archive keeps the previous freeze (reported,
    never silent).
    """
    from quant_fund_agent.mcp import research_client

    book = [{"factor_id": fid, "code": code}
            for fid, code in factor_controller.archive_programs()]
    if not book:
        log.warning("factor archive is empty — keeping frozen signals v%d",
                    sota.frozen_signals_version)
        return {"refrozen": False, "reason": "empty factor archive",
                "version": sota.frozen_signals_version}

    version = sota.frozen_signals_version + 1
    out = research_client.freeze_signals(
        book, out_dir=out_dir, version=version, target_horizon=target_horizon,
        is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers, fields=fields, specs=specs)
    if not out.get("ok"):
        raise RuntimeError(f"re-freeze v{version} failed: {out.get('error')}")
    audit = out["manifest"].get("poison_audit", {})
    if audit.get("passed") is False:
        raise FrozenSignalAuditError(
            f"frozen signals v{version} FAILED the poison audit — a leaky "
            "evaluation signal would launder look-ahead into every executor "
            "score; refusing to continue")

    sota.set_book(book)
    sota.frozen_signals_version = version
    sota.frozen_signals_manifest = out["manifest_path"]

    report: dict[str, Any] = {"refrozen": True, "version": version,
                              "k": out["manifest"]["k"],
                              "book_size": len(book), "rescore": None}
    if exec_loop is not None and getattr(exec_loop.controller, "kept_pool", None):
        rescore = exec_loop.rescore_archive(out["manifest_path"])
        # re-scores are fresh VAL looks (NOT new hypotheses): joint count only
        ledger.bill("exec", rescore.get("n_looks", 0), rescore=True)
        report["rescore"] = rescore
    return report


def update_sota_executor(sota: SOTAState, exec_loop: Any) -> dict[str, Any]:
    """Refresh the frozen-SOTA executor view after an exec block."""
    best = exec_loop.sota_executor()
    changed = bool(best) and (
        sota.sota_executor is None
        or best["genome_id"] != sota.sota_executor.get("genome_id"))
    if best is not None:
        sota.sota_executor = best
    return {"updated": bool(best), "changed": changed,
            "executor_id": (best or {}).get("executor_id")}
