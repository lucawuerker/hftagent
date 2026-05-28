"""Multi-PM committee orchestrator.

When you want more than one PM running side-by-side and you want them
to *converge on a single allocation* (rather than each running its own
sleeve), use ``run_pm_committee``.

The committee:

1. Runs every PM in **propose-only** mode (no DB mutations).  Each PM
   produces a ``PortfolioRecord`` proposal containing its picks, its
   weights, its monitor recommendations (flag / retire), and its
   rationale.
2. Aggregates the proposals via one of three voting methods:
     * ``SIMPLE_AVERAGE``   – mean weight per strategy across PMs.
     * ``WEIGHTED_AVERAGE`` – mean weighted by ``pm_weights``.
     * ``LLM_MODERATOR``    – one LLM call synthesises the final
       allocation from all proposals + rationales.
3. Resolves monitor consensus: a strategy is flagged / retired when at
   least ``flag_threshold`` of the PMs vote that way.
4. Materialises the consensus once — exactly one ``PortfolioRecord`` is
   written and exactly one wave of ``pm_status`` updates is applied.

Single-PM passthrough
---------------------
If exactly one PM is passed, the committee short-circuits and invokes
that PM with its original ``propose_only`` value (typically False), so
single-PM behaviour is byte-for-byte identical to calling
``portfolio_manager_graph.invoke(state)`` directly.

Why a separate module
---------------------
The PM agent's job is *one* portfolio.  Committee aggregation is a
strictly higher concern: it composes PMs without recursing into the
graph.  Keeping the two separate means
``portfolio_manager_graph.invoke(state)`` is exactly what it was
before, and the committee adds zero overhead to single-PM runs.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from quant_fund_agent.agents.portfolio_manager.graph import portfolio_manager_graph
from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState
from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
from quant_fund_agent.portfolio import (
    expected_portfolio_metrics,
    get_personality_profile,
)
from quant_fund_agent.schemas import (
    ConstructionMethod,
    PMMode,
    PMPersonality,
    PMStatus,
    PortfolioAllocation,
    PortfolioRecord,
    VotingMethod,
)

log = logging.getLogger("portfolio_manager.committee")

LLM_MODEL = os.getenv("PM_LLM_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CommitteeConfig:
    """How to combine multiple PMs' proposals.

    Attributes
    ----------
    voting_method
        Aggregation algorithm — see ``VotingMethod`` docstring.
    pm_weights
        Per-PM weights (keyed by ``pm_name``) for weighted-average voting.
        Missing PMs default to 1.0.  Normalised internally.
    flag_threshold
        Fraction of PMs (in [0, 1]) that must vote to flag a strategy
        for it to actually be flagged.  0.5 = simple majority.
    retire_threshold
        Same, for retire votes.  Defaults to 0.5 as well; raise it for
        a more cautious committee (e.g. 0.99 → requires unanimous).
    pm_name
        Name of the committee in the audit log.
    """
    voting_method: VotingMethod = VotingMethod.SIMPLE_AVERAGE
    pm_weights: dict[str, float] = field(default_factory=dict)
    flag_threshold: float = 0.5
    retire_threshold: float = 0.5
    pm_name: str = "committee"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pm_committee(
    pm_states: list[PortfolioManagerState],
    config: CommitteeConfig | None = None,
    strategy_db: StrategyDatabase | None = None,
    portfolio_db: PortfolioDatabase | None = None,
) -> PortfolioRecord:
    """Invoke a committee of PMs and return the consensus PortfolioRecord.

    Parameters
    ----------
    pm_states
        One ``PortfolioManagerState`` per PM.  Must all share the same
        ``strategy_db`` / ``portfolio_db``, or pass them explicitly.
    config
        Aggregation config.  Defaults to simple-average.
    strategy_db, portfolio_db
        Optional explicit DBs.  If omitted, taken from the first PM.

    Returns
    -------
    PortfolioRecord
        The committee's final, materialised allocation.  Always also
        persisted into ``portfolio_db``.
    """
    if not pm_states:
        raise ValueError("run_pm_committee: need at least one PM state")

    config = config or CommitteeConfig()
    strategy_db = strategy_db or pm_states[0].strategy_db
    portfolio_db = portfolio_db or pm_states[0].portfolio_db

    # ── Single-PM passthrough ───────────────────────────────────────
    # Preserve byte-for-byte single-PM behaviour: no propose_only
    # rewriting, no aggregation, no extra PortfolioRecord.
    if len(pm_states) == 1:
        only = pm_states[0]
        only.strategy_db = strategy_db
        only.portfolio_db = portfolio_db
        result = portfolio_manager_graph.invoke(only)
        return result["portfolio_record"]

    # ── Committee path ───────────────────────────────────────────────
    log.info(
        "[%s] Running %d-PM committee (voting=%s, retire_thr=%.2f, flag_thr=%.2f)",
        config.pm_name, len(pm_states), config.voting_method.value,
        config.retire_threshold, config.flag_threshold,
    )

    proposals: list[PortfolioRecord] = []
    for state in pm_states:
        # Deep-copy so we don't accidentally mutate the caller's state,
        # and force propose_only so this PM doesn't touch any DB.
        proposal_state = copy.copy(state)
        proposal_state.strategy_db = strategy_db
        proposal_state.portfolio_db = portfolio_db
        proposal_state.propose_only = True
        log.info("[%s] Collecting proposal from PM '%s' …",
                 config.pm_name, state.pm_name)
        out = portfolio_manager_graph.invoke(proposal_state)
        proposals.append(out["portfolio_record"])

    # Aggregate.
    if config.voting_method == VotingMethod.LLM_MODERATOR:
        consensus_weights, moderator_text, chosen_method = _llm_moderate(
            proposals, strategy_db,
        )
    else:
        weights_per_pm = _per_pm_weights(proposals, config)
        consensus_weights = _aggregate_weights(proposals, weights_per_pm)
        moderator_text = (
            f"{config.voting_method.value} of {len(proposals)} PM proposals."
        )
        # Pick the most-used construction method as the recorded one,
        # purely for audit.  This doesn't change the weights we apply.
        method_counter = Counter(p.construction_method for p in proposals)
        chosen_method, _ = method_counter.most_common(1)[0]

    flagged_ids, retired_ids = _consensus_monitor_actions(proposals, config)

    # Enforce the invariant: a strategy the committee decided to flag
    # or retire must not also appear in the consensus allocations.
    # Renormalise so the remaining weights still sum to 1.
    excluded = set(flagged_ids) | set(retired_ids)
    if excluded:
        kept = {k: v for k, v in consensus_weights.items() if k not in excluded}
        total = sum(kept.values())
        if total > 0:
            consensus_weights = {k: v / total for k, v in kept.items()}
        else:
            consensus_weights = kept  # all excluded — nothing left to size

    record = _materialise(
        consensus_weights=consensus_weights,
        proposals=proposals,
        chosen_method=chosen_method,
        flagged_ids=flagged_ids,
        retired_ids=retired_ids,
        moderator_text=moderator_text,
        config=config,
        strategy_db=strategy_db,
        portfolio_db=portfolio_db,
    )
    log.info("[%s] Committee finalised %d allocations.",
             config.pm_name, len(record.allocations))
    return record


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

def _per_pm_weights(
    proposals: list[PortfolioRecord],
    config: CommitteeConfig,
) -> list[float]:
    """Normalised vote weight per PM, in proposal order."""
    if config.voting_method == VotingMethod.SIMPLE_AVERAGE:
        return [1.0 / len(proposals)] * len(proposals)

    raw = np.array(
        [config.pm_weights.get(p.pm_name, 1.0) for p in proposals],
        dtype=float,
    )
    total = raw.sum()
    if total <= 0:
        return [1.0 / len(proposals)] * len(proposals)
    return list(raw / total)


def _aggregate_weights(
    proposals: list[PortfolioRecord],
    pm_weights: list[float],
) -> dict[str, float]:
    """Take a weighted average of per-PM weight vectors.

    PMs that didn't pick a strategy contribute 0 — so a strategy
    chosen by only 1 of 3 PMs ends up with roughly 1/3 of its
    full-vote weight.  This naturally diversifies into the union of
    every PM's picks while still favouring strategies multiple PMs
    agree on.
    """
    acc: dict[str, float] = {}
    for proposal, w_pm in zip(proposals, pm_weights):
        for a in proposal.allocations:
            if not a.enabled:
                continue
            acc[a.strategy_id] = acc.get(a.strategy_id, 0.0) + w_pm * a.weight

    # Strip near-zero entries that survived rounding, then renormalise
    # so weights sum to 1 (committee output is fully invested).
    cleaned = {k: v for k, v in acc.items() if abs(v) > 1e-6}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def _consensus_monitor_actions(
    proposals: list[PortfolioRecord],
    config: CommitteeConfig,
) -> tuple[list[str], list[str]]:
    """Aggregate flag / retire votes across PMs."""
    n = len(proposals)
    flag_counts: Counter[str] = Counter()
    retire_counts: Counter[str] = Counter()
    for p in proposals:
        flag_counts.update(p.flagged_strategy_ids)
        retire_counts.update(p.retired_strategy_ids)

    flagged = [sid for sid, c in flag_counts.items()
               if c / n >= config.flag_threshold]
    retired = [sid for sid, c in retire_counts.items()
               if c / n >= config.retire_threshold]
    # If a strategy is both flagged and retired by majorities,
    # retire wins (it's the stronger action).
    flagged = [sid for sid in flagged if sid not in retired]
    return flagged, retired


# ---------------------------------------------------------------------------
# LLM moderator
# ---------------------------------------------------------------------------

MODERATOR_PROMPT = """\
You are the chair of a quantitative-fund portfolio committee.  Several
portfolio managers (PMs) — each with their own risk personality — have
independently proposed allocations.  Your job is to read every
proposal and produce ONE final allocation that synthesises their
views.

YOU SHOULD
----------
- Prefer strategies that multiple PMs converged on.
- Respect each PM's reasoning where it's well argued; don't blindly
  average if one PM has flagged a strategy as fragile.
- Avoid concentrating into a small handful of strategies unless every
  PM explicitly wants that.
- Output weights that sum to 1.0 and are non-negative.

PM PROPOSALS
============
{proposals_text}

UNIVERSE-WIDE CORRELATION HINTS
-------------------------------
{correlation_hints}

Respond in JSON with three keys:
  "chosen_method": one of {{"equal_weight", "inverse_volatility",
    "min_variance", "mean_variance", "max_sharpe", "risk_parity",
    "hierarchical_risk_parity", "custom_llm"}} — what aggregation
    label best describes your choice (for audit).  Use "custom_llm"
    when you've materially diverged from any single PM's proposal.
  "weights": object mapping strategy_id → weight.  Weights must be
    non-negative and sum to (approximately) 1.0.
  "rationale": short paragraph explaining your synthesis.
"""


def _llm_moderate(
    proposals: list[PortfolioRecord],
    strategy_db: StrategyDatabase,
) -> tuple[dict[str, float], str, ConstructionMethod]:
    """One LLM call to synthesise a consensus from N proposals."""
    proposals_text = []
    for p in proposals:
        lines = [
            f"### PM '{p.pm_name}' ({p.personality.value}) — "
            f"method={p.construction_method.value}",
        ]
        for a in p.allocations:
            if a.enabled:
                lines.append(f"  - {a.strategy_id}: w = {a.weight:+.4f}")
        if p.flagged_strategy_ids:
            lines.append(f"  flagged: {p.flagged_strategy_ids}")
        if p.retired_strategy_ids:
            lines.append(f"  retired: {p.retired_strategy_ids}")
        if p.pm_rationale:
            lines.append(f"  rationale: {p.pm_rationale[:400]}")
        proposals_text.append("\n".join(lines))

    # Compact correlation summary: any pair with |ρ| > 0.7 mentioned by
    # at least one proposal.  Helps the moderator avoid double-counting.
    corr = strategy_db.correlation_matrix
    mentioned = {
        a.strategy_id for p in proposals for a in p.allocations if a.enabled
    }
    pairs = []
    if not corr.empty:
        keep = [s for s in mentioned if s in corr.index]
        sub = corr.loc[keep, keep]
        for i, si in enumerate(keep):
            for sj in keep[i + 1:]:
                rho = float(sub.loc[si, sj])
                if abs(rho) > 0.7:
                    pairs.append(f"  ρ({si}, {sj}) = {rho:+.2f}")
    corr_hints = "\n".join(pairs) if pairs else "  (no |ρ| > 0.7 pairs)"

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        resp = llm.invoke(MODERATOR_PROMPT.format(
            proposals_text="\n\n".join(proposals_text),
            correlation_hints=corr_hints,
        ))
        parsed = _parse_json(resp.content)
    except Exception as e:
        log.warning("Committee LLM moderator failed (%s); "
                    "falling back to simple average.", e)
        weights = _aggregate_weights(proposals, [1.0 / len(proposals)] * len(proposals))
        return weights, f"LLM moderator failed: {e!s}; used simple_average.", \
               ConstructionMethod.CUSTOM_LLM

    raw_weights = parsed.get("weights") or {}
    valid = {a.strategy_id for p in proposals for a in p.allocations}
    weights = {
        k: float(v) for k, v in raw_weights.items()
        if k in valid and isinstance(v, (int, float)) and v >= 0
    }
    total = sum(weights.values())
    if total <= 0:
        log.warning("LLM moderator returned no valid weights; "
                    "falling back to simple average.")
        weights = _aggregate_weights(proposals, [1.0 / len(proposals)] * len(proposals))
    else:
        weights = {k: v / total for k, v in weights.items()}

    try:
        method = ConstructionMethod(
            parsed.get("chosen_method", "custom_llm"),
        )
    except ValueError:
        method = ConstructionMethod.CUSTOM_LLM

    return weights, parsed.get("rationale", ""), method


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def _materialise(
    consensus_weights: dict[str, float],
    proposals: list[PortfolioRecord],
    chosen_method: ConstructionMethod,
    flagged_ids: Iterable[str],
    retired_ids: Iterable[str],
    moderator_text: str,
    config: CommitteeConfig,
    strategy_db: StrategyDatabase,
    portfolio_db: PortfolioDatabase,
) -> PortfolioRecord:
    """Apply the consensus to the live DBs and write a PortfolioRecord."""
    # Build the allocation list.
    allocations = [
        PortfolioAllocation(
            strategy_id=sid,
            weight=float(w),
            enabled=abs(float(w)) > 1e-6,
            rationale=f"committee {config.voting_method.value}",
        )
        for sid, w in consensus_weights.items()
    ]
    allocations.sort(key=lambda a: -abs(a.weight))

    # Apply flag / retire consensus.
    flagged_ids = list(flagged_ids)
    retired_ids = list(retired_ids)
    for sid in retired_ids:
        strategy_db.retire_strategy(
            sid, reason=f"Retired by committee ({config.pm_name}).",
        )
    for sid in flagged_ids:
        strategy_db.flag_strategy(
            sid, reason=f"Flagged by committee ({config.pm_name}).",
        )

    # Apply LIVE / PAUSED updates.
    new_live_ids = {a.strategy_id for a in allocations if a.enabled}
    for rec in strategy_db.list_strategies():
        if rec.pm_status in (PMStatus.RETIRED, PMStatus.FLAGGED_FOR_REVIEW):
            continue
        if rec.id in new_live_ids:
            strategy_db.set_pm_status(rec.id, PMStatus.LIVE)
        elif rec.pm_status == PMStatus.LIVE:
            strategy_db.set_pm_status(
                rec.id, PMStatus.PAUSED,
                notes=f"De-selected by committee ({config.pm_name}).",
            )

    # Compute ex-ante expected metrics off the consensus weights.
    expected_returns = {}
    for sid in new_live_ids:
        rec = strategy_db.get_strategy(sid)
        if rec and rec.backtest_metrics:
            expected_returns[sid] = float(rec.backtest_metrics.annualised_return or 0.0)
    cov = strategy_db.covariance_matrix()
    metrics = (
        expected_portfolio_metrics(
            {a.strategy_id: a.weight for a in allocations},
            expected_returns=expected_returns,
            cov=cov,
        )
        if not cov.empty
        else {}
    )

    # Compose the committee's rationale.
    pm_lines = [
        f"  • {p.pm_name} ({p.personality.value}, method={p.construction_method.value})"
        for p in proposals
    ]
    rationale = (
        f"Committee aggregation ({config.voting_method.value}) of "
        f"{len(proposals)} PMs:\n"
        + "\n".join(pm_lines)
        + ("\n\n" + moderator_text if moderator_text else "")
    )

    # Use the committee's name as the audit identity.  Personality is
    # the majority personality across PMs.
    pers_counter = Counter(p.personality for p in proposals)
    majority_personality, _ = pers_counter.most_common(1)[0]
    mode = PMMode.ACTIVE if config.voting_method == VotingMethod.LLM_MODERATOR else PMMode.SELECTOR

    record = PortfolioRecord(
        id=f"portfolio_{uuid.uuid4().hex[:10]}",
        pm_name=config.pm_name,
        mode=mode,
        personality=majority_personality,
        construction_method=chosen_method,
        target_n_strategies=max(
            (p.target_n_strategies for p in proposals), default=len(allocations),
        ),
        allocations=allocations,
        flagged_strategy_ids=flagged_ids,
        retired_strategy_ids=retired_ids,
        expected_metrics=metrics,
        pm_rationale=rationale,
        metadata={
            "voting_method": config.voting_method.value,
            "flag_threshold": config.flag_threshold,
            "retire_threshold": config.retire_threshold,
            "n_pms": len(proposals),
            "pm_names": [p.pm_name for p in proposals],
            "pm_proposals": [
                {
                    "pm_name": p.pm_name,
                    "personality": p.personality.value,
                    "construction_method": p.construction_method.value,
                    "weights": {a.strategy_id: a.weight
                                for a in p.allocations if a.enabled},
                    "flagged": p.flagged_strategy_ids,
                    "retired": p.retired_strategy_ids,
                    "rationale": p.pm_rationale,
                }
                for p in proposals
            ],
        },
    )
    portfolio_db.add_portfolio(record)
    return record
