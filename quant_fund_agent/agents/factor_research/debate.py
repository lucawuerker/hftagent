"""Adversarial debate over a factor hypothesis: proposer → skeptic → moderator.

Switchable (``--debate {on,off}`` — the on/off switch is itself a thesis
ablation: *does adversarial critique raise OOS quality?*).  Placed **before
codegen** so tokens are not spent implementing weak ideas, per the design.

Flow (:func:`run_debate`): the *proposer's* hypothesis (a structured idea dict)
is attacked by the **skeptic** — economic soundness, look-ahead risk,
likely-already-arbitraged, redundancy with the current book (it is shown the
book summary and the data scope) — then the **moderator** rules
``accept | revise | reject``.  On ``revise`` the proposer gets ONE revision
round (the critique + guidance are fed back), after which the skeptic/moderator
rule again and ``revise`` degrades to ``reject`` (at most one loop, per the
design).  The full transcript is returned for the audit trail.

The debate only ever shapes *which ideas get implemented* — its words never
touch the deterministic fitness, so the reward channel stays un-gameable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("factor_research.debate")

SKEPTIC_PROMPT = """\
You are the in-house skeptic at a quantitative trading firm — the person who
kills weak research before it burns compute.  A researcher proposes the factor
hypothesis below.  Attack it on exactly these dimensions:

1. ECONOMIC SOUNDNESS — is the stated mechanism a real market phenomenon with a
   plausible causal story, or a just-so narrative over a formula?
2. LOOK-AHEAD / IMPLEMENTABILITY — does computing it at time t require anything
   not knowable at t?  Does it depend on fields outside the data scope below?
3. CROWDING — is this a well-known, likely-arbitraged effect presented as new?
4. REDUNDANCY — is it materially the same signal as something already in the
   book (listed below)?

Do not penalize an idea merely because it is unfamiliar, nonlinear, or requires
a non-standard deterministic transform.  Attack novelty only when the mechanism
is incoherent, uncomputable from the data scope, leaky, or just a renamed common
factor.

DATA SCOPE (fields available this run)
--------------------------------------
{data_context}

CURRENT BOOK (accepted factors)
-------------------------------
{book_summary}

PROPOSED HYPOTHESIS
-------------------
{idea_json}

Respond with strict JSON:
{{
  "issues": [
    {{"dimension": "economic|lookahead|crowding|redundancy",
      "severity": "fatal|major|minor",
      "critique": "<1-3 specific sentences>"}}
  ],
  "overall": "<2-3 sentence summary of your judgement>"
}}
An empty "issues" list means you found nothing substantive to attack.
"""

MODERATOR_PROMPT = """\
You moderate a research debate at a quantitative trading firm.  A proposed
factor hypothesis and the skeptic's critique are below.  Decide the outcome:

- "accept": the idea survives; minor issues at most, or the critique is generic.
- "revise": specific, fixable flaws — say precisely what must change.
- "reject": a fatal flaw (unimplementable, blatant look-ahead, pure redundancy,
  or no coherent mechanism).

Be decisive.  Fatal look-ahead or redundancy issues are NOT fixable by wording.

PROPOSED HYPOTHESIS
-------------------
{idea_json}

SKEPTIC'S CRITIQUE
------------------
{critique_json}

Respond with strict JSON:
{{"verdict": "accept" | "revise" | "reject",
  "guidance": "<what to change if revise; why if reject; empty if accept>"}}
"""

REVISION_PROMPT = """\
You proposed the factor hypothesis below.  The firm's skeptic attacked it and
the moderator requires a revision.  Produce a REVISED hypothesis that fixes the
named issues while keeping the core insight (or replacing it with a sounder
neighbour if the mechanism itself was the problem).  Stay within the data scope.

DATA SCOPE (fields available this run)
--------------------------------------
{data_context}

YOUR ORIGINAL HYPOTHESIS
------------------------
{idea_json}

SKEPTIC'S CRITIQUE
------------------
{critique_json}

MODERATOR'S GUIDANCE
--------------------
{guidance}

Respond with strict JSON — the SAME schema as the original hypothesis (keep the
same "factor_id" unless the mechanism changed entirely):
{idea_schema}
"""

# The hypothesis schema the revision must echo (matches the structured idea the
# Hypothesis step emits — a superset of the brainstorm idea schema).
IDEA_SCHEMA = """\
{{
  "factor_id": "<snake_case id>",
  "name": "<short name>",
  "category": "<momentum|mean_reversion|volatility|microstructure|statistical_arbitrage|carry|sentiment|other>",
  "trading_idea": "<the mechanism: 2-4 sentences>",
  "description": "<what the signal computes>",
  "prediction_horizon": <positive int bars>,
  "suggested_horizons": [<int>, ...],
  "expected_sign": <1 or -1>,
  "regime_dependence": "<when it should work / fail, 1-2 sentences>",
  "source_paper_ids": ["<paper_id>", ...]
}}"""


def _invoke_json(llm: Any, prompt: str) -> dict:
    from quant_fund_agent.agents.factor_research.graph import _parse_json

    resp = llm.invoke(prompt)
    return _parse_json(getattr(resp, "content", str(resp)))


def book_summary(entries: list[dict[str, Any]]) -> str:
    """Render the accepted book for the skeptic: id, category, one-line thesis."""
    if not entries:
        return "(the book is currently empty)"
    lines = []
    for e in entries:
        idea = (e.get("trading_idea") or e.get("description") or "").strip()
        idea = " ".join(idea.split())
        if len(idea) > 160:
            idea = idea[:157] + "..."
        lines.append(f"- {e.get('factor_id')} [{e.get('category', '?')}]: {idea}")
    return "\n".join(lines)


def run_debate(
    llm: Any,
    idea: dict[str, Any],
    *,
    data_context: str,
    book: list[dict[str, Any]] | None = None,
    max_revisions: int = 1,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Debate one hypothesis → ``(verdict, final_idea, transcript)``.

    ``verdict`` is ``"accept"`` or ``"reject"`` (``revise`` is internal: after
    ``max_revisions`` revision rounds an unresolved ``revise`` becomes
    ``reject``).  ``final_idea`` is the (possibly revised) idea dict.  The
    transcript records every skeptic/moderator/revision payload for the audit
    trail.  Any LLM/parsing failure fails **open** (accept, logged): the debate
    is a quality filter, not a single point of failure for the whole loop.
    """
    book_txt = book_summary(book or [])
    transcript: list[dict[str, Any]] = []
    current = dict(idea)

    for round_no in range(max_revisions + 1):
        idea_json = json.dumps(current, indent=2, default=str)
        try:
            critique = _invoke_json(llm, SKEPTIC_PROMPT.format(
                data_context=data_context, book_summary=book_txt,
                idea_json=idea_json))
        except Exception as e:  # noqa: BLE001 — fail open
            log.warning("skeptic call failed (%s) — debate skipped, idea accepted", e)
            transcript.append({"round": round_no, "error": f"skeptic: {e}"})
            return "accept", current, transcript

        try:
            ruling = _invoke_json(llm, MODERATOR_PROMPT.format(
                idea_json=idea_json, critique_json=json.dumps(critique, default=str)))
        except Exception as e:  # noqa: BLE001 — fail open
            log.warning("moderator call failed (%s) — debate skipped, idea accepted", e)
            transcript.append({"round": round_no, "critique": critique,
                               "error": f"moderator: {e}"})
            return "accept", current, transcript

        verdict = str(ruling.get("verdict", "accept")).strip().lower()
        guidance = str(ruling.get("guidance", ""))
        transcript.append({"round": round_no, "critique": critique,
                           "verdict": verdict, "guidance": guidance})

        if verdict == "accept":
            return "accept", current, transcript
        if verdict == "reject":
            return "reject", current, transcript
        if round_no >= max_revisions:  # unresolved revise → reject (≤1 loop)
            transcript.append({"round": round_no, "note":
                               "revision budget exhausted — treated as reject"})
            return "reject", current, transcript

        try:
            current = _invoke_json(llm, REVISION_PROMPT.format(
                data_context=data_context, idea_json=idea_json,
                critique_json=json.dumps(critique, default=str),
                guidance=guidance, idea_schema=IDEA_SCHEMA))
            transcript.append({"round": round_no, "revised_idea": current})
        except Exception as e:  # noqa: BLE001 — a failed revision keeps the original
            log.warning("revision call failed (%s) — original idea kept", e)
            transcript.append({"round": round_no, "error": f"revision: {e}"})

    return "accept", current, transcript  # pragma: no cover — loop always returns
