"""E3: skeptic debate + RAG grounding for execution programs.

**Debate** — before an LLM-proposed executor costs an evaluation, a skeptic
attacks it along the execution-specific lines (DESIGN §Agent pipeline): cost
realism, capacity/liquidity, leverage disguises, redundancy with the archived
executors, and backtest-artifact exploitation (the causality probe's manual
twin).  ≤1 revision round; any LLM/parsing failure fails **open** (the debate
is a quality filter, never a single point of failure).

**RAG** — `execution_literature_snippets` retrieves execution/microstructure
passages (transaction costs, optimal execution, volatility targeting, turnover)
from the existing `knowledge.embed_store` corpus and splices them into the
mutation prompt; empty corpus → empty string (fails open).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("execution_research.debate")

_ATTACK_LINES = """\
Attack the proposal along EXACTLY these lines (be concrete, cite its code):
1. COST REALISM — does the edge survive realistic transaction costs, or does
   it churn the book faster than the spread can be paid?
2. CAPACITY / LIQUIDITY — does it concentrate into names or moments where the
   assumed fills are fantasy?
3. LEVERAGE DISGUISE — does apparent skill come from silently scaling gross
   exposure rather than better timing/selection?
4. REDUNDANCY — is it a re-skin of an archived executor's mechanism?
5. BACKTEST ARTIFACTS — full-sample statistics, centred windows, bar-boundary
   tricks, anything the truncation-replay causality probe would catch."""


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in skeptic response")
    return json.loads(raw[start:end + 1])


def build_skeptic_prompt(payload: dict[str, Any],
                         archive_mechanisms: list[str] | None = None) -> str:
    mech_txt = ("\n".join(f"- {m}" for m in (archive_mechanisms or [])[:10])
                or "(archive empty)")
    return f"""You are the execution-desk SKEPTIC of a quant fund. A researcher
proposes a new execution program (signal → target book).

PROPOSAL (mechanism: {payload.get('mechanism') or '?'};
expected effect: {payload.get('expected_effect') or '?'};
regime: {payload.get('regime')}):
```python
{payload.get('code', '')}
```

Mechanisms already in the executor archive:
{mech_txt}

{_ATTACK_LINES}

Respond with ONLY a JSON object:
{{"verdict": "accept" | "revise" | "reject",
 "critique": "your strongest concrete objections",
 "required_change": "what a revision MUST fix (empty when accept/reject)"}}"""


def run_exec_debate(
    llm: Any,
    payload: dict[str, Any],
    *,
    archive_mechanisms: list[str] | None = None,
    max_revisions: int = 1,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Debate one executor proposal → ``(verdict, final_payload, transcript)``.

    ``verdict`` ∈ {"accept", "reject"}; after ``max_revisions`` an unresolved
    ``revise`` becomes ``reject``.  Fails open on any LLM/parse error.
    """
    transcript: list[dict[str, Any]] = []
    current = dict(payload)
    for round_no in range(max_revisions + 1):
        try:
            resp = llm.invoke(build_skeptic_prompt(current, archive_mechanisms))
            verdict_obj = _extract_json(getattr(resp, "content", str(resp)))
        except Exception as e:  # noqa: BLE001 — fail OPEN
            log.info("exec debate failed open (%s)", e)
            transcript.append({"round": round_no, "error": str(e)})
            return "accept", current, transcript
        transcript.append({"round": round_no, **verdict_obj})
        verdict = str(verdict_obj.get("verdict", "accept")).lower()
        if verdict == "accept":
            return "accept", current, transcript
        if verdict == "reject":
            return "reject", current, transcript
        if round_no >= max_revisions:
            return "reject", current, transcript
        # one revision round: hand the critique back to the proposer LLM
        try:
            fix_prompt = (
                "Revise your execution program to address this critique — keep "
                "the SAME executor_id and the JSON response format "
                '{"executor_id", "name", "regime", "mechanism", '
                '"expected_effect", "code"}:\n'
                f"CRITIQUE: {verdict_obj.get('critique', '')}\n"
                f"REQUIRED CHANGE: {verdict_obj.get('required_change', '')}\n\n"
                "Your previous proposal:\n```python\n"
                f"{current.get('code', '')}\n```")
            resp = llm.invoke(fix_prompt)
            revised = _extract_json(getattr(resp, "content", str(resp)))
            if revised.get("code"):
                revised["executor_id"] = current["executor_id"]
                current = {**current, **revised}
        except Exception as e:  # noqa: BLE001 — a failed revision keeps the original
            log.info("exec debate revision failed (%s) — keeping original", e)
    return "accept", current, transcript


# ── RAG grounding (execution / microstructure literature) ─────────────────────

EXEC_QUERY = ("transaction costs optimal execution turnover volatility "
              "targeting position sizing market impact drawdown control "
              "portfolio rebalancing")


def execution_literature_snippets(k: int = 3, max_chars: int = 2400) -> str:
    """Top-k execution-literature chunks from the knowledge corpus (or "")."""
    try:
        from quant_fund_agent.knowledge.embed_store import EmbedStore, load_corpus

        docs = load_corpus()
        if not docs:
            return ""
        store = EmbedStore(docs)
        chunks = store.retrieve_chunks(EXEC_QUERY, k=k)
        parts = []
        for c in chunks:
            src = getattr(c, "paper_id", None) or getattr(c, "doc_id", "?")
            text = (getattr(c, "text", "") or "")[: max_chars // max(1, k)]
            parts.append(f"[{src}] {text}")
        if not parts:
            return ""
        return ("RELEVANT EXECUTION LITERATURE (retrieved; ground your "
                "mechanism in it where it helps):\n" + "\n".join(parts))
    except Exception as e:  # noqa: BLE001 — RAG is a bonus, never a blocker
        log.info("execution RAG unavailable (%s)", e)
        return ""
