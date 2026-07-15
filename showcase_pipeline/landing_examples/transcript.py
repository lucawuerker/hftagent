"""Grounded chat-demo transcript for one exported example.

Deterministic template filled from the real run: the user idea is the run's
actual hypothesis, the clarification names the real universe, the research
progress steps carry real counts, and the final verdict message uses the exact
card numbers and wording.  ``polish_with_llm`` optionally smooths the *user*
turn only (one call, flagged in provenance) — the verdict message is never
touched by an LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from .metrics import CardMetrics
from .verdict import check_compliance

log = logging.getLogger("landing_examples.transcript")


def _shorten(text: str, max_len: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut.rstrip(".,;: ") + "…"


def build_transcript(
    candidate: dict[str, Any],
    card: CardMetrics,
    badge: str,
    note: str,
) -> dict[str, Any]:
    """The chat-demo message sequence, every number from the real run."""
    universe = candidate.get("universe") or {}
    uni_label = universe.get("universe") or "the configured universe"
    n_factors = len(card.factor_ids)
    n_trials = card.n_trials or len(candidate.get("trial_history") or [])

    user_idea = _shorten(candidate.get("hypothesis")
                         or "Build me a strategy from the researched factors.")

    pbo_str = f"{card.pbo:.0%}" if card.pbo is not None else "n/a"
    dsr_str = f"{card.dsr_prob:.0%}" if card.dsr_prob is not None else "n/a"

    messages = [
        {"role": "user", "text": user_idea},
        {"role": "lodestar",
         "text": (f"On it — researching factors and backtesting on {uni_label} "
                  f"({universe.get('frequency', 'daily')} data), with the "
                  "overfitting checks on.")},
        {"role": "progress", "steps": [
            f"Selected {n_factors} candidate factor(s) from the researched library",
            f"Fitted and backtested {n_trials} strategy variant(s) in-sample",
            "Scored the winner on held-out data it never saw",
            f"Deflated the results for the {n_trials} variant(s) tried "
            f"(probability of overfitting: {pbo_str})",
        ]},
        {"role": "lodestar", "verdict": badge,
         "text": (f"{card.strategy_name} — {badge}. In-sample Sharpe "
                  f"{card.is_sharpe:.1f}" if card.is_sharpe is not None else
                  f"{card.strategy_name} — {badge}.")},
        {"role": "lodestar", "text": note},
    ]

    # Compliance-check every string we publish.
    for msg in messages:
        if "text" in msg:
            check_compliance(msg["text"])
        for step in msg.get("steps", []):
            check_compliance(step)

    return {
        "strategy": card.strategy_name,
        "badge": badge,
        "stats": {"pbo": pbo_str, "deflated_sharpe_prob": dsr_str,
                  "is_sharpe": card.is_sharpe, "oos_sharpe": card.oos_sharpe},
        "messages": messages,
        "llm_polished": False,
    }


def polish_with_llm(transcript: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Optionally smooth the USER turn's wording with one LLM call.

    Only the user's idea message is rewritten (to sound like a person typing,
    not a selector hypothesis); every Lodestar/verdict message stays verbatim
    deterministic.  The result is flagged ``llm_polished: true``.
    """
    import os

    from langchain_openai import ChatOpenAI

    user_msg = next(m for m in transcript["messages"] if m["role"] == "user")
    llm = ChatOpenAI(model=model or os.getenv("SELECTOR_LLM_MODEL", "gpt-4o-mini"),
                     temperature=0.3)
    prompt = (
        "Rewrite this quant trading hypothesis as a short, casual one-or-two "
        "sentence message a retail investor would type into a chat box, keeping "
        "the substance identical. No promises of returns, no hype words. "
        f"Hypothesis: {user_msg['text']}"
    )
    try:
        polished = llm.invoke(prompt).content.strip().strip('"')
        user_msg["text"] = check_compliance(_shorten(polished))
        transcript["llm_polished"] = True
    except Exception as e:  # noqa: BLE001 — polish is cosmetic, never fatal
        log.warning("LLM polish failed (%s) — keeping deterministic wording", e)
    return transcript
