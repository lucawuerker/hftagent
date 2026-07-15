"""Deterministic badge + plain-English verdict for a card.

The badge is derived ONLY from the harness's numbers (DSR probability, CSCV
PBO, OOS test verdict) — never from the Statistician LLM's accept/reject prose
— so the "the AI proposes, a harness it can't game disposes" story holds for
the marketing artifacts themselves.  The thresholds align with the harness's
own bars (deflated_sharpe.py: PASS ≥ 0.75, WARN floor 0.60).

Every emitted string passes ``check_compliance`` (company-brain/CLAUDE.md
banned words), and every card carries a fixed backtest caveat.
"""

from __future__ import annotations

import re

from .metrics import CardMetrics

BADGE_ROBUST = "Robust"
BADGE_OVERFIT = "Likely overfit"
BADGE_WORTH_TESTING = "Worth testing"

# From company-brain/CLAUDE.md + brand/BRAND.md — never in any published copy.
BANNED_WORDS = (
    "guaranteed", "risk-free", "sure thing", "beat the market",
    "get rich", "can't lose", "cant lose", "proven returns",
)

CAVEAT = ("Backtested on historical data. Past performance does not indicate "
          "future results; this is research, not investment advice.")


class ComplianceError(ValueError):
    """A published string violated the brand's compliance guardrails."""


def check_compliance(text: str) -> str:
    """Raise if ``text`` contains a banned claim; return it unchanged otherwise."""
    lowered = re.sub(r"\s+", " ", text.lower())
    for word in BANNED_WORDS:
        if word in lowered:
            raise ComplianceError(
                f"banned phrase {word!r} in published copy: {text[:120]!r}")
    return text


def assign_badge(m: CardMetrics) -> str:
    """Deterministic badge from the harness numbers only."""
    dsr = m.dsr_prob if m.dsr_prob is not None else 0.0
    oos = (m.oos_verdict or "").lower()
    if (m.pbo is not None and m.pbo >= 0.50) or dsr < 0.60 or oos == "fail":
        return BADGE_OVERFIT
    if m.pbo is not None and m.pbo <= 0.25 and dsr >= 0.75 and oos == "pass":
        return BADGE_ROBUST
    return BADGE_WORTH_TESTING


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _num(x: float | None, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"


def verdict_note(m: CardMetrics, badge: str) -> str:
    """1–2 sentence plain-English verdict, template-filled with the numbers."""
    n = m.n_trials or 1
    if badge == BADGE_OVERFIT:
        # Cite only the harness signals that actually triggered the badge.
        reasons: list[str] = []
        if m.pbo is not None and m.pbo >= 0.50:
            reasons.append(f"the probability of backtest overfitting across the "
                           f"{n} variants tried is {_pct(m.pbo)}")
        if (m.dsr_prob or 0.0) < 0.60:
            reasons.append(f"after correcting for the {n} variants tried, the "
                           f"confidence the true edge is real drops to "
                           f"{_pct(m.dsr_prob)}")
        if (m.oos_verdict or "").lower() == "fail":
            reasons.append("the edge fades on data the model never saw")
        joined = "; ".join(reasons) or "the harness checks flag it as fragile"
        note = (
            f"In-sample Sharpe {_num(m.is_sharpe, 1)} looks strong, but {joined}. "
            f"We'd tell you not to risk real money on it."
        )
    elif badge == BADGE_ROBUST:
        note = (
            f"The edge held on data the model never saw — out-of-sample Sharpe "
            f"{_num(m.oos_sharpe, 1)} vs {_num(m.oos_is_sharpe, 1)} in-sample — with a "
            f"{_pct(m.pbo)} probability of backtest overfitting across the {n} "
            f"variants tried. Still a backtest, not a promise of future results."
        )
    else:
        note = (
            f"Some signal survives out-of-sample (Sharpe {_num(m.oos_sharpe, 1)} vs "
            f"{_num(m.oos_is_sharpe, 1)} in-sample), but with a {_pct(m.pbo)} "
            f"probability of backtest overfitting over {n} variants the evidence "
            f"isn't strong enough yet. Worth watching, not worth conviction."
        )
    return check_compliance(note)
