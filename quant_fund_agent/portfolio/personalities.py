"""Pre-baked PM personality profiles.

A "personality" bundles:
  * **screening filters** — which strategies the PM is willing to deploy
    (min Sharpe, max drawdown, max correlation to the existing book).
  * **portfolio shape** — how many strategies the PM wants to run, the
    max weight per strategy, whether shorting is allowed.
  * **construction preference** — which method the PM defaults to in
    SELECTOR mode.

Treating personalities as plain dataclasses (not hard-coded into the
agent) lets us run multiple PMs side-by-side with different sleeves of
capital — for example one defensive PM and one aggressive PM consuming
the same strategy DB.
"""

from __future__ import annotations

from dataclasses import field

# Pydantic dataclass (not plain @dataclass) so the LangGraph state can
# carry a PersonalityProfile between nodes — Pydantic 2.x re-validates
# state on every node transition and won't coerce a plain @dataclass.
from pydantic.dataclasses import dataclass

from quant_fund_agent.schemas import ConstructionMethod, PMPersonality


@dataclass
class PersonalityProfile:
    """Configuration for one PM "personality"."""

    name: str
    personality: PMPersonality

    # ── screening filters ──
    # Strategies failing any of these are *not* deployed.
    min_sharpe: float = 0.5
    max_drawdown: float = -0.10            # e.g. -10 %, more negative is worse
    max_pairwise_correlation: float = 0.85  # vs other LIVE strategies
    min_hit_rate: float | None = None

    # ── monitoring / flagging filters ──
    # Applied to live (out-of-sample / in-prod) metrics.  A live strategy
    # whose performance drifts beyond these gets flagged for the
    # research team to revisit, and retired if the drift is severe.
    live_sharpe_floor: float = 0.0
    live_sharpe_kill_floor: float = -0.5
    live_dd_floor: float = -0.15

    # ── portfolio shape ──
    target_n_strategies: int = 10
    max_weight_per_strategy: float = 0.25
    allow_short: bool = False

    # ── construction ──
    default_construction_method: ConstructionMethod = ConstructionMethod.EQUAL_WEIGHT
    risk_aversion: float = 1.0
    risk_free_rate: float = 0.0

    # ── audit ──
    description: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-defined personalities
# ---------------------------------------------------------------------------

DEFENSIVE = PersonalityProfile(
    name="defensive_pm",
    personality=PMPersonality.DEFENSIVE,
    min_sharpe=0.8,
    max_drawdown=-0.05,
    max_pairwise_correlation=0.70,
    min_hit_rate=0.50,
    live_sharpe_floor=0.3,
    live_sharpe_kill_floor=-0.2,
    live_dd_floor=-0.08,
    target_n_strategies=12,
    max_weight_per_strategy=0.15,
    allow_short=False,
    default_construction_method=ConstructionMethod.MIN_VARIANCE,
    risk_aversion=4.0,
    description=(
        "Capital-preservation PM.  Favours low-vol strategies, tight DD, "
        "diversifies broadly across many uncorrelated strategies."
    ),
)


BALANCED = PersonalityProfile(
    name="balanced_pm",
    personality=PMPersonality.BALANCED,
    min_sharpe=0.6,
    max_drawdown=-0.10,
    max_pairwise_correlation=0.80,
    min_hit_rate=None,
    live_sharpe_floor=0.1,
    live_sharpe_kill_floor=-0.4,
    live_dd_floor=-0.12,
    target_n_strategies=10,
    max_weight_per_strategy=0.25,
    allow_short=False,
    default_construction_method=ConstructionMethod.RISK_PARITY,
    risk_aversion=1.0,
    description=(
        "Default PM.  Equal-risk-contribution by default, moderate "
        "screening, modest concentration caps."
    ),
)


AGGRESSIVE = PersonalityProfile(
    name="aggressive_pm",
    personality=PMPersonality.AGGRESSIVE,
    min_sharpe=0.4,
    max_drawdown=-0.20,
    max_pairwise_correlation=0.90,
    min_hit_rate=None,
    live_sharpe_floor=-0.2,
    live_sharpe_kill_floor=-0.8,
    live_dd_floor=-0.25,
    target_n_strategies=6,
    max_weight_per_strategy=0.40,
    allow_short=True,
    default_construction_method=ConstructionMethod.MAX_SHARPE,
    risk_aversion=0.25,
    description=(
        "Return-seeking PM.  Tolerates larger drawdowns, concentrates "
        "into a handful of high-Sharpe strategies, willing to short."
    ),
)


_REGISTRY: dict[PMPersonality, PersonalityProfile] = {
    PMPersonality.DEFENSIVE: DEFENSIVE,
    PMPersonality.BALANCED: BALANCED,
    PMPersonality.AGGRESSIVE: AGGRESSIVE,
}


def get_personality_profile(personality: PMPersonality) -> PersonalityProfile:
    """Return the pre-baked profile for ``personality``.

    For ``PMPersonality.CUSTOM`` the caller is expected to supply its
    own ``PersonalityProfile`` directly — we still return BALANCED as a
    sensible default so naive callers don't crash.
    """
    return _REGISTRY.get(personality, BALANCED)
