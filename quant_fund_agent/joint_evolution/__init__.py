"""Joint factor×execution evolution — the block-coordinate outer layer (J0–J4).

RD-Agent(Q)-style alternation with the statistics RD-Agent lacks: shared SOTA
state (curated factor book ↔ best executor), a deterministic re-freeze protocol
at block boundaries, a {sequential, round_robin, random, bandit} scheduler, a
shared cross-arm N_trials ledger (per-family gate counts + a joint look count),
and — the headline protocol — a joint walk-forward where the WHOLE outer loop
re-runs per fold.  Design anchor: ``docs/joint-evolution/DESIGN.md``.

No LLM lives in this package: the scheduler, re-freeze and joint objective are
all deterministic; LLM agency stays inside the arms' mutation operators.
"""

from quant_fund_agent.joint_evolution.ledger import TrialsLedger  # noqa: F401
from quant_fund_agent.joint_evolution.state import SOTAState  # noqa: F401
