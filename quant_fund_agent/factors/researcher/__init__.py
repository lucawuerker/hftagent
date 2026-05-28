"""Auto-generated factors produced by the Factor Researcher Agent.

This package is intentionally empty in source control.  Every ``*.py``
file inside it is written at runtime by the researcher agent and
removed again by ``purge_researcher_factors`` between simulation runs.

Design rationale
----------------
Putting generated factors inside the existing ``quant_fund_agent.factors``
package means they participate in the *same* discovery + registration
flow as seed factors (``discover_factors()`` → ``@register_factor``).
There is only one runtime path for executing factors, regardless of
who wrote them.

The seed factors (the WorldQuant-style alphas, candlestick patterns,
etc.) live in sibling packages (``factors/momentum/``, ``factors/mean_reversion/``,
…) and are version-controlled.  Researcher output lives only here and
is gitignored so:

1. It cannot pollute the deterministic baseline ensemble.
2. It can be wiped cleanly between 1-year backtest simulations.
3. Reviewers reading the repo aren't confused by ephemeral LLM output.

Do not place hand-written factors in this directory.
"""
