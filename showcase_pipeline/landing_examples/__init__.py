"""Landing-page example generator for the Lodestar spinoff.

Runs the real strategy pipeline (Selector → Architect → Statistician) over a
researched prerun, dumps EVERY attempt (approved and rejected) with full trial
history and stat-test details, and exports chosen examples — verdict-card data
pack, "Behind the verdict" story, grounded chat transcript, provenance — into
the company-brain marketing repo.

The badge and every published number come from the deterministic harness
(deflated Sharpe, CSCV probability of backtest overfitting, held-out OOS
backtest) — never from an LLM's judgement — so the marketing artifacts obey the
same "the AI never grades itself" rule the product sells.

CLI::

    python -m showcase_pipeline.landing_examples run    --prerun <p> ...
    python -m showcase_pipeline.landing_examples list   --prerun <p>
    python -m showcase_pipeline.landing_examples export --prerun <p> --pick ...
"""
