"""Showcase — Stage 2: Selector Agent (no market data required).

The Selector reads the factor catalog and, using only the *economic*
description of each factor (deliberately NOT its IC numbers), proposes one
trading hypothesis and picks the factors that best serve it.

This stage needs no order-book / price data at all — it reasons purely over the
factor catalog — so it runs end-to-end here, identically to production.  It
reads the *showcase* factor database, so any factors created by Stage 1
(``1_factor_research.py``) are part of the catalog it chooses from.

    load_factor_catalog  →  formulate_hypothesis  →  select_factors

Usage::

    ./venv/bin/python showcase_pipeline/2_selector.py
"""

from __future__ import annotations

import logging

import showcase_common as sc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    sc.banner("SHOWCASE · STAGE 2 · SELECTOR")
    print("Reads the factor catalog → formulates a hypothesis → selects factors.")
    print("Needs no market data: it reasons over factor economics, not IC values.")

    sc.require_openai_key()
    sc.ensure_showcase_factor_db()

    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.factors import discover_factors

    discover_factors()

    result = selector_graph.invoke({})

    catalog = result.get("factor_catalog", [])
    selected = result.get("selected_factor_ids", [])

    sc.section(f"FACTOR CATALOG ({len(catalog)} factors available)")
    print(f"  Source: {sc.FACTOR_DB_SHOWCASE}")
    for f in catalog:
        print(f"  - {f.factor_id} | {f.name} | cat={f.category}")

    sc.section("TRADING HYPOTHESIS")
    print(result.get("hypothesis", "(none)"))

    sc.section("REASONING")
    print(result.get("reasoning", "(none)"))

    sc.section(f"SELECTED FACTORS ({len(selected)})")
    for fid in selected:
        match = next((f for f in catalog if f.factor_id == fid), None)
        name = match.name if match else "?"
        print(f"  - {fid}  ({name})")

    sc.section("SELECTION RATIONALE")
    print(result.get("selection_rationale", "(none)"))

    sc.banner("STAGE 2 COMPLETE")
    print("  The hypothesis + selected factors feed the Architect.")
    print("  Next: ./venv/bin/python showcase_pipeline/3_architect.py")


if __name__ == "__main__":
    main()
