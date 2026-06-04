"""Stage 2: Selector Agent.

The Selector reads the factor catalog and, using only the *economic* description
of each factor (deliberately NOT its IC numbers), proposes one trading hypothesis
and picks the factors that best serve it:

    load_factor_catalog → formulate_hypothesis → select_factors

This stage needs no market data — it reasons purely over the factor catalog — so
it runs identically to production.  It reads the **real** factor database, so any
factors kept by Stage 1 are part of the catalog it chooses from.

Usage::

    ./venv/bin/python run_pipeline/2_selector.py
    ./venv/bin/python run_pipeline/2_selector.py --mcp
"""

from __future__ import annotations

import argparse
import logging

import pipeline_common as pc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_common_args(p, with_data=False)  # --mcp only (no market data needed)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.banner("STAGE 2 · SELECTOR")
    print("Reads the factor catalog → formulates a hypothesis → selects factors.")

    pc.require_openai_key()
    pc.configure_runtime(use_mcp=args.mcp)

    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.factors import discover_factors

    discover_factors()

    result = selector_graph.invoke({})

    catalog = result.get("factor_catalog", [])
    selected = result.get("selected_factor_ids", [])

    pc.section(f"FACTOR CATALOG ({len(catalog)} factors available)")
    for f in catalog:
        print(f"  - {f.factor_id} | {f.name} | cat={f.category}")

    pc.section("TRADING HYPOTHESIS")
    print(result.get("hypothesis", "(none)"))

    pc.section("REASONING")
    print(result.get("reasoning", "(none)"))

    pc.section(f"SELECTED FACTORS ({len(selected)})")
    for fid in selected:
        match = next((f for f in catalog if f.factor_id == fid), None)
        print(f"  - {fid}  ({match.name if match else '?'})")

    pc.section("SELECTION RATIONALE")
    print(result.get("selection_rationale", "(none)"))

    pc.banner("STAGE 2 COMPLETE")
    print("  The hypothesis + selected factors feed the Architect.")
    print("  Next: ./venv/bin/python run_pipeline/3_architect.py")


if __name__ == "__main__":
    main()
