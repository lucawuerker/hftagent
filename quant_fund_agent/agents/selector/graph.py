"""Selector Agent – LangGraph subgraph.

Workflow
-------
1. **load_factor_catalog** – read factor_db.json, build a compact summary
   of every factor (id, category, description, IC/ICIR at each horizon).
2. **formulate_hypothesis** – LLM proposes a trading hypothesis grounded in
   the available factor catalog.
3. **select_factors** – LLM picks the specific factors that best serve the
   hypothesis and explains why.

The output (``SelectorState.selected_factor_ids`` + ``hypothesis``) is
passed downstream to the Architect Agent.
"""

from __future__ import annotations

import json
import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.selector.state import FactorSummary, SelectorState


LLM_MODEL = os.getenv("SELECTOR_LLM_MODEL", "gpt-4o-mini")


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.7)


# ── node 1: load factor catalog ───────────────────────────────────────

def load_factor_catalog(state: SelectorState) -> dict:
    """Read the persisted factor database and build a compact catalog.

    The catalog is loaded through the quant-catalog MCP server (with an
    in-process fallback); the returned JSON rows are hydrated into
    ``FactorSummary`` models for the downstream LLM nodes.
    """
    from quant_fund_agent.mcp import catalog_client

    rows = catalog_client.load_factor_catalog()
    catalog = [FactorSummary(**row) for row in rows]
    return {"factor_catalog": catalog}


# ── node 2: formulate hypothesis ──────────────────────────────────────

HYPOTHESIS_PROMPT = """\
You are a senior quantitative researcher at a systematic trading firm.

Below is a catalog of alpha factors that have been backtested on 10-second
intraday US equity data (50 large-cap stocks, 1 month).  Each factor has
IC and ICIR at three horizons: 1 bar (10 s), 6 bars (1 min), 60 bars (10 min).

FACTOR CATALOG
--------------
{catalog}

Your task:
1. Study the factors, their categories, descriptions, and IC/ICIR values.
2. Formulate ONE clear, specific trading hypothesis that could be
   implemented by combining several of these factors.
3. The hypothesis should be grounded in quantitative finance theory
   (mean-reversion, momentum, microstructure, behavioural finance, etc.) and should explain
   *why* the combination would work.

Respond in JSON with exactly two keys:
  "hypothesis": a 2-3 sentence description of the trading idea,
  "reasoning": a short paragraph explaining the financial logic.
"""


def formulate_hypothesis(state: SelectorState) -> dict:
    """Use the LLM to propose a trading hypothesis."""
    catalog_text = "\n".join(
        f"- {f.factor_id} | {f.name} | cat={f.category} | "
        f"IC(10s)={f.ic_1} ICIR(10s)={f.icir_1} | "
        f"IC(1m)={f.ic_6} ICIR(1m)={f.icir_6} | "
        f"IC(10m)={f.ic_60} ICIR(10m)={f.icir_60}\n"
        f"  desc: {f.description}"
        for f in state.factor_catalog
    )

    llm = _get_llm()
    resp = llm.invoke(HYPOTHESIS_PROMPT.format(catalog=catalog_text))
    content = resp.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])

    return {
        "hypothesis": parsed.get("hypothesis", ""),
        "reasoning": parsed.get("reasoning", ""),
    }


# ── node 3: select factors ───────────────────────────────────────────

SELECT_PROMPT = """\
You are a senior quantitative researcher.

TRADING HYPOTHESIS
------------------
{hypothesis}

REASONING
---------
{reasoning}

FACTOR CATALOG
--------------
{catalog}

Your task:
Select between as many factors as you see fit from the catalog that best serve the
hypothesis above.  Consider:
- IC and ICIR values (higher absolute values are better)
- Category alignment with the hypothesis
- Diversity (don't pick 5 factors that are essentially the same signal)
- Coverage across horizons if the hypothesis involves multiple time scales

Respond in JSON with exactly two keys:
  "selected_factor_ids": a list of factor_id strings,
  "rationale": a short paragraph explaining each pick.
"""


def select_factors(state: SelectorState) -> dict:
    """Use the LLM to pick factors that fit the hypothesis."""
    catalog_text = "\n".join(
        f"- {f.factor_id} | {f.name} | cat={f.category} | "
        f"IC(10s)={f.ic_1} ICIR(10s)={f.icir_1} | "
        f"IC(1m)={f.ic_6} ICIR(1m)={f.icir_6} | "
        f"IC(10m)={f.ic_60} ICIR(10m)={f.icir_60}\n"
        f"  desc: {f.description}"
        for f in state.factor_catalog
    )

    llm = _get_llm()
    resp = llm.invoke(SELECT_PROMPT.format(
        hypothesis=state.hypothesis,
        reasoning=state.reasoning,
        catalog=catalog_text,
    ))
    content = resp.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])

    known_ids = {f.factor_id for f in state.factor_catalog}
    selected = [fid for fid in parsed.get("selected_factor_ids", []) if fid in known_ids]

    return {
        "selected_factor_ids": selected,
        "selection_rationale": parsed.get("rationale", ""),
    }


# ── graph construction ────────────────────────────────────────────────

def build_selector_graph() -> StateGraph:
    graph = StateGraph(SelectorState)

    graph.add_node("load_factor_catalog", load_factor_catalog)
    graph.add_node("formulate_hypothesis", formulate_hypothesis)
    graph.add_node("select_factors", select_factors)

    graph.add_edge(START, "load_factor_catalog")
    graph.add_edge("load_factor_catalog", "formulate_hypothesis")
    graph.add_edge("formulate_hypothesis", "select_factors")
    graph.add_edge("select_factors", END)

    return graph


selector_graph = build_selector_graph().compile()
