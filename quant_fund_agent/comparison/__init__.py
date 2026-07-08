"""Compare the factor output of different *research* LLMs.

A named **prerun** (``quant_fund_agent/factors/preruns.py``) is one batch of
factor research mined by a chosen LLM.  This package evaluates and compares the
factor sets of several preruns on three increasingly-integrated axes:

1. **single-factor IC** (:mod:`~quant_fund_agent.comparison.ic`) — the raw
   Pearson IC of every researched factor on a shared panel
   (``backtesting/engine.py``); "how good is each factor on its own?";
2. **brute-force ML** (:mod:`~quant_fund_agent.comparison.bruteforce`) — the
   factors fed straight into the model catalog (+ an ensemble), no agents;
   "are these factors good raw features?";
3. **downstream agents** (:mod:`~quant_fund_agent.comparison.downstream`) — the
   factors run through the full Selector → Architect → Statistician → PM
   pipeline; "do the agents build a good fund from them?".

Tracks 1–2 are deterministic and LLM-free; only track 3 (and creating the
preruns) spends LLM budget.  :mod:`~quant_fund_agent.comparison.plots` and
:mod:`~quant_fund_agent.comparison.report` turn the results into
presentation-ready figures, tables, a Markdown report and a notebook under
``data/comparisons/<id>/``.  ``run_model_comparison.py`` drives the whole thing.
"""

from quant_fund_agent.comparison.config import ComparisonConfig

__all__ = ["ComparisonConfig"]
