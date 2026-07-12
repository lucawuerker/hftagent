"""Evolutionary Execution Researcher — the thin arm over the shared machinery.

Reuses the factor arm's genome-agnostic `EvolutionController` (NSGA-II /
archive / N_trials / islands / lineage / checkpoint) verbatim; swaps only the
program type (`ExecutionProgram`), the mutation operators (param-jitter in E1,
LLM-semantic in E2) and the evaluation call
(`research_client.evaluate_executor_fitness`).
"""

from quant_fund_agent.agents.execution_research.evolution.genome import (  # noqa: F401
    ExecutionProgram,
)
from quant_fund_agent.agents.execution_research.evolution.loop import (  # noqa: F401
    ExecEvolutionLoop,
    ExecEvolutionRunConfig,
)
