"""The execution layer — signal → target-book programs (``BaseExecutor``).

E0 of ``docs/execution-evolution/DESIGN.md``: the single home for the
signal→positions mapping (previously hardcoded in two divergent pipelines),
plus the codegen/validation and frozen-signal machinery the evolutionary
Execution Researcher builds on.
"""

from quant_fund_agent.execution.base import (  # noqa: F401
    EXECUTOR_REGISTRY,
    BaseExecutor,
    BookState,
    get_executor,
    list_executors,
    register_executor,
    run_executor,
    validate_weights,
)
