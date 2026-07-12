"""Genome #0 — the two baseline pipelines as *self-contained* executor source.

The deployment registry (``quant_fund_agent/execution/seeds.py``) delegates to
the legacy functions; the evolution seeds below re-express the SAME logic as
standalone code, because a genome must be mutable — the LLM (E2) and the param
jitter (E1) operate on the actual construction logic, not a delegation wrapper.
Byte-equivalence of both representations against the legacy functions is
enforced by ``tests/test_exec_evolution_loop.py`` — if that test fails, the
baseline arm of every experiment is invalid.

Note the numeric ``params`` dicts: they are the declared jitter surface.
"""

from __future__ import annotations

from quant_fund_agent.agents.execution_research.evolution.genome import (
    ExecutionProgram,
)

TOPK_SEED_CODE = '''\
"""Baseline cross-sectional executor: z-score, top-K by |z|, dollar-neutral."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class SeedTopKDollarNeutral(BaseExecutor):
    executor_id = "seed_topk_dollar_neutral"
    name = "Seed: top-K dollar-neutral"
    regime = "cross_sectional"
    inputs = ["signal"]
    params = {"holding_period": 1, "max_positions": 20, "min_conviction": 0.0}

    def target_weights(self, signal, state):
        p = type(self).params
        hp = int(p["holding_period"])
        if hp > 1:
            mask = np.arange(len(signal)) % hp == 0
            signal = signal.where(pd.Series(mask, index=signal.index), other=np.nan)
            signal = signal.ffill()
        mu = signal.mean(axis=1)
        std = signal.std(axis=1).replace(0, np.nan)
        z = signal.sub(mu, axis=0).div(std, axis=0)
        mc = float(p["min_conviction"])
        if mc > 0:
            z = z.where(z.abs() >= mc, other=0.0)
        rk = z.abs().rank(axis=1, ascending=False, method="first")
        z = z.where(rk <= int(p["max_positions"]), other=0.0)
        abs_sum = z.abs().sum(axis=1).replace(0, np.nan)
        return z.div(abs_sum, axis=0)
'''

ZTHRESH_SEED_CODE = '''\
"""Baseline per-underlying executor: causal z-score, threshold band, 1/n."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class SeedZScoreThreshold(BaseExecutor):
    executor_id = "seed_zscore_threshold"
    name = "Seed: z-threshold equal-weight"
    regime = "per_underlying"
    inputs = ["signal"]
    params = {"n_max_positions": 6, "holding_period": 1, "threshold": 1.0}

    def target_weights(self, signal, state):
        p = type(self).params
        hp = int(p["holding_period"])
        if hp > 1:
            mask = np.arange(len(signal)) % hp == 0
            signal = signal.where(pd.Series(mask, index=signal.index), other=np.nan)
            signal = signal.ffill()
        m = signal.expanding(min_periods=2).mean()
        s = signal.expanding(min_periods=2).std()
        z = (signal - m) / s.replace(0, np.nan)
        t = float(p["threshold"])
        pos = pd.DataFrame(0.0, index=z.index, columns=z.columns)
        pos = pos.mask(z > t, 1.0).mask(z < -t, -1.0)
        return pos / float(max(1, int(p["n_max_positions"])))
'''


def seed_execution_programs() -> list[ExecutionProgram]:
    """The two baseline pipelines as evolution genomes (no LLM call needed)."""
    return [
        ExecutionProgram(
            executor_id="seed_topk_dollar_neutral",
            code=TOPK_SEED_CODE,
            name="Seed: top-K dollar-neutral",
            regime="cross_sectional",
            mechanism="cross-sectional relative-value: trade the extremes of "
                      "the signal cross-section, dollar-neutral",
            expected_effect="reproduces the deployed cross-sectional baseline "
                            "book exactly",
        ),
        ExecutionProgram(
            executor_id="seed_zscore_threshold",
            code=ZTHRESH_SEED_CODE,
            name="Seed: z-threshold equal-weight",
            regime="per_underlying",
            mechanism="directional per-name band: act only on |z|>threshold "
                      "extremes of each name's own history",
            expected_effect="reproduces the deployed per-underlying baseline "
                            "book exactly",
        ),
    ]
