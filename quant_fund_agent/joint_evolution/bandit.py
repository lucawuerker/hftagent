"""The contextual Thompson-sampling block scheduler (J2) — RD-Agent(Q) adapted.

Arms ``{factor, exec}``; one Bayesian linear regression per arm; a Thompson
step samples coefficients from each posterior and picks the arm with the
higher sampled expected reward.  Reward = the block's **deflated ΔJ** (already
deflated by the joint look count, so it cannot drift upward merely because
looks accumulated).

Honest cold-start handling — a thesis run has ~5–15 block decisions, not
RD-Agent's 30+ iterations:

* the first pick of an UNSEEN arm is forced (one observation per arm before
  any Thompson draw; block 0 is factor by the outer loop's rule anyway);
* a heavy-shrinkage prior (``prior_scale``) keeps posteriors wide, so
  exploration survives small n;
* ``context="off"`` degenerates to non-contextual Gaussian TS on rewards
  alone (a simpler, lower-variance arm we also report);
* the RNG is seeded per (seed, decision index), so a resumed run repeats the
  identical schedule.

Context vector (``context="on"``), derived deterministically from the outer
loop's block history (the last row's joint-score diagnostics)::

    [1 (bias), J, val_net_sharpe, val_gross_sharpe, dsr_prob,
     mean_turnover, book_size/10, frozen_version/5]

Posterior state serialises into ``joint_state.json`` so a resumed run picks up
exactly where it stopped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund_agent.joint_evolution.scheduler import Scheduler

ARMS = ("factor", "exec")
DIM = 8


def context_vector(history: list[dict[str, Any]]) -> np.ndarray:
    """The 8-dim state the contextual bandit conditions on (deterministic)."""
    x = np.zeros(DIM)
    x[0] = 1.0                                    # bias
    if not history:
        return x
    last = history[-1]
    js = last.get("joint_score") or {}

    def _f(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    x[1] = _f(last.get("J_after"))
    x[2] = _f(js.get("val_net_sharpe"))
    x[3] = _f(js.get("val_gross_sharpe"))
    x[4] = _f(js.get("dsr_prob"), 0.5)
    x[5] = _f(js.get("mean_turnover"))
    x[6] = _f(js.get("n_book")) / 10.0
    x[7] = _f(last.get("frozen_signals_version")) / 5.0
    return x


class BanditScheduler(Scheduler):
    """Per-arm Bayesian linear regression + Thompson sampling."""

    kind = "bandit"

    def __init__(self, seed: int = 0, *, context: str = "on",
                 prior_scale: float = 1.0, noise_var: float = 0.01):
        self.seed = int(seed)
        self.context = context                    # "on" | "off"
        self.prior_scale = float(prior_scale)
        self.noise_var = float(noise_var)
        self.n_decisions = 0
        # per-arm posterior: precision P (DIM×DIM) and vector b, with
        # θ_hat = P⁻¹ b  (standard Bayesian ridge bookkeeping)
        self._P = {a: np.eye(DIM) / self.prior_scale for a in ARMS}
        self._b = {a: np.zeros(DIM) for a in ARMS}
        self._n_obs = {a: 0 for a in ARMS}
        self._last_context: dict[str, list[float]] = {}

    # ── the TS step ───────────────────────────────────────────────────────────

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1_000_003 + self.n_decisions)

    def _sample_value(self, arm: str, x: np.ndarray,
                      rng: np.random.Generator) -> float:
        # Standard Bayesian ridge: Λ = Λ₀ + Σxx'/σ²,  μ = Λ⁻¹·(Σx·r)/σ²,
        # posterior cov = Λ⁻¹.  (_P accumulates Λ, _b accumulates Σx·r.)
        P, b = self._P[arm], self._b[arm]
        cov = np.linalg.inv(P)
        mu = cov @ (b / self.noise_var)
        theta = rng.multivariate_normal(mu, cov, method="cholesky")
        return float(x @ theta)

    def pick(self, block_index: int, history: list[dict[str, Any]]) -> str:
        self.n_decisions += 1
        # warmup: force one observation per arm before any Thompson draw
        for arm in ARMS:
            if self._n_obs[arm] == 0:
                seen = {r.get("arm") for r in history}
                if arm not in seen:
                    self._last_context[arm] = list(context_vector(history))
                    return arm
        x = (context_vector(history) if self.context == "on"
             else np.array([1.0] + [0.0] * (DIM - 1)))
        rng = self._rng()
        values = {arm: self._sample_value(arm, x, rng) for arm in ARMS}
        chosen = max(ARMS, key=lambda a: values[a])
        self._last_context[chosen] = [float(v) for v in x]
        return chosen

    def update(self, arm: str, reward: float | None,
               context: Any | None = None) -> None:
        if arm not in ARMS or reward is None:
            return
        x = np.asarray(context if context is not None
                       else self._last_context.get(arm, [1.0] + [0.0] * (DIM - 1)),
                       dtype=float)
        if len(x) != DIM:
            x = np.resize(x, DIM)
        self._P[arm] = self._P[arm] + np.outer(x, x) / self.noise_var
        self._b[arm] = self._b[arm] + x * float(reward)
        self._n_obs[arm] += 1

    # ── persistence ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "seed": self.seed, "context": self.context,
            "prior_scale": self.prior_scale, "noise_var": self.noise_var,
            "n_decisions": self.n_decisions,
            "P": {a: self._P[a].tolist() for a in ARMS},
            "b": {a: self._b[a].tolist() for a in ARMS},
            "n_obs": dict(self._n_obs),
            "last_context": dict(self._last_context),
        }

    def load_state(self, d: dict[str, Any]) -> None:
        if d.get("kind") != "bandit":
            return
        self.seed = int(d.get("seed", self.seed))
        self.context = d.get("context", self.context)
        self.prior_scale = float(d.get("prior_scale", self.prior_scale))
        self.noise_var = float(d.get("noise_var", self.noise_var))
        self.n_decisions = int(d.get("n_decisions", 0))
        for a in ARMS:
            if a in d.get("P", {}):
                self._P[a] = np.asarray(d["P"][a], dtype=float)
            if a in d.get("b", {}):
                self._b[a] = np.asarray(d["b"][a], dtype=float)
        self._n_obs = {a: int(d.get("n_obs", {}).get(a, 0)) for a in ARMS}
        self._last_context = {k: list(v)
                              for k, v in d.get("last_context", {}).items()}
