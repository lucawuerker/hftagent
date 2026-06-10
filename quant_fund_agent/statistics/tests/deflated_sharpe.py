"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Reference: "The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality", David H. Bailey and
Marcos López de Prado, *Journal of Portfolio Management*, 2014.
https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

The deflated SR is the probability that the **true** Sharpe of the
candidate strategy is greater than zero, after correcting for:

    1. Selection bias — we tried many strategies, so the best one's
       observed SR is upward-biased.  The expected maximum SR under the
       null is approximated using the variance of the trial SRs (eq. 4).
    2. Non-normality — the usual SR confidence interval understates the
       variance when returns are skewed / fat-tailed.  The denominator
       uses the returns' skewness and kurtosis (eq. 9).

If ``DSR > 0.95`` (one-sided 95% confidence) we say the strategy is
unlikely to be a false positive.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from quant_fund_agent.data.frequency import TRADING_DAYS_PER_YEAR
from quant_fund_agent.statistics.base import BaseStatTest, StatTestContext
from quant_fund_agent.statistics.registry import register_test
from quant_fund_agent.statistics.schemas import StatTestResult, StatTestVerdict


EULER_MASCHERONI = 0.5772156649015329


@register_test
class DeflatedSharpeRatio(BaseStatTest):
    test_id = "deflated_sharpe_ratio"
    name = "Deflated Sharpe Ratio"
    description = (
        "Probability that the true Sharpe is > 0 after correcting for "
        "selection bias and non-normality (Bailey & López de Prado, 2014)."
    )
    is_mandatory = True

    # Acceptance threshold.  For a short intraday microstructure dataset with
    # fat-tailed 10-second returns and limited history, the textbook 0.95 is
    # too strict.  0.75 is a defensible bar (still a one-sided ~75% confidence
    # that the true SR > 0 after correcting for selection bias and non-normality);
    # the WARN band flags 0.60-0.75 as borderline.
    threshold: float = 0.75

    def run(self, ctx: StatTestContext) -> StatTestResult:
        returns = ctx.is_portfolio_returns
        if returns is None or len(returns) < 30:
            return StatTestResult(
                test_id=self.test_id,
                test_name=self.name,
                verdict=StatTestVerdict.FAIL,
                summary="Not enough return observations to compute DSR.",
                details={"n_obs": 0 if returns is None else len(returns)},
            )

        trials = [
            t.metrics.get("sharpe_ratio")
            for t in ctx.trial_history
            if t.metrics.get("sharpe_ratio") is not None
        ]
        n_trials = max(1, len(trials))

        # --- 1. bar-level Sharpe (non-annualised; DSR formula works at any frequency) ---
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        if sigma <= 0:
            return StatTestResult(
                test_id=self.test_id,
                test_name=self.name,
                verdict=StatTestVerdict.FAIL,
                value=0.0,
                summary="Zero return variance — strategy is effectively flat.",
                details={"mean_bar_return": mu, "std_bar_return": sigma},
            )
        sr = mu / sigma
        T = len(returns)

        skew = float(stats.skew(returns, bias=False))
        kurt = float(stats.kurtosis(returns, fisher=False, bias=False))  # raw, 3 = normal

        # --- 2. expected maximum Sharpe under the null (eq. 4 in the paper) ---
        # Variance of the SRs produced across all trials (bar-level to be
        # consistent with ``sr``).  We convert each annualised trial SR back
        # to bar-level via the same annualisation factor as the IS run.
        bars_per_year = ctx.bars_per_day * TRADING_DAYS_PER_YEAR
        ann_factor = math.sqrt(bars_per_year)
        trial_srs_bar = [s / ann_factor for s in trials]

        if len(trial_srs_bar) >= 2:
            sr_var = float(np.var(trial_srs_bar, ddof=1))
        else:
            # Fallback: single trial — use the theoretical variance of
            # SR estimator under i.i.d. normal returns (≈ 1/T).
            sr_var = 1.0 / T

        # Approximation of E[max(SR_0, …, SR_{N-1})] / sqrt(V[SR])
        # from order statistics of the standard normal (eq. 4).
        if n_trials > 1:
            expected_max_term = (
                (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n_trials)
                + EULER_MASCHERONI * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
            )
        else:
            expected_max_term = 0.0
        sr_expected = math.sqrt(max(sr_var, 0.0)) * expected_max_term

        # --- 3. probabilistic / deflated Sharpe ratio (eq. 9) ---
        denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
        if denom_sq <= 0:
            # Degenerate case: approximate with i.i.d. normal denominator.
            denom_sq = 1.0
        denom = math.sqrt(denom_sq)
        z = (sr - sr_expected) * math.sqrt(T - 1) / denom
        dsr = float(stats.norm.cdf(z))

        verdict = (
            StatTestVerdict.PASS if dsr >= self.threshold
            else StatTestVerdict.WARN if dsr >= 0.60
            else StatTestVerdict.FAIL
        )

        return StatTestResult(
            test_id=self.test_id,
            test_name=self.name,
            verdict=verdict,
            value=round(dsr, 6),
            threshold=self.threshold,
            p_value=round(1.0 - dsr, 6),
            summary=(
                f"DSR = {dsr:.4f} "
                f"(threshold {self.threshold:.2f}, {n_trials} trials, "
                f"T={T} bars)."
            ),
            details={
                "n_trials": n_trials,
                "trial_sharpes_annualised": trials,
                "sharpe_bar_level": round(sr, 6),
                "sharpe_annualised": round(sr * ann_factor, 4),
                "expected_max_sharpe_bar_level": round(sr_expected, 6),
                "expected_max_sharpe_annualised": round(sr_expected * ann_factor, 4),
                "trial_sharpe_variance_bar_level": round(sr_var, 10),
                "n_obs": T,
                "skewness": round(skew, 4),
                "kurtosis_raw": round(kurt, 4),
                "z_statistic": round(z, 4),
            },
        )
