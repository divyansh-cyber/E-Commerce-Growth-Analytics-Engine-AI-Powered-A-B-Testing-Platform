"""
ab_testing.py
-------------
Statistical experimentation engine.

Implements:
  • Sample Ratio Mismatch (SRM) guard via Chi-Square
  • Two-Sample Welch's T-Test  →  Average Order Value (continuous)
  • Chi-Square Test of Independence  →  Conversion Rate (categorical)
  • Confidence Intervals (Wilson method for proportions, analytical for means)
  • Effect sizes (Cohen's d, Cohen's h)
  • Statistical power and Minimum Detectable Effect (MDE)
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")


class ABTestEngine:
    """
    Runs a full A/B test analysis on a DataFrame with columns:
        user_id, group_name, converted, order_value, session_duration_sec
    """

    def __init__(self, df: pd.DataFrame):
        self.df        = df.copy()
        self.control   = df[df["group_name"] == "control"].copy()
        self.treatment = df[df["group_name"] == "treatment"].copy()

    # ── 1. Sample Ratio Mismatch ─────────────────────────────────────────────

    def check_srm(self, expected_split: float = 0.50) -> dict:
        """
        Guard-rail: test whether the observed group sizes deviate significantly
        from the expected randomisation ratio using a one-way Chi-Square test.

        A significant result (p < 0.01) indicates data integrity issues
        (e.g., bot traffic, logging bugs) and the experiment results should NOT
        be trusted.
        """
        n_ctrl = len(self.control)
        n_trt  = len(self.treatment)
        n_tot  = n_ctrl + n_trt

        exp_ctrl = n_tot * expected_split
        exp_trt  = n_tot * (1 - expected_split)

        chi2, p_value = stats.chisquare(
            f_obs=[n_ctrl, n_trt],
            f_exp=[exp_ctrl, exp_trt],
        )

        srm_detected = bool(p_value < 0.01)
        return {
            "n_control"       : n_ctrl,
            "n_treatment"     : n_trt,
            "n_total"         : n_tot,
            "expected_split"  : expected_split,
            "actual_split"    : round(n_ctrl / n_tot, 4),
            "chi2_statistic"  : round(float(chi2), 4),
            "p_value"         : round(float(p_value), 4),
            "srm_detected"    : srm_detected,
            "interpretation"  : (
                "⚠️  SRM DETECTED — group sizes deviate significantly from the "
                "expected 50/50 split. Investigate before drawing conclusions."
                if srm_detected else
                "✅  No SRM detected — randomisation looks clean."
            ),
        }

    # ── 2. Conversion Rate (Chi-Square) ─────────────────────────────────────

    def test_conversion_rate(self, alpha: float = 0.05) -> dict:
        """
        Chi-Square Test of Independence on a 2×2 contingency table.

        H0: Conversion rate is identical for control and treatment.
        H1: Conversion rates differ.
        """
        c_conv  = int(self.control["converted"].sum())
        c_tot   = len(self.control)
        t_conv  = int(self.treatment["converted"].sum())
        t_tot   = len(self.treatment)

        c_rate = c_conv / c_tot
        t_rate = t_conv / t_tot

        contingency = np.array([
            [c_conv,       c_tot - c_conv],
            [t_conv,       t_tot - t_conv],
        ])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency, correction=False)

        # Wilson confidence intervals (better than normal approx for proportions)
        c_ci = proportion_confint(c_conv, c_tot, alpha=alpha, method="wilson")
        t_ci = proportion_confint(t_conv, t_tot, alpha=alpha, method="wilson")

        # Cohen's h (effect size for proportions)
        cohens_h = float(
            2 * np.arcsin(np.sqrt(t_rate)) - 2 * np.arcsin(np.sqrt(c_rate))
        )

        # Statistical power
        power = TTestIndPower().solve_power(
            effect_size=abs(cohens_h),
            nobs1=c_tot,
            ratio=t_tot / c_tot,
            alpha=alpha,
        )

        rel_uplift = (t_rate - c_rate) / c_rate

        is_sig = bool(p_val < alpha)
        winner = (
            "treatment" if (is_sig and t_rate > c_rate) else
            "control"   if (is_sig and c_rate > t_rate) else
            "no_winner"
        )

        return {
            "metric"            : "Conversion Rate",
            "test_type"         : "Chi-Square (Pearson)",
            "ctrl_rate"         : round(c_rate, 4),
            "trt_rate"          : round(t_rate, 4),
            "ctrl_conversions"  : c_conv,
            "trt_conversions"   : t_conv,
            "ctrl_total"        : c_tot,
            "trt_total"         : t_tot,
            "absolute_uplift"   : round(t_rate - c_rate, 4),
            "relative_uplift_pct": round(rel_uplift * 100, 2),
            "chi2_statistic"    : round(float(chi2), 4),
            "p_value"           : round(float(p_val), 6),
            "degrees_of_freedom": int(dof),
            "ctrl_ci_lower"     : round(c_ci[0], 4),
            "ctrl_ci_upper"     : round(c_ci[1], 4),
            "trt_ci_lower"      : round(t_ci[0], 4),
            "trt_ci_upper"      : round(t_ci[1], 4),
            "cohens_h"          : round(cohens_h, 4),
            "statistical_power" : round(float(power), 4),
            "alpha"             : alpha,
            "is_significant"    : is_sig,
            "winner"            : winner,
        }

    # ── 3. Average Order Value (Welch's T-Test) ─────────────────────────────

    def test_average_order_value(self, alpha: float = 0.05) -> dict:
        """
        Welch's (unequal-variance) Two-Sample T-Test on AOV among converters.

        H0: Mean AOV is identical for control and treatment converters.
        H1: Mean AOV differs.

        Note: Only converted users have an order_value — this is intentional.
        """
        c_aov = self.control[self.control["converted"] == 1]["order_value"].dropna()
        t_aov = self.treatment[self.treatment["converted"] == 1]["order_value"].dropna()

        c_mean, c_std, n1 = c_aov.mean(), c_aov.std(), len(c_aov)
        t_mean, t_std, n2 = t_aov.mean(), t_aov.std(), len(t_aov)

        # Welch's T-Test
        t_stat, p_val = stats.ttest_ind(t_aov, c_aov, equal_var=False)

        # Welch–Satterthwaite degrees of freedom
        dof = (
            (c_aov.var() / n1 + t_aov.var() / n2) ** 2
            / (
                (c_aov.var() / n1) ** 2 / (n1 - 1)
                + (t_aov.var() / n2) ** 2 / (n2 - 1)
            )
        )

        # 95% CI on the mean difference
        se_diff = np.sqrt(c_aov.var() / n1 + t_aov.var() / n2)
        t_crit  = stats.t.ppf(1 - alpha / 2, df=dof)
        diff    = t_mean - c_mean
        ci_lo   = diff - t_crit * se_diff
        ci_hi   = diff + t_crit * se_diff

        # Cohen's d (pooled std)
        pooled_std = np.sqrt(
            (c_aov.var() * (n1 - 1) + t_aov.var() * (n2 - 1)) / (n1 + n2 - 2)
        )
        cohens_d = diff / pooled_std

        # Statistical power
        power = TTestIndPower().solve_power(
            effect_size=abs(cohens_d),
            nobs1=n1,
            ratio=n2 / n1,
            alpha=alpha,
        )

        # MDE at 80% power (expressed in dollars)
        mde_es = TTestIndPower().solve_power(
            effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=alpha, power=0.80
        )
        mde_dollars = mde_es * pooled_std

        is_sig = bool(p_val < alpha)
        winner = (
            "treatment" if (is_sig and t_mean > c_mean) else
            "control"   if (is_sig and c_mean > t_mean) else
            "no_winner"
        )

        return {
            "metric"              : "Average Order Value",
            "test_type"           : "Welch's Two-Sample T-Test",
            "ctrl_mean"           : round(float(c_mean), 2),
            "trt_mean"            : round(float(t_mean), 2),
            "ctrl_std"            : round(float(c_std), 2),
            "trt_std"             : round(float(t_std), 2),
            "ctrl_n"              : int(n1),
            "trt_n"               : int(n2),
            "mean_difference"     : round(float(diff), 2),
            "ci_lower"            : round(float(ci_lo), 2),
            "ci_upper"            : round(float(ci_hi), 2),
            "relative_uplift_pct" : round(float(diff / c_mean * 100), 2),
            "t_statistic"         : round(float(t_stat), 4),
            "p_value"             : round(float(p_val), 6),
            "degrees_of_freedom"  : round(float(dof), 1),
            "cohens_d"            : round(float(cohens_d), 4),
            "statistical_power"   : round(float(power), 4),
            "mde_effect_size"     : round(float(mde_es), 4),
            "mde_dollars"         : round(float(mde_dollars), 2),
            "alpha"               : alpha,
            "is_significant"      : is_sig,
            "winner"              : winner,
            # Raw arrays for visualisation
            "ctrl_values"         : c_aov.tolist(),
            "trt_values"          : t_aov.tolist(),
        }

    # ── 4. Full Analysis Orchestrator ────────────────────────────────────────

    def run_full_analysis(self, alpha: float = 0.05) -> dict:
        """Run all three tests and return a unified results dict."""
        srm        = self.check_srm()
        conversion = self.test_conversion_rate(alpha)
        aov        = self.test_average_order_value(alpha)

        recommendation = self._get_recommendation(srm, conversion, aov)

        return {
            "srm_check"          : srm,
            "conversion_rate"    : conversion,
            "average_order_value": aov,
            "experiment_summary" : {
                "total_users"              : srm["n_total"],
                "test_duration_days"       : 30,
                "primary_metric_winner"    : conversion["winner"],
                "secondary_metric_winner"  : aov["winner"],
                "overall_recommendation"   : recommendation,
                "alpha"                    : alpha,
            },
        }

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_recommendation(srm: dict, conv: dict, aov: dict) -> str:
        if srm["srm_detected"]:
            return "🚨 INVALID — Fix data pipeline before making any decisions."
        c_sig = conv["is_significant"]
        c_win = conv["winner"] == "treatment"
        a_sig = aov["is_significant"]
        a_win = aov["winner"] == "treatment"

        if c_sig and c_win and a_sig and a_win:
            return "🚀 SHIP — Strong evidence on BOTH conversion rate and AOV."
        if c_sig and c_win:
            return "✅ SHIP — Significant lift in conversion rate; AOV is neutral."
        if c_sig and not c_win:
            return "🛑 DO NOT SHIP — Treatment hurts conversion rate."
        if a_sig and a_win and not c_sig:
            return "⚠️  ITERATE — AOV improved but primary metric inconclusive."
        return "🔄 INCONCLUSIVE — Continue test or increase sample size."
