"""E-Commerce Conversion, AOV & A/B Test Analytics Simulator.

Models commercial storefront impact (AOV, GMV, CTR, attach rate)
and simulates a 30-day randomized A/B test with Student's t-test and proportion z-tests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CommercialImpactModel:
    """Calculates unit economics, conversion funnel, and annualized GMV uplift."""

    # Baseline: Standard Lexical Keyword Search
    baseline_aov: float = 1850.0
    baseline_ctr: float = 0.142            # 14.2% Search-to-PDP CTR
    baseline_attach_rate: float = 1.18     # items per transaction
    baseline_conv_rate: float = 0.024      # 2.4% Search-to-Order Conversion Rate
    baseline_abandonment_rate: float = 0.280 # 28.0% zero-result / bounce rate

    # Variant: ContextStyle Semantic & Bundling Engine
    variant_aov: float = 2480.0
    variant_ctr: float = 0.225             # 22.5% Search-to-PDP CTR
    variant_attach_rate: float = 2.45      # items per transaction
    variant_conv_rate: float = 0.039       # 3.9% Search-to-Order Conversion Rate
    variant_abandonment_rate: float = 0.085  # 8.5% zero-result / bounce rate

    def calculate_annual_impact(
        self,
        monthly_search_sessions: int = 250000
    ) -> Dict[str, Any]:
        """Computes monthly and annualized baseline vs variant commercial performance."""
        # Monthly orders
        monthly_baseline_orders = int(monthly_search_sessions * self.baseline_conv_rate)
        monthly_variant_orders = int(monthly_search_sessions * self.variant_conv_rate)

        # Monthly GMV
        monthly_baseline_gmv = monthly_baseline_orders * self.baseline_aov
        monthly_variant_gmv = monthly_variant_orders * self.variant_aov

        # Annualized GMV
        annual_baseline_gmv = monthly_baseline_gmv * 12.0
        annual_variant_gmv = monthly_variant_gmv * 12.0
        annual_gmv_uplift = annual_variant_gmv - annual_baseline_gmv
        annual_gmv_lift_pct = ((annual_variant_gmv - annual_baseline_gmv) / annual_baseline_gmv) * 100.0

        # AOV lift
        aov_absolute_lift = self.variant_aov - self.baseline_aov
        aov_lift_pct = (aov_absolute_lift / self.baseline_aov) * 100.0

        # Attach rate lift
        attach_rate_lift = self.variant_attach_rate - self.baseline_attach_rate
        attach_rate_lift_pct = (attach_rate_lift / self.baseline_attach_rate) * 100.0

        # Additional items sold annually
        annual_baseline_units = int(monthly_baseline_orders * 12 * self.baseline_attach_rate)
        annual_variant_units = int(monthly_variant_orders * 12 * self.variant_attach_rate)
        annual_additional_units = annual_variant_units - annual_baseline_units

        # Abandonment reduction
        recovered_monthly_sessions = int(monthly_search_sessions * (self.baseline_abandonment_rate - self.variant_abandonment_rate))

        return {
            "monthly_search_sessions": monthly_search_sessions,
            "monthly_baseline_orders": monthly_baseline_orders,
            "monthly_variant_orders": monthly_variant_orders,
            "monthly_baseline_gmv": round(monthly_baseline_gmv, 2),
            "monthly_variant_gmv": round(monthly_variant_gmv, 2),
            "annual_baseline_gmv": round(annual_baseline_gmv, 2),
            "annual_variant_gmv": round(annual_variant_gmv, 2),
            "annual_gmv_uplift": round(annual_gmv_uplift, 2),
            "annual_gmv_lift_pct": round(annual_gmv_lift_pct, 2),
            "baseline_aov": self.baseline_aov,
            "variant_aov": self.variant_aov,
            "aov_absolute_lift": round(aov_absolute_lift, 2),
            "aov_lift_pct": round(aov_lift_pct, 2),
            "baseline_ctr": self.baseline_ctr,
            "variant_ctr": self.variant_ctr,
            "ctr_lift_pct": round(((self.variant_ctr - self.baseline_ctr) / self.baseline_ctr) * 100.0, 2),
            "baseline_attach_rate": self.baseline_attach_rate,
            "variant_attach_rate": self.variant_attach_rate,
            "attach_rate_lift_pct": round(attach_rate_lift_pct, 2),
            "annual_additional_units": annual_additional_units,
            "recovered_monthly_sessions": recovered_monthly_sessions,
            "abandonment_drop_pct_points": round((self.baseline_abandonment_rate - self.variant_abandonment_rate) * 100.0, 2)
        }


class ABExperimentSimulator:
    """Simulates 30-day randomized A/B storefront experiment and computes statistical significance."""

    def __init__(self, impact_model: Optional[CommercialImpactModel] = None):
        self.impact_model = impact_model or CommercialImpactModel()

    def simulate_30_day_experiment(
        self,
        daily_traffic: int = 10000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Simulates 30 days of 50/50 A/B test data with statistical hypothesis testing."""
        np.random.seed(seed)
        days = 30
        control_traffic = daily_traffic // 2
        variant_traffic = daily_traffic // 2

        daily_records = []
        all_control_order_values = []
        all_variant_order_values = []

        for d in range(1, days + 1):
            # Add day-of-week seasonality (weekends +15% traffic and intent)
            dow_factor = 1.15 if (d % 7 in [5, 6]) else 1.0

            # Control conversions
            c_conv_p = np.clip(np.random.normal(self.impact_model.baseline_conv_rate * dow_factor, 0.002), 0.015, 0.040)
            c_orders = int(np.random.binomial(control_traffic, c_conv_p))

            # Variant conversions
            v_conv_p = np.clip(np.random.normal(self.impact_model.variant_conv_rate * dow_factor, 0.0025), 0.025, 0.055)
            v_orders = int(np.random.binomial(variant_traffic, v_conv_p))

            # Order values (simulated order basket values)
            c_vals = np.random.normal(self.impact_model.baseline_aov, 350, size=max(1, c_orders))
            c_vals = np.clip(c_vals, 499, 8999)
            all_control_order_values.extend(c_vals.tolist())

            v_vals = np.random.normal(self.impact_model.variant_aov, 420, size=max(1, v_orders))
            v_vals = np.clip(v_vals, 699, 12999)
            all_variant_order_values.extend(v_vals.tolist())

            c_aov = float(np.mean(c_vals))
            v_aov = float(np.mean(v_vals))

            # Bundle adoption in variant: ~38% with noise
            bundle_adoption = float(np.clip(np.random.normal(0.385, 0.02), 0.30, 0.48))

            c_gmv = float(c_orders * c_aov)
            v_gmv = float(v_orders * v_aov)

            daily_records.append({
                "day": d,
                "control_sessions": control_traffic,
                "variant_sessions": variant_traffic,
                "control_orders": c_orders,
                "variant_orders": v_orders,
                "control_conv_rate": round(c_orders / control_traffic, 4),
                "variant_conv_rate": round(v_orders / variant_traffic, 4),
                "control_aov": round(c_aov, 2),
                "variant_aov": round(v_aov, 2),
                "control_gmv": round(c_gmv, 2),
                "variant_gmv": round(v_gmv, 2),
                "variant_bundle_adoption_rate": round(bundle_adoption, 4)
            })

        daily_df = pd.DataFrame(daily_records)

        # ----------------- Statistical Tests -----------------
        # 1. Two-sample Student's t-test on AOV
        c_vals_arr = np.array(all_control_order_values)
        v_vals_arr = np.array(all_variant_order_values)
        t_stat, t_p_value = stats.ttest_ind(v_vals_arr, c_vals_arr, equal_var=False)

        mean_diff_aov = np.mean(v_vals_arr) - np.mean(c_vals_arr)
        se_aov = np.sqrt(np.var(v_vals_arr, ddof=1)/len(v_vals_arr) + np.var(c_vals_arr, ddof=1)/len(c_vals_arr))
        aov_ci_lower = mean_diff_aov - 1.96 * se_aov
        aov_ci_upper = mean_diff_aov + 1.96 * se_aov

        # 2. Two-proportion Z-test on Conversion Rate
        total_c_sessions = daily_df["control_sessions"].sum()
        total_v_sessions = daily_df["variant_sessions"].sum()
        total_c_orders = daily_df["control_orders"].sum()
        total_v_orders = daily_df["variant_orders"].sum()

        p1 = total_v_orders / total_v_sessions
        p2 = total_c_orders / total_c_sessions
        p_pooled = (total_v_orders + total_c_orders) / (total_v_sessions + total_c_sessions)
        se_p = np.sqrt(p_pooled * (1 - p_pooled) * (1/total_v_sessions + 1/total_c_sessions))
        z_stat = (p1 - p2) / se_p
        z_p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        se_diff = np.sqrt(p1*(1-p1)/total_v_sessions + p2*(1-p2)/total_c_sessions)
        conv_ci_lower = (p1 - p2) - 1.96 * se_diff
        conv_ci_upper = (p1 - p2) + 1.96 * se_diff

        return {
            "daily_df": daily_df,
            "overall_summary": {
                "total_control_sessions": int(total_c_sessions),
                "total_variant_sessions": int(total_v_sessions),
                "total_control_orders": int(total_c_orders),
                "total_variant_orders": int(total_v_orders),
                "overall_control_conv_rate": round(p2, 4),
                "overall_variant_conv_rate": round(p1, 4),
                "conv_rate_relative_lift_pct": round(((p1 - p2) / p2) * 100.0, 2),
                "overall_control_aov": round(float(np.mean(c_vals_arr)), 2),
                "overall_variant_aov": round(float(np.mean(v_vals_arr)), 2),
                "aov_relative_lift_pct": round(((np.mean(v_vals_arr) - np.mean(c_vals_arr)) / np.mean(c_vals_arr)) * 100.0, 2),
                "total_control_gmv": round(float(np.sum(c_vals_arr)), 2),
                "total_variant_gmv": round(float(np.sum(v_vals_arr)), 2),
                "total_gmv_lift_pct": round(((np.sum(v_vals_arr) - np.sum(c_vals_arr)) / np.sum(c_vals_arr)) * 100.0, 2),
                "avg_bundle_adoption_rate": round(float(daily_df["variant_bundle_adoption_rate"].mean()), 4)
            },
            "aov_hypothesis_test": {
                "test_name": "Two-Sample Welch's t-test",
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(t_p_value),
                "statistically_significant": bool(t_p_value < 0.05),
                "mean_difference": round(float(mean_diff_aov), 2),
                "ci_95_lower": round(float(aov_ci_lower), 2),
                "ci_95_upper": round(float(aov_ci_upper), 2)
            },
            "conversion_hypothesis_test": {
                "test_name": "Two-Proportion Z-Test",
                "z_statistic": round(float(z_stat), 4),
                "p_value": float(z_p_value),
                "statistically_significant": bool(z_p_value < 0.05),
                "rate_difference_pct_points": round(float((p1 - p2) * 100.0), 3),
                "ci_95_lower": round(float(conv_ci_lower * 100.0), 3),
                "ci_95_upper": round(float(conv_ci_upper * 100.0), 3)
            }
        }
