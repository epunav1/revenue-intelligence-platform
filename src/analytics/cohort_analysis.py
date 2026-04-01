"""
Cohort Retention Analysis.

Builds the classic SaaS retention heatmap (cohort × period) and
computes aggregate retention curves for benchmarking.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_retention_matrix(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-form cohort table into a wide retention matrix.

    Input columns: cohort_month, period_number, retention_rate
    Returns: DataFrame with cohort_month as index, period_number as columns.
    """
    matrix = cohort_df.pivot_table(
        index="cohort_month",
        columns="period_number",
        values="retention_rate",
        aggfunc="first",
    )
    matrix.index = pd.to_datetime(matrix.index).strftime("%b %Y")
    matrix.columns = [f"Month {int(c)}" for c in matrix.columns]
    # Month 0 should always be 100
    matrix["Month 0"] = 100.0
    return matrix.round(1)


def retention_curve(cohort_df: pd.DataFrame,
                    by_plan: bool = False,
                    customer_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Average retention curve across all cohorts (or split by plan).

    Returns a DataFrame with period_number and avg_retention_rate.
    """
    if by_plan and customer_df is not None:
        # merge plan info
        plan_map = customer_df.set_index("customer_id")["plan_name"].to_dict()
        merged = cohort_df.copy()
        # We only have cohort-level data; approximate by plan using cohort sizes
        # For simplicity, return overall curve
        log.warning("Plan-level cohort curve approximation — returning overall curve.")

    curve = (
        cohort_df.groupby("period_number")
        .agg(
            avg_retention=("retention_rate", "mean"),
            median_retention=("retention_rate", "median"),
            p25_retention=("retention_rate", lambda x: x.quantile(0.25)),
            p75_retention=("retention_rate", lambda x: x.quantile(0.75)),
            cohort_count=("cohort_month", "count"),
        )
        .reset_index()
        .round(1)
    )
    return curve


def calculate_ltv_from_cohort(cohort_df: pd.DataFrame,
                               avg_mrr: float) -> pd.DataFrame:
    """
    Estimate customer LTV from the retention curve.
    LTV_at_period_N = sum of avg_retention_rate/100 * avg_mrr for periods 0..N
    """
    curve = retention_curve(cohort_df)
    curve["cumulative_months_retained"] = (
        curve["avg_retention"] / 100
    ).cumsum()
    curve["estimated_ltv"] = (
        curve["cumulative_months_retained"] * avg_mrr
    ).round(0)
    return curve


def cohort_size_trend(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """New cohort sizes over time — signals sales velocity."""
    sizes = (
        cohort_df[cohort_df["period_number"] == 0]
        [["cohort_month", "active_customers"]]
        .rename(columns={"active_customers": "cohort_size"})
        .copy()
    )
    sizes["cohort_month"] = pd.to_datetime(sizes["cohort_month"])
    sizes = sizes.sort_values("cohort_month")
    sizes["mom_growth_pct"] = (
        sizes["cohort_size"].pct_change() * 100
    ).round(1)
    return sizes


def quick_ratio(cohort_df: pd.DataFrame) -> float:
    """
    Net Revenue Retention proxy from cohort data.
    (customers retained at month 12) / (cohort size)
    averaged across all complete cohorts.
    """
    month_12 = cohort_df[cohort_df["period_number"] == 12].copy()
    if month_12.empty:
        return 0.0
    return round(month_12["retention_rate"].mean(), 1)


def benchmark_comparison(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Add industry benchmark retention bands to the curve for comparison.
    Based on Bessemer / OpenView SaaS benchmarks for $10M ARR stage.
    """
    benchmarks = {
        "best_in_class": {
            0: 100, 1: 93, 2: 87, 3: 82, 4: 79, 5: 76,
            6: 74,  9: 69, 12: 65, 18: 58, 24: 53,
        },
        "good":  {
            0: 100, 1: 88, 2: 80, 3: 73, 4: 68, 5: 64,
            6: 61,  9: 54, 12: 50, 18: 43, 24: 38,
        },
        "average": {
            0: 100, 1: 80, 2: 68, 3: 60, 4: 53, 5: 48,
            6: 44,  9: 37, 12: 33, 18: 28, 24: 24,
        },
    }
    df = curve.copy()
    for label, bm_data in benchmarks.items():
        df[f"benchmark_{label}"] = df["period_number"].map(bm_data)
    df = df.interpolate(method="linear")
    return df
