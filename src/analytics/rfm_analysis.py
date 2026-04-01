"""
RFM (Recency, Frequency, Monetary) Customer Segmentation.

Produces quintile-scored RFM profiles and business-meaningful segments
aligned with how SaaS CS teams actually work with accounts.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Segment definitions keyed by (R_tier, F_tier, M_tier) pattern
# Using simplified 3-tier (High/Mid/Low) after quintile scoring
SEGMENT_RULES: list[tuple[str, str, str, str]] = [
    # (r_tier, f_tier, m_tier, segment_name)
    ("High",  "High",  "High",  "Champions"),
    ("High",  "High",  "Mid",   "Loyal Customers"),
    ("High",  "Mid",   "High",  "Loyal Customers"),
    ("High",  "Mid",   "Mid",   "Potential Loyalists"),
    ("High",  "Low",   "Low",   "New Customers"),
    ("High",  "Low",   "Mid",   "Promising"),
    ("Mid",   "High",  "High",  "At Risk"),
    ("Mid",   "High",  "Mid",   "Needs Attention"),
    ("Mid",   "Mid",   "High",  "Needs Attention"),
    ("Low",   "High",  "High",  "Can't Lose Them"),
    ("Low",   "Mid",   "High",  "Can't Lose Them"),
    ("Low",   "Low",   "High",  "Hibernating"),
    ("Low",   "Low",   "Mid",   "Hibernating"),
    ("Low",   "Low",   "Low",   "Lost"),
]

SEGMENT_COLORS = {
    "Champions":          "#10b981",
    "Loyal Customers":    "#34d399",
    "Potential Loyalists":"#6ee7b7",
    "New Customers":      "#00d4ff",
    "Promising":          "#38bdf8",
    "At Risk":            "#f59e0b",
    "Needs Attention":    "#fb923c",
    "Can't Lose Them":    "#ef4444",
    "Hibernating":        "#8b5cf6",
    "Lost":               "#6b7280",
}

SEGMENT_PRIORITY = {  # lower = higher priority for CS outreach
    "Can't Lose Them": 1,
    "At Risk":         2,
    "Champions":       3,
    "Loyal Customers": 4,
    "Needs Attention": 5,
    "Promising":       6,
    "Potential Loyalists": 7,
    "New Customers":   8,
    "Hibernating":     9,
    "Lost":            10,
}


def _tier(score: float) -> str:
    if score >= 3.5:
        return "High"
    if score >= 2.5:
        return "Mid"
    return "Low"


def compute_rfm(df: pd.DataFrame,
                recency_col: str  = "days_since_last_event",
                frequency_col: str = "total_transactions",
                monetary_col: str  = "ltv",
                n_quantiles: int   = 5) -> pd.DataFrame:
    """
    Score each customer on Recency, Frequency, and Monetary value.

    Parameters
    ----------
    df            : Customer-level DataFrame (typically mart_customer_360)
    recency_col   : Column with days since last activity (lower = more recent)
    frequency_col : Column with transaction/interaction count (higher = better)
    monetary_col  : Column with lifetime revenue (higher = better)
    n_quantiles   : Number of quantile buckets (default 5 → quintiles)

    Returns
    -------
    DataFrame with added RFM columns.
    """
    result = df.copy()

    # Fill nulls so scoring doesn't break
    result[recency_col]   = result[recency_col].fillna(999)
    result[frequency_col] = result[frequency_col].fillna(0)
    result[monetary_col]  = result[monetary_col].fillna(0)

    # Quintile labels 1-5
    labels = list(range(1, n_quantiles + 1))

    # Recency: lower days = better = higher score → reverse rank
    result["r_score"] = pd.qcut(
        result[recency_col].rank(method="first", ascending=False),
        q=n_quantiles, labels=labels, duplicates="drop"
    ).astype(int)

    result["f_score"] = pd.qcut(
        result[frequency_col].rank(method="first"),
        q=n_quantiles, labels=labels, duplicates="drop"
    ).astype(int)

    result["m_score"] = pd.qcut(
        result[monetary_col].rank(method="first"),
        q=n_quantiles, labels=labels, duplicates="drop"
    ).astype(int)

    result["rfm_score"] = result[["r_score", "f_score", "m_score"]].mean(axis=1).round(2)

    # Map to segment name using rule table
    result["r_tier"] = result["r_score"].apply(lambda s: _tier(s))
    result["f_tier"] = result["f_score"].apply(lambda s: _tier(s))
    result["m_tier"] = result["m_score"].apply(lambda s: _tier(s))

    def _segment(row) -> str:
        for r_t, f_t, m_t, seg in SEGMENT_RULES:
            if row["r_tier"] == r_t and row["f_tier"] == f_t and row["m_tier"] == m_t:
                return seg
        # fallback: use combined score
        if row["rfm_score"] >= 4.0:
            return "Champions"
        if row["rfm_score"] >= 3.0:
            return "Loyal Customers"
        if row["rfm_score"] >= 2.0:
            return "Needs Attention"
        return "Lost"

    result["rfm_segment"] = result.apply(_segment, axis=1)
    result["segment_color"]    = result["rfm_segment"].map(SEGMENT_COLORS)
    result["segment_priority"] = result["rfm_segment"].map(SEGMENT_PRIORITY)

    log.info(
        "RFM scored %d customers across %d segments",
        len(result),
        result["rfm_segment"].nunique(),
    )
    return result


def segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate RFM results into a segment-level summary table.
    """
    agg = (
        rfm_df.groupby("rfm_segment")
        .agg(
            customer_count=("customer_id", "count"),
            pct_of_total=("customer_id", lambda x: round(100 * len(x) / len(rfm_df), 1)),
            avg_ltv=("ltv", "mean"),
            total_mrr=("mrr", "sum"),
            avg_health_score=("health_score", "mean"),
            avg_rfm_score=("rfm_score", "mean"),
            churned_count=("is_churned", "sum"),
        )
        .reset_index()
    )

    agg["avg_ltv"]          = agg["avg_ltv"].round(0)
    agg["total_mrr"]        = agg["total_mrr"].round(0)
    agg["avg_health_score"] = agg["avg_health_score"].round(1)
    agg["avg_rfm_score"]    = agg["avg_rfm_score"].round(2)
    agg["churn_rate_pct"]   = (
        agg["churned_count"] / agg["customer_count"] * 100
    ).round(1)
    agg["color"]            = agg["rfm_segment"].map(SEGMENT_COLORS)
    agg["priority"]         = agg["rfm_segment"].map(SEGMENT_PRIORITY)

    return agg.sort_values("priority")


def top_opportunities(rfm_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return the top N customers for CS outreach: high MRR + high churn risk.
    """
    at_risk_segs = {"At Risk", "Can't Lose Them", "Needs Attention", "Hibernating"}
    return (
        rfm_df[rfm_df["rfm_segment"].isin(at_risk_segs)]
        .sort_values(["mrr", "rfm_score"], ascending=[False, True])
        .head(n)[["customer_id", "company_name", "plan_name", "mrr", "ltv",
                  "rfm_segment", "rfm_score", "health_score",
                  "days_since_last_event", "csm"]]
        .reset_index(drop=True)
    )
