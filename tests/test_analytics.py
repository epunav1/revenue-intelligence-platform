"""Tests for analytics modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import pandas as pd

from src.data_generation.synthetic_data import (
    generate_customers, generate_subscriptions,
    generate_transactions, generate_product_events,
)
from src.analytics.rfm_analysis import compute_rfm, segment_summary, top_opportunities
from src.analytics.churn_prediction import ChurnPredictor, churn_risk_summary
from src.analytics.revenue_forecast import RevenueForecaster


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_customers():
    return generate_customers(n=80)

@pytest.fixture(scope="module")
def small_subs(small_customers):
    return generate_subscriptions(small_customers)

@pytest.fixture(scope="module")
def small_txns(small_customers, small_subs):
    return generate_transactions(small_customers, small_subs)

@pytest.fixture(scope="module")
def small_events(small_customers):
    return generate_product_events(small_customers)

@pytest.fixture(scope="module")
def customer_metrics(small_customers, small_subs, small_txns, small_events):
    """Simulate int_customer_metrics by merging."""
    base = small_customers.copy()
    sub_agg = (
        small_subs.groupby("customer_id")
        .agg(plan_name=("plan", "last"), billing_cycle=("billing_cycle", "last"),
             mrr=("mrr", "sum"), is_active=("status", lambda s: (s=="active").any()))
        .reset_index()
    )
    tx_agg = (
        small_txns[small_txns["status"] == "success"]
        .groupby("customer_id")
        .agg(total_transactions=("transaction_id","count"),
             ltv=("amount","sum"),
             last_payment_date=("transaction_date","max"),
             failed_payments=("status", lambda s: (s=="failed").sum()))
        .reset_index()
    )
    # Compute active_months from event_date directly (event_month only exists post-staging)
    ev_copy = small_events.copy()
    ev_copy["_month"] = pd.to_datetime(ev_copy["event_date"]).dt.to_period("M")
    ev_agg = (
        ev_copy.groupby("customer_id")
        .agg(total_events_all=("event_id","count"),
             active_months=("_month","nunique"),
             last_event_date=("event_date","max"),
             total_events_90d=("event_id",lambda x: len(x)),
             active_days_90d=("event_date","nunique"),
             logins_90d=("event_type", lambda x: (x=="login").sum()),
             support_tickets_90d=("event_type", lambda x: (x=="support_ticket").sum()),
             engagement_score_90d=("event_id","count"),
             engagement_score_all=("event_id","count"))
        .reset_index()
    )
    df = base.merge(sub_agg, on="customer_id", how="left")
    df = df.merge(tx_agg,  on="customer_id", how="left")
    df = df.merge(ev_agg,  on="customer_id", how="left")

    df["days_as_customer"]      = (pd.Timestamp("2025-03-31") - pd.to_datetime(df["signup_date"])).dt.days
    df["months_as_customer"]    = df["days_as_customer"] // 30
    df["days_since_last_event"] = (pd.Timestamp("2025-03-31") - pd.to_datetime(df["last_event_date"].fillna("2024-01-01"))).dt.days
    df["days_since_last_payment"] = 30
    df["refunded_payments"]     = 0
    df["employee_count"]        = df["employee_count"].fillna(10)
    df["health_score"]          = df["health_score"].fillna(70)
    df["seats"]                 = df["seats"].fillna(1)
    df["ltv"]                   = df["ltv"].fillna(0)
    df["total_transactions"]    = df["total_transactions"].fillna(0)
    df["failed_payments"]       = df["failed_payments"].fillna(0)
    return df


# ── RFM Tests ─────────────────────────────────────────────────────────────────

class TestRFMAnalysis:
    def test_compute_rfm_returns_dataframe(self, customer_metrics):
        result = compute_rfm(customer_metrics)
        assert isinstance(result, pd.DataFrame)

    def test_rfm_columns_added(self, customer_metrics):
        result = compute_rfm(customer_metrics)
        for col in ["r_score", "f_score", "m_score", "rfm_score", "rfm_segment"]:
            assert col in result.columns, f"Missing: {col}"

    def test_scores_in_range(self, customer_metrics):
        result = compute_rfm(customer_metrics)
        assert result["r_score"].between(1, 5).all()
        assert result["f_score"].between(1, 5).all()
        assert result["m_score"].between(1, 5).all()

    def test_rfm_score_is_average(self, customer_metrics):
        result = compute_rfm(customer_metrics)
        expected = result[["r_score","f_score","m_score"]].mean(axis=1)
        pd.testing.assert_series_equal(
            result["rfm_score"].round(2), expected.round(2), check_names=False
        )

    def test_all_customers_segmented(self, customer_metrics):
        result = compute_rfm(customer_metrics)
        assert result["rfm_segment"].notna().all()

    def test_segment_summary_shape(self, customer_metrics):
        rfm = compute_rfm(customer_metrics)
        summary = segment_summary(rfm)
        assert len(summary) == rfm["rfm_segment"].nunique()
        assert "customer_count" in summary.columns
        assert "total_mrr" in summary.columns

    def test_top_opportunities_returns_at_risk(self, customer_metrics):
        rfm = compute_rfm(customer_metrics)
        opps = top_opportunities(rfm, n=10)
        at_risk_segs = {"At Risk", "Can't Lose Them", "Needs Attention", "Hibernating"}
        if len(opps) > 0:
            assert set(opps["rfm_segment"].unique()).issubset(at_risk_segs)


# ── Churn Prediction Tests ────────────────────────────────────────────────────

class TestChurnPrediction:
    def test_train_returns_metrics(self, customer_metrics):
        predictor = ChurnPredictor()
        metrics = predictor.train(customer_metrics)
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert 0.5 <= metrics["roc_auc"] <= 1.0

    def test_predict_adds_columns(self, customer_metrics):
        predictor = ChurnPredictor()
        predictor.train(customer_metrics)
        scored = predictor.predict(customer_metrics)
        assert "churn_probability" in scored.columns
        assert "risk_tier" in scored.columns
        assert "risk_rank" in scored.columns

    def test_probabilities_in_range(self, customer_metrics):
        predictor = ChurnPredictor()
        predictor.train(customer_metrics)
        scored = predictor.predict(customer_metrics)
        assert scored["churn_probability"].between(0, 1).all()

    def test_risk_tiers_valid(self, customer_metrics):
        predictor = ChurnPredictor()
        predictor.train(customer_metrics)
        scored = predictor.predict(customer_metrics)
        valid_tiers = {"Low", "Medium", "High"}
        actual = set(scored["risk_tier"].astype(str).unique()) - {"nan", "None"}
        assert actual.issubset(valid_tiers)

    def test_churn_risk_summary(self, customer_metrics):
        predictor = ChurnPredictor()
        predictor.train(customer_metrics)
        scored = predictor.predict(customer_metrics)
        summary = churn_risk_summary(scored)
        assert "high_risk_count" in summary
        assert "mrr_at_risk" in summary
        assert summary["total_active"] > 0


# ── Revenue Forecast Tests ────────────────────────────────────────────────────

class TestRevenueForecaster:
    @pytest.fixture
    def mock_revenue(self):
        months = pd.date_range("2022-01-01", "2025-03-01", freq="MS")
        np.random.seed(42)
        base = 300_000
        trend = np.linspace(0, 800_000, len(months))
        noise = np.random.normal(0, 10_000, len(months))
        mrr = base + trend + noise
        return pd.DataFrame({"month": months, "total_mrr": mrr.clip(0)})

    def test_fit_succeeds(self, mock_revenue):
        fc = RevenueForecaster()
        result = fc.fit(mock_revenue)
        assert result.fitted

    def test_forecast_returns_dataframe(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        assert isinstance(daily, pd.DataFrame)

    def test_forecast_has_required_columns(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        for col in ["ds", "forecast", "lower_bound", "upper_bound"]:
            assert col in daily.columns

    def test_forecast_length(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        assert len(daily) == 90

    def test_confidence_band_valid(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        assert (daily["upper_bound"] >= daily["forecast"]).all()
        assert (daily["forecast"] >= daily["lower_bound"]).all()

    def test_horizon_summary_structure(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        summary = fc.horizon_summary(daily)
        for key in ["30d", "60d", "90d", "last_actual_mrr"]:
            assert key in summary
        for key in ["30d", "60d", "90d"]:
            assert "forecast" in summary[key]
            assert "pct_change" in summary[key]

    def test_forecasts_are_positive(self, mock_revenue):
        fc = RevenueForecaster()
        fc.fit(mock_revenue)
        daily = fc.forecast(horizon_days=90)
        assert (daily["forecast"] > 0).all()
        assert (daily["lower_bound"] >= 0).all()
