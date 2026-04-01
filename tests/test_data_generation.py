"""Tests for synthetic data generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np

from src.data_generation.synthetic_data import (
    generate_customers,
    generate_subscriptions,
    generate_transactions,
    generate_product_events,
)
from src.config import PLANS, SIM_START, SIM_END


class TestGenerateCustomers:
    def setup_method(self):
        self.customers = generate_customers(n=100)

    def test_returns_dataframe(self):
        assert isinstance(self.customers, pd.DataFrame)

    def test_correct_row_count(self):
        assert len(self.customers) == 100

    def test_required_columns(self):
        required = [
            "customer_id", "company_name", "industry", "country",
            "plan", "billing_cycle", "seats", "signup_date",
            "is_churned", "health_score",
        ]
        for col in required:
            assert col in self.customers.columns, f"Missing column: {col}"

    def test_unique_customer_ids(self):
        assert self.customers["customer_id"].nunique() == len(self.customers)

    def test_valid_plans(self):
        assert set(self.customers["plan"].unique()).issubset(set(PLANS.keys()))

    def test_valid_billing_cycles(self):
        assert set(self.customers["billing_cycle"].unique()).issubset({"monthly", "annual"})

    def test_health_score_range(self):
        assert self.customers["health_score"].between(0, 100).all()

    def test_signup_dates_in_range(self):
        dates = pd.to_datetime(self.customers["signup_date"])
        assert (dates >= pd.Timestamp(SIM_START)).all()
        assert (dates <= pd.Timestamp(SIM_END)).all()

    def test_churn_flag_consistency(self):
        churned_mask = self.customers["is_churned"].astype(bool)
        # All churned customers should have a churned_at date
        assert self.customers.loc[churned_mask, "churned_at"].notna().all()
        # All active customers should have NULL churned_at
        assert self.customers.loc[~churned_mask, "churned_at"].isna().all()

    def test_some_customers_churned(self):
        # Statistical: with 100 customers over 3 years, expect >5 churned
        assert self.customers["is_churned"].sum() > 5

    def test_seats_positive(self):
        assert (self.customers["seats"] > 0).all()


class TestGenerateSubscriptions:
    def setup_method(self):
        self.customers = generate_customers(n=50)
        self.subs = generate_subscriptions(self.customers)

    def test_returns_dataframe(self):
        assert isinstance(self.subs, pd.DataFrame)

    def test_required_columns(self):
        required = ["subscription_id", "customer_id", "plan", "billing_cycle",
                    "mrr", "start_date", "status"]
        for col in required:
            assert col in self.subs.columns

    def test_all_customers_have_subscription(self):
        customer_ids = set(self.customers["customer_id"])
        sub_customer_ids = set(self.subs["customer_id"])
        assert customer_ids == sub_customer_ids

    def test_mrr_positive(self):
        assert (self.subs["mrr"] > 0).all()

    def test_valid_status(self):
        valid = {"active", "churned", "upgraded", "paused"}
        assert set(self.subs["status"].unique()).issubset(valid)

    def test_unique_subscription_ids(self):
        assert self.subs["subscription_id"].nunique() == len(self.subs)


class TestGenerateTransactions:
    def setup_method(self):
        self.customers = generate_customers(n=30)
        self.subs = generate_subscriptions(self.customers)
        self.txns = generate_transactions(self.customers, self.subs)

    def test_returns_dataframe(self):
        assert isinstance(self.txns, pd.DataFrame)

    def test_required_columns(self):
        required = ["transaction_id", "customer_id", "subscription_id",
                    "transaction_date", "amount", "currency", "transaction_type", "status"]
        for col in required:
            assert col in self.txns.columns

    def test_currency_is_usd(self):
        assert (self.txns["currency"] == "USD").all()

    def test_amount_non_negative(self):
        assert (self.txns["amount"] >= 0).all()

    def test_valid_status(self):
        assert set(self.txns["status"].unique()).issubset({"success", "failed", "refunded"})

    def test_many_transactions_generated(self):
        # 30 customers over 3 years: some annual, some monthly; mix of churn
        # conservatively expect at least 30 (at least 1 per customer)
        assert len(self.txns) >= 30


class TestGenerateProductEvents:
    def setup_method(self):
        self.customers = generate_customers(n=20)
        self.events = generate_product_events(self.customers)

    def test_returns_dataframe(self):
        assert isinstance(self.events, pd.DataFrame)

    def test_required_columns(self):
        for col in ["event_id", "customer_id", "event_type", "event_date"]:
            assert col in self.events.columns

    def test_only_known_customers(self):
        known = set(self.customers["customer_id"])
        assert set(self.events["customer_id"].unique()).issubset(known)

    def test_event_dates_in_range(self):
        dates = pd.to_datetime(self.events["event_date"])
        assert (dates >= pd.Timestamp(SIM_START)).all()
        assert (dates <= pd.Timestamp(SIM_END)).all()

    def test_no_null_event_ids(self):
        assert self.events["event_id"].notna().all()
