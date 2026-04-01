"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import (
    build_velocity_features,
    build_behavioral_features,
    build_network_features,
    FraudFeatureTransformer,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_transactions(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    rows = []
    users = [f"user_{i}" for i in range(5)]
    merchants = [f"merch_{i}" for i in range(10)]
    categories = ["grocery", "electronics", "atm", "retail", "online_retail"]
    countries = ["US", "GB", "NG"]

    for i in range(n):
        ts = base_time + timedelta(hours=rng.integers(0, 720))
        rows.append({
            "transaction_id": f"txn_{i:04d}",
            "user_id": rng.choice(users),
            "merchant_id": rng.choice(merchants),
            "merchant_category": rng.choice(categories),
            "merchant_country": rng.choice(countries),
            "amount": float(rng.lognormal(3.5, 1.0)),
            "timestamp": ts,
            "is_online": bool(rng.random() > 0.5),
            "device_fingerprint": f"dev_{rng.integers(0, 20)}",
            "user_home_country": "US",
            "credit_limit": 5000.0,
            "account_age_days": int(rng.integers(30, 1000)),
            "hour_of_day": ts.hour,
            "day_of_week": ts.weekday(),
            "is_weekend": int(ts.weekday() >= 5),
            "is_night": int(ts.hour >= 22 or ts.hour <= 5),
            "is_cross_border": 0,
            "is_high_risk_country": 0,
            "is_fraud": int(rng.random() < 0.02),
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


@pytest.fixture
def sample_df():
    return make_transactions(100)


# ---------------------------------------------------------------------------
# Velocity feature tests
# ---------------------------------------------------------------------------

class TestVelocityFeatures:
    def test_output_columns_exist(self, sample_df):
        result = build_velocity_features(sample_df)
        expected = [
            "txn_count_1h", "txn_count_24h", "txn_count_7d",
            "amt_sum_1h", "amt_sum_24h",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_lookahead(self, sample_df):
        """First transaction for each user must have zero velocity counts."""
        result = build_velocity_features(sample_df)
        first_txns = result.groupby("user_id").first().reset_index()
        for _, row in first_txns.iterrows():
            assert row["txn_count_1h"] == 0, "First txn should have 0 count in 1h window"
            assert row["txn_count_7d"] == 0, "First txn should have 0 count in 7d window"

    def test_non_negative(self, sample_df):
        result = build_velocity_features(sample_df)
        for col in ["txn_count_1h", "amt_sum_24h", "txn_count_7d"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_count_monotone_with_window(self, sample_df):
        """7d count >= 24h count >= 1h count for every row."""
        result = build_velocity_features(sample_df)
        assert (result["txn_count_7d"] >= result["txn_count_24h"]).all()
        assert (result["txn_count_24h"] >= result["txn_count_1h"]).all()

    def test_row_count_preserved(self, sample_df):
        result = build_velocity_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Behavioral feature tests
# ---------------------------------------------------------------------------

class TestBehavioralFeatures:
    @pytest.fixture
    def df_with_velocity(self, sample_df):
        df = build_velocity_features(sample_df)
        # Patch missing columns that behavioral needs
        df["amt_std_7d"] = df.groupby("user_id")["amount"].transform("std").fillna(0)
        return df

    def test_zscore_zero_when_std_zero(self, df_with_velocity):
        df = df_with_velocity.copy()
        df["amt_std_7d"] = 0
        result = build_behavioral_features(df)
        assert (result["amt_zscore_7d"] == 0.0).all()

    def test_burst_ratio_flag(self, df_with_velocity):
        df = df_with_velocity.copy()
        df["txn_count_1h"] = 15
        df["txn_count_24h"] = 20
        result = build_behavioral_features(df)
        assert (result["time_pressure_flag"] == 1).all()

    def test_credit_utilization_bounded(self, df_with_velocity):
        result = build_behavioral_features(df_with_velocity)
        assert (result["credit_utilization_24h"] >= 0).all()

    def test_new_device_flag(self, df_with_velocity):
        df = df_with_velocity.copy()
        df["txn_count_7d"] = 0
        result = build_behavioral_features(df)
        assert (result["is_new_device"] == 1).all()


# ---------------------------------------------------------------------------
# Network feature tests
# ---------------------------------------------------------------------------

class TestNetworkFeatures:
    def test_device_shared_flag(self, sample_df):
        df = build_velocity_features(sample_df)
        df = build_behavioral_features(df)
        # Force a shared device
        df.loc[:5, "device_fingerprint"] = "shared_dev"
        df.loc[:5, "user_id"] = [f"u_{i}" for i in range(6)]
        result = build_network_features(df)
        shared_mask = result["device_fingerprint"] == "shared_dev"
        assert (result.loc[shared_mask, "device_is_shared"] == 1).all()

    def test_merchant_txn_count_positive(self, sample_df):
        df = build_velocity_features(sample_df)
        df = build_behavioral_features(df)
        result = build_network_features(df)
        assert (result["merchant_txn_count"] > 0).all()

    def test_high_risk_category(self, sample_df):
        df = build_velocity_features(sample_df)
        df = build_behavioral_features(df)
        df.loc[0, "merchant_category"] = "jewelry"
        result = build_network_features(df)
        assert result.loc[0, "is_high_risk_category"] == 1


# ---------------------------------------------------------------------------
# Transformer tests
# ---------------------------------------------------------------------------

class TestFraudFeatureTransformer:
    def _full_pipeline(self, df):
        df = build_velocity_features(df)
        df["amt_std_7d"] = df.groupby("user_id")["amount"].transform("std").fillna(0)
        df = build_behavioral_features(df)
        df = build_network_features(df)
        return df

    def test_fit_transform_shape(self, sample_df):
        df = self._full_pipeline(sample_df)
        t = FraudFeatureTransformer()
        t.fit(df)
        X = t.transform(df)
        assert X.ndim == 2
        assert X.shape[0] == len(df)
        assert X.shape[1] > 10

    def test_feature_names_match_transform(self, sample_df):
        df = self._full_pipeline(sample_df)
        t = FraudFeatureTransformer()
        t.fit(df)
        X = t.transform(df)
        names = t.get_feature_names_out()
        assert len(names) == X.shape[1]

    def test_unseen_categories_handled(self, sample_df):
        df = self._full_pipeline(sample_df)
        t = FraudFeatureTransformer()
        t.fit(df)
        df2 = df.copy()
        df2.loc[0, "merchant_category"] = "UNSEEN_CAT"
        X = t.transform(df2)
        assert not np.isnan(X).any()

    def test_no_nan_in_output(self, sample_df):
        df = self._full_pipeline(sample_df)
        t = FraudFeatureTransformer()
        t.fit(df)
        X = t.transform(df)
        assert not np.isnan(X).any(), "NaN values found in transformer output"

    def test_idempotent_transform(self, sample_df):
        df = self._full_pipeline(sample_df)
        t = FraudFeatureTransformer()
        t.fit(df)
        X1 = t.transform(df)
        X2 = t.transform(df)
        np.testing.assert_array_equal(X1, X2)
