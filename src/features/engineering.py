"""
Feature Engineering Pipeline

Three feature families:
1. Velocity features   — transaction frequency/amount aggregates over rolling windows
2. Behavioral features — deviation from user's historical baseline
3. Network features    — merchant/device risk signals
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VELOCITY_WINDOWS = [
    ("1h", 3600),
    ("6h", 21600),
    ("24h", 86400),
    ("7d", 604800),
]

CATEGORICAL_COLS = ["merchant_category", "merchant_country", "user_home_country"]
NUMERIC_COLS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_online",
    "is_cross_border",
    "is_high_risk_country",
    "account_age_days",
    "credit_limit",
]


# ---------------------------------------------------------------------------
# Velocity feature builder (operates on sorted DataFrame)
# ---------------------------------------------------------------------------

def build_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, compute aggregates over rolling time windows
    using only past transactions (no look-ahead leakage).
    """
    df = df.sort_values("timestamp").copy()
    df["timestamp_unix"] = df["timestamp"].astype(np.int64) // 10 ** 9

    velocity_rows = []

    # Group by user for efficiency
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        ts_arr = group["timestamp_unix"].values
        amt_arr = group["amount"].values
        online_arr = group["is_online"].values.astype(int)
        merchant_arr = group["merchant_id"].values

        n = len(group)
        row_features = {k: np.zeros(n) for k in [
            "txn_count_1h", "txn_count_6h", "txn_count_24h", "txn_count_7d",
            "amt_sum_1h", "amt_sum_6h", "amt_sum_24h", "amt_sum_7d",
            "amt_mean_1h", "amt_mean_24h", "amt_mean_7d",
            "amt_max_24h", "amt_max_7d",
            "online_count_1h", "online_count_24h",
            "unique_merchants_24h", "unique_merchants_7d",
            "amt_std_7d",
        ]}

        for i in range(n):
            t = ts_arr[i]
            for label, window_sec in VELOCITY_WINDOWS:
                mask = (ts_arr[:i] >= t - window_sec) & (ts_arr[:i] < t)
                past_amts = amt_arr[:i][mask]
                past_online = online_arr[:i][mask]
                past_merchants = merchant_arr[:i][mask]
                cnt = mask.sum()

                if label == "1h":
                    row_features["txn_count_1h"][i] = cnt
                    row_features["amt_sum_1h"][i] = past_amts.sum() if cnt else 0
                    row_features["amt_mean_1h"][i] = past_amts.mean() if cnt else 0
                    row_features["online_count_1h"][i] = past_online.sum() if cnt else 0
                elif label == "6h":
                    row_features["txn_count_6h"][i] = cnt
                    row_features["amt_sum_6h"][i] = past_amts.sum() if cnt else 0
                elif label == "24h":
                    row_features["txn_count_24h"][i] = cnt
                    row_features["amt_sum_24h"][i] = past_amts.sum() if cnt else 0
                    row_features["amt_mean_24h"][i] = past_amts.mean() if cnt else 0
                    row_features["amt_max_24h"][i] = past_amts.max() if cnt else 0
                    row_features["online_count_24h"][i] = past_online.sum() if cnt else 0
                    row_features["unique_merchants_24h"][i] = len(set(past_merchants)) if cnt else 0
                elif label == "7d":
                    row_features["txn_count_7d"][i] = cnt
                    row_features["amt_sum_7d"][i] = past_amts.sum() if cnt else 0
                    row_features["amt_mean_7d"][i] = past_amts.mean() if cnt else 0
                    row_features["amt_max_7d"][i] = past_amts.max() if cnt else 0
                    row_features["amt_std_7d"][i] = past_amts.std() if cnt > 1 else 0
                    row_features["unique_merchants_7d"][i] = len(set(past_merchants)) if cnt else 0

        feat_df = pd.DataFrame(row_features, index=group.index)
        velocity_rows.append(feat_df)

    velocity_df = pd.concat(velocity_rows).sort_index()
    return pd.concat([df, velocity_df], axis=1)


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation of current transaction from user's historical mean."""
    df = df.copy()

    # Amount z-score vs user's 7d history
    df["amt_zscore_7d"] = np.where(
        df["amt_std_7d"] > 0,
        (df["amount"] - df["amt_mean_7d"]) / df["amt_std_7d"],
        0.0,
    )

    # Amount ratio vs 24h mean
    df["amt_ratio_24h"] = np.where(
        df["amt_mean_24h"] > 0,
        df["amount"] / df["amt_mean_24h"],
        df["amount"],
    )

    # Burst indicator: txn count in 1h / typical 24h rate
    df["burst_ratio_1h"] = np.where(
        df["txn_count_24h"] > 0,
        df["txn_count_1h"] / (df["txn_count_24h"] / 24 + 1e-6),
        df["txn_count_1h"],
    )

    # Credit utilization spike
    df["credit_utilization_24h"] = np.where(
        df["credit_limit"] > 0,
        df["amt_sum_24h"] / df["credit_limit"],
        0.0,
    )

    # New device flag (simplistic: device not seen in 7d)
    # In production this would query a device history store
    df["is_new_device"] = (df["txn_count_7d"] == 0).astype(int)

    # Time since last transaction (seconds) — using txn_count proxy
    df["time_pressure_flag"] = (df["txn_count_1h"] >= 5).astype(int)

    return df


def build_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merchant and device-level risk signals."""
    df = df.copy()

    # Merchant risk: fraction of this merchant's txns that were fraud
    # (leakage-safe only when computed on train set and joined; here we use a proxy)
    merchant_fraud_rate = (
        df.groupby("merchant_id")["is_fraud"].transform("mean")
        if "is_fraud" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["merchant_fraud_rate"] = merchant_fraud_rate

    # Merchant transaction count (popularity)
    df["merchant_txn_count"] = df.groupby("merchant_id")["transaction_id"].transform("count")

    # Device reuse count across users (mule / shared device signal)
    df["device_user_count"] = df.groupby("device_fingerprint")["user_id"].transform("nunique")
    df["device_is_shared"] = (df["device_user_count"] > 1).astype(int)

    # High-risk category flag
    HIGH_RISK_CATS = {"jewelry", "electronics", "atm", "transfer"}
    df["is_high_risk_category"] = df["merchant_category"].isin(HIGH_RISK_CATS).astype(int)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Input must have raw transaction columns."""
    print("Building velocity features...")
    df = build_velocity_features(df)
    print("Building behavioral features...")
    df = build_behavioral_features(df)
    print("Building network features...")
    df = build_network_features(df)
    return df


# ---------------------------------------------------------------------------
# Sklearn-compatible transformer for inference pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Raw
    "amount", "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "is_online", "is_cross_border", "is_high_risk_country", "account_age_days",
    # Velocity
    "txn_count_1h", "txn_count_6h", "txn_count_24h", "txn_count_7d",
    "amt_sum_1h", "amt_sum_6h", "amt_sum_24h", "amt_sum_7d",
    "amt_mean_1h", "amt_mean_24h", "amt_mean_7d",
    "amt_max_24h", "amt_max_7d", "amt_std_7d",
    "online_count_1h", "online_count_24h",
    "unique_merchants_24h", "unique_merchants_7d",
    # Behavioral
    "amt_zscore_7d", "amt_ratio_24h", "burst_ratio_1h",
    "credit_utilization_24h", "is_new_device", "time_pressure_flag",
    # Network
    "merchant_fraud_rate", "merchant_txn_count",
    "device_user_count", "device_is_shared",
    "is_high_risk_category",
    # Encoded categoricals (added after encoding)
    "merchant_category_enc", "merchant_country_enc",
]


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer that encodes categoricals and scales numerics.
    Fit on training data; apply to any new batch.
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # Label encode categoricals
        for col in ["merchant_category", "merchant_country"]:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna("unknown"))
            self.label_encoders[col] = le

        # Build encoded df for scaler fitting
        df = self._encode(df)

        # Scale only the numeric-ish columns we'll actually use
        available = [c for c in FEATURE_COLS if c in df.columns]
        self.scaler.fit(df[available].fillna(0))
        self._feature_cols = available
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        df = X.copy()
        df = self._encode(df)
        available = self._feature_cols
        out = df[available].fillna(0).values.astype(np.float32)
        return out

    def get_feature_names_out(self) -> list[str]:
        return self._feature_cols

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["merchant_category", "merchant_country"]:
            le = self.label_encoders.get(col)
            if le is None:
                df[f"{col}_enc"] = 0
            else:
                vals = df[col].astype(str).fillna("unknown")
                # Handle unseen labels
                known = set(le.classes_)
                vals = vals.apply(lambda v: v if v in known else le.classes_[0])
                df[f"{col}_enc"] = le.transform(vals)
        return df
