"""
Model Drift Monitoring

Uses Evidently for:
- Data drift detection (feature distribution shift)
- Prediction drift (score distribution shift)
- Model performance degradation tracking (when labels are available)

Exposes a Prometheus metrics endpoint and generates HTML reports.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
)
from evidently.report import Report
from prometheus_client import Counter, Gauge, Histogram, start_http_server

REPORTS_DIR = Path("./logs/drift_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

FRAUD_SCORE_HIST = Histogram(
    "fraud_score",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FRAUD_ALERT_COUNTER = Counter(
    "fraud_alerts_total",
    "Total fraud alerts triggered",
    ["risk_level"],
)
DRIFT_GAUGE = Gauge(
    "feature_drift_detected",
    "1 if drift detected on latest check, 0 otherwise",
)
MODEL_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "End-to-end model inference latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
TRANSACTIONS_PROCESSED = Counter(
    "transactions_processed_total",
    "Total transactions scored",
)


def start_prometheus(port: int = 9090):
    """Start Prometheus metrics HTTP server."""
    try:
        start_http_server(port)
        print(f"Prometheus metrics available at http://localhost:{port}/metrics")
    except OSError:
        print(f"Prometheus server already running on port {port}")


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Compares a rolling window of production data against a reference dataset.

    Usage:
        monitor = DriftMonitor(reference_path="models_store/reference_data.parquet")
        monitor.load_reference()
        report = monitor.check_drift(current_df)
    """

    def __init__(
        self,
        reference_path: str = "./models_store/reference_data.parquet",
        feature_cols: Optional[list[str]] = None,
        drift_threshold: float = 0.1,
    ):
        self.reference_path = reference_path
        self.drift_threshold = drift_threshold
        self.reference_df: Optional[pd.DataFrame] = None
        self.feature_cols = feature_cols or [
            "amount", "hour_of_day", "is_online", "is_cross_border",
            "txn_count_1h", "txn_count_24h", "amt_sum_24h", "amt_zscore_7d",
            "burst_ratio_1h", "credit_utilization_24h",
            "merchant_fraud_rate", "device_is_shared",
        ]
        self._drift_history: list[dict] = []

    def load_reference(self):
        self.reference_df = pd.read_parquet(self.reference_path)
        print(f"Reference data loaded: {len(self.reference_df):,} rows")

    def check_drift(self, current_df: pd.DataFrame, save_report: bool = True) -> dict:
        if self.reference_df is None:
            self.load_reference()

        available_cols = [c for c in self.feature_cols if c in current_df.columns
                          and c in self.reference_df.columns]

        ref = self.reference_df[available_cols].copy()
        cur = current_df[available_cols].copy()

        column_mapping = ColumnMapping()

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset(),
        ])
        report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

        DRIFT_GAUGE.set(1 if drift_detected else 0)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "n_reference": len(ref),
            "n_current": len(cur),
            "columns_checked": len(available_cols),
        }
        self._drift_history.append(summary)

        if save_report:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            html_path = REPORTS_DIR / f"drift_report_{ts}.html"
            report.save_html(str(html_path))
            summary["report_path"] = str(html_path)
            print(f"Drift report saved: {html_path}")

        return summary

    def get_drift_history(self) -> list[dict]:
        return self._drift_history

    def check_score_drift(
        self,
        current_scores: list[float],
        reference_scores: Optional[list[float]] = None,
    ) -> dict:
        """KS test on score distributions."""
        from scipy.stats import ks_2samp

        if reference_scores is None:
            # Use uniform baseline as placeholder
            reference_scores = np.random.uniform(0, 0.3, size=len(current_scores)).tolist()

        stat, p_value = ks_2samp(reference_scores, current_scores)
        drift_detected = p_value < 0.05

        return {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def performance_report(
        self,
        y_true: list[int],
        y_score: list[float],
        threshold: float = 0.5,
    ) -> dict:
        """Compute performance metrics for a labeled production window."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_score, recall_score, f1_score,
        )
        y_pred = [1 if s >= threshold else 0 for s in y_score]
        return {
            "roc_auc": roc_auc_score(y_true, y_score),
            "avg_precision": average_precision_score(y_true, y_score),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "timestamp": datetime.utcnow().isoformat(),
        }
