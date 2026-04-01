"""
Churn Prediction Model.

Trains a gradient-boosted classifier on customer behavioural features
and returns calibrated churn probabilities with SHAP-based explanations.

Model pipeline:
  1. Feature engineering from int_customer_metrics
  2. Class-balanced GradientBoostingClassifier (or XGBoost if available)
  3. Platt scaling calibration for reliable probabilities
  4. SHAP TreeExplainer for per-customer risk factors
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import MART_DIR, RANDOM_SEED

log = logging.getLogger(__name__)
MODEL_PATH = MART_DIR / "churn_model.pkl"

FEATURES = [
    "days_as_customer",
    "months_as_customer",
    "mrr",
    "seats",
    "employee_count",
    "health_score",
    "total_transactions",
    "failed_payments",
    "refunded_payments",
    "total_events_90d",
    "active_days_90d",
    "logins_90d",
    "support_tickets_90d",
    "engagement_score_90d",
    "engagement_score_all",
    "active_months",
    "days_since_last_event",
    "days_since_last_payment",
    "plan_encoded",
    "billing_encoded",
    "country_encoded",
    "industry_encoded",
]

RISK_THRESHOLDS = {
    "High":   0.55,
    "Medium": 0.30,
    "Low":    0.00,
}


class ChurnPredictor:
    """End-to-end churn prediction pipeline."""

    def __init__(self):
        self.model:           Optional[Pipeline]     = None
        self.label_encoders:  dict[str, LabelEncoder] = {}
        self.feature_names:   list[str]               = []
        self.explainer                                 = None
        self.metrics:         dict                    = {}

    # ── Feature Engineering ──────────────────────────────────────────────────

    def _encode_categoricals(self,
                              df: pd.DataFrame,
                              fit: bool = True) -> pd.DataFrame:
        out = df.copy()
        for col, target in [
            ("plan_name",  "plan_encoded"),
            ("billing_cycle", "billing_encoded"),
            ("country",    "country_encoded"),
            ("industry",   "industry_encoded"),
        ]:
            if col not in out.columns:
                out[target] = 0
                continue
            if fit:
                le = LabelEncoder()
                out[target] = le.fit_transform(out[col].fillna("Unknown"))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    out[target] = out[col].apply(
                        lambda x: le.transform([x])[0] if x in known else -1
                    )
                else:
                    out[target] = 0
        return out

    def _fill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        numeric_fills = {
            "health_score":           70.0,
            "total_events_90d":        0.0,
            "active_days_90d":         0.0,
            "logins_90d":              0.0,
            "support_tickets_90d":     0.0,
            "engagement_score_90d":    0.0,
            "engagement_score_all":    0.0,
            "active_months":           0.0,
            "days_since_last_event":   365.0,
            "days_since_last_payment": 60.0,
            "failed_payments":         0.0,
            "refunded_payments":       0.0,
            "mrr":                     299.0,
            "seats":                   1.0,
            "employee_count":          10.0,
        }
        for col, fill in numeric_fills.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill)
        return out

    def prepare_features(self,
                         df: pd.DataFrame,
                         fit: bool = True) -> pd.DataFrame:
        out = self._fill_features(df)
        out = self._encode_categoricals(out, fit=fit)
        available = [f for f in FEATURES if f in out.columns]
        self.feature_names = available
        return out[available]

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the churn model.  Uses XGBoost if available, falls back to
        sklearn's GradientBoostingClassifier.

        Parameters
        ----------
        df : int_customer_metrics or mart_customer_360 table

        Returns
        -------
        dict of evaluation metrics
        """
        log.info("Preparing churn training data …")
        X = self.prepare_features(df, fit=True)
        y = df["is_churned"].astype(int)

        log.info("  Samples: %d  |  Churned: %d (%.1f%%)",
                 len(y), y.sum(), 100 * y.mean())

        if XGBOOST_AVAILABLE:
            log.info("  Using XGBoost classifier")
            base = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                eval_metric="auc",
                random_state=RANDOM_SEED,
                use_label_encoder=False,
                verbosity=0,
            )
        else:
            log.info("  Using GradientBoostingClassifier")
            base = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )

        # Calibrate for reliable probabilities
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(calibrated, X, y, cv=cv, scoring="roc_auc")
        log.info("  CV ROC-AUC: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

        # Fit on full data
        calibrated.fit(X, y)
        self.model = calibrated

        # Evaluate on training set (for reporting only)
        y_prob = calibrated.predict_proba(X)[:, 1]
        self.metrics = {
            "roc_auc":      round(roc_auc_score(y, y_prob), 4),
            "avg_precision": round(average_precision_score(y, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std":  round(cv_scores.std(), 4),
            "n_train":      len(y),
            "n_churned":    int(y.sum()),
            "churn_rate":   round(float(y.mean()), 4),
        }

        # SHAP explainer — unwrap CalibratedClassifierCV to get base estimator
        if SHAP_AVAILABLE:
            try:
                # CalibratedClassifierCV stores per-fold classifiers;
                # use the first fold's underlying estimator for SHAP
                base_est = calibrated.calibrated_classifiers_[0].estimator
                self.explainer = shap.TreeExplainer(base_est)
                log.info("  SHAP explainer initialised")
            except Exception as e:
                log.warning("  SHAP init failed: %s", e)

        self.save()
        log.info("  Model trained and saved.  ROC-AUC: %.4f", self.metrics["roc_auc"])
        return self.metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all customers and return enriched DataFrame.

        Adds columns:
          churn_probability, risk_tier, risk_rank
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.prepare_features(df, fit=False)
        probs = self.model.predict_proba(X)[:, 1]

        result = df.copy()
        result["churn_probability"] = probs.round(4)

        result["risk_tier"] = pd.cut(
            result["churn_probability"],
            bins=[-0.001, RISK_THRESHOLDS["Medium"], RISK_THRESHOLDS["High"], 1.001],
            labels=["Low", "Medium", "High"],
        )

        result["risk_rank"] = result["churn_probability"].rank(
            ascending=False, method="first"
        ).astype(int)

        return result

    def get_top_risk_factors(self,
                              customer_row: pd.Series,
                              n_factors: int = 5) -> list[dict]:
        """
        Return top N risk factors for a single customer using SHAP values.
        Falls back to rule-based explanation if SHAP unavailable.
        """
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                X_row = self.prepare_features(
                    pd.DataFrame([customer_row]), fit=False
                )
                raw = self.explainer.shap_values(X_row)
                # TreeExplainer returns list[array] for binary; take class-1 values
                sv = raw[1][0] if isinstance(raw, list) else raw[0]
                vals = list(zip(self.feature_names, sv))
                vals.sort(key=lambda x: abs(x[1]), reverse=True)
                return [
                    {"feature": f, "impact": round(float(v), 4),
                     "direction": "increases" if v > 0 else "decreases"}
                    for f, v in vals[:n_factors]
                ]
            except Exception:
                pass

        # Rule-based fallback
        factors = []
        row = customer_row
        if row.get("days_since_last_event", 0) > 21:
            factors.append({"feature": "low_recent_engagement",
                            "impact": 0.4, "direction": "increases"})
        if row.get("failed_payments", 0) >= 2:
            factors.append({"feature": "payment_failures",
                            "impact": 0.35, "direction": "increases"})
        if row.get("support_tickets_90d", 0) >= 3:
            factors.append({"feature": "high_support_volume",
                            "impact": 0.25, "direction": "increases"})
        if row.get("health_score", 100) < 50:
            factors.append({"feature": "low_health_score",
                            "impact": 0.3, "direction": "increases"})
        if row.get("logins_90d", 0) < 3:
            factors.append({"feature": "minimal_logins",
                            "impact": 0.2, "direction": "increases"})
        return factors[:n_factors]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None):
        path = path or MODEL_PATH
        with open(path, "wb") as f:
            pickle.dump({"model": self.model,
                         "encoders": self.label_encoders,
                         "features": self.feature_names,
                         "metrics":  self.metrics}, f)
        log.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ChurnPredictor":
        path = path or MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        p = cls()
        p.model         = data["model"]
        p.label_encoders = data["encoders"]
        p.feature_names  = data["features"]
        p.metrics        = data["metrics"]
        log.info("Model loaded from %s  ROC-AUC: %.4f",
                 path, p.metrics.get("roc_auc", 0))
        return p


def churn_risk_summary(scored_df: pd.DataFrame) -> dict:
    """Aggregate churn risk metrics across all customers."""
    active = scored_df[~scored_df["is_churned"].astype(bool)]
    at_risk = active[active["risk_tier"] == "High"]
    return {
        "total_active":         len(active),
        "high_risk_count":      len(at_risk),
        "high_risk_pct":        round(100 * len(at_risk) / max(len(active), 1), 1),
        "mrr_at_risk":          round(at_risk["mrr"].sum(), 0),
        "pct_mrr_at_risk":      round(
            100 * at_risk["mrr"].sum() / max(active["mrr"].sum(), 1), 1
        ),
        "avg_churn_probability": round(active["churn_probability"].mean(), 3),
    }
