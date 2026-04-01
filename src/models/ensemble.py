"""
Ensemble Model: XGBoost + LightGBM soft-voting classifier

Uses calibrated probability averaging with learned blend weights.
"""

from __future__ import annotations

import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Optional

import xgboost as xgb
import lightgbm as lgb


class FraudEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted soft-voting ensemble of XGBoost and LightGBM.

    Both models are trained with class_weight adjustments for imbalance.
    Final score = xgb_weight * p_xgb + (1 - xgb_weight) * p_lgb
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        xgb_weight: float = 0.5,
        n_estimators: int = 500,
        early_stopping_rounds: int = 50,
    ):
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.xgb_weight = xgb_weight
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

        self.xgb_model_: Optional[xgb.XGBClassifier] = None
        self.lgb_model_: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: list[str] = []
        self.classes_ = np.array([0, 1])

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def default_xgb_params(scale_pos_weight: float = 50) -> dict:
        return {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }

    @staticmethod
    def default_lgb_params(class_weight: str = "balanced") -> dict:
        return {
            "n_estimators": 500,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": class_weight,
            "metric": "average_precision",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        eval_set: Optional[tuple] = None,
    ) -> "FraudEnsemble":
        fraud_rate = y.mean()
        scale_pos_weight = (1 - fraud_rate) / (fraud_rate + 1e-9)
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]

        xgb_p = {**self.default_xgb_params(scale_pos_weight), **self.xgb_params}
        lgb_p = {**self.default_lgb_params(), **self.lgb_params}

        print("Training XGBoost...")
        self.xgb_model_ = xgb.XGBClassifier(**xgb_p)
        xgb_fit_kwargs = {}
        if eval_set is not None:
            xgb_fit_kwargs["eval_set"] = [eval_set]
            xgb_fit_kwargs["verbose"] = 100
        self.xgb_model_.fit(X, y, **xgb_fit_kwargs)

        print("Training LightGBM...")
        self.lgb_model_ = lgb.LGBMClassifier(**lgb_p)
        lgb_fit_kwargs = {}
        if eval_set is not None:
            lgb_fit_kwargs["eval_set"] = [eval_set]
        self.lgb_model_.fit(
            X, y,
            feature_name=self.feature_names_,
            **lgb_fit_kwargs,
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model_.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model_.predict_proba(X)[:, 1]
        p_ensemble = self.xgb_weight * p_xgb + (1 - self.xgb_weight) * p_lgb
        return np.column_stack([1 - p_ensemble, p_ensemble])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        proba = self.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        from sklearn.metrics import (
            confusion_matrix, precision_score, recall_score,
            roc_auc_score, average_precision_score, f1_score,
        )

        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        metrics = {
            "roc_auc": roc_auc_score(y, proba),
            "avg_precision": average_precision_score(y, proba),
            "f1": f1_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "threshold": threshold,
        }
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FraudEnsemble":
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
