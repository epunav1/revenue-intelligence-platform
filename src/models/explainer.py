"""
SHAP Explainability Layer

Generates per-prediction explanations using TreeExplainer.
Returns top-K contributing features and a waterfall-ready payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import shap


@dataclass
class FeatureContribution:
    feature: str
    value: float        # actual feature value
    shap_value: float   # SHAP contribution (positive = pushes toward fraud)
    direction: str      # "increases_risk" | "decreases_risk"


@dataclass
class ExplanationResult:
    transaction_id: str
    fraud_score: float
    base_value: float
    top_factors: list[FeatureContribution]
    raw_shap_values: list[float]
    feature_names: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=float)


class FraudExplainer:
    """
    Wraps shap.TreeExplainer for both XGBoost and LightGBM components.
    Explanation is averaged across the ensemble weighted by xgb_weight.
    """

    def __init__(self, ensemble_model, feature_names: list[str]):
        self.ensemble = ensemble_model
        self.feature_names = feature_names
        self._xgb_explainer: Optional[shap.TreeExplainer] = None
        self._lgb_explainer: Optional[shap.TreeExplainer] = None
        self._base_value: float = 0.0

    def build(self):
        """Initialize SHAP explainers. Call once after model is loaded."""
        print("Building XGBoost SHAP explainer...")
        self._xgb_explainer = shap.TreeExplainer(
            self.ensemble.xgb_model_,
            feature_names=self.feature_names,
        )
        print("Building LightGBM SHAP explainer...")
        self._lgb_explainer = shap.TreeExplainer(
            self.ensemble.lgb_model_,
            feature_names=self.feature_names,
        )
        # Base value (log-odds of fraud in training data)
        xgb_base = float(np.mean(self._xgb_explainer.expected_value))
        lgb_base = float(np.mean(self._lgb_explainer.expected_value))
        self._base_value = (
            self.ensemble.xgb_weight * xgb_base
            + (1 - self.ensemble.xgb_weight) * lgb_base
        )
        print("SHAP explainers ready.")

    def explain(
        self,
        X: np.ndarray,
        transaction_ids: list[str],
        fraud_scores: list[float],
        top_k: int = 8,
    ) -> list[ExplanationResult]:
        if self._xgb_explainer is None:
            self.build()

        # SHAP values shape: (n_samples, n_features)
        xgb_shap = self._xgb_explainer.shap_values(X)
        lgb_shap = self._lgb_explainer.shap_values(X)

        # LGB returns list [class0, class1] for binary; grab class1
        if isinstance(lgb_shap, list):
            lgb_shap = lgb_shap[1]
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]

        blended = (
            self.ensemble.xgb_weight * xgb_shap
            + (1 - self.ensemble.xgb_weight) * lgb_shap
        )

        results = []
        for i, (txn_id, score) in enumerate(zip(transaction_ids, fraud_scores)):
            shap_row = blended[i]
            feature_row = X[i]

            # Sort by absolute SHAP value
            ranked_idx = np.argsort(np.abs(shap_row))[::-1][:top_k]

            contributions = []
            for idx in ranked_idx:
                fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
                sv = float(shap_row[idx])
                contributions.append(FeatureContribution(
                    feature=fname,
                    value=float(feature_row[idx]),
                    shap_value=sv,
                    direction="increases_risk" if sv > 0 else "decreases_risk",
                ))

            results.append(ExplanationResult(
                transaction_id=txn_id,
                fraud_score=float(score),
                base_value=self._base_value,
                top_factors=contributions,
                raw_shap_values=shap_row.tolist(),
                feature_names=self.feature_names,
            ))

        return results

    def explain_single(
        self,
        x: np.ndarray,
        transaction_id: str,
        fraud_score: float,
        top_k: int = 8,
    ) -> ExplanationResult:
        return self.explain(
            x.reshape(1, -1), [transaction_id], [fraud_score], top_k
        )[0]
