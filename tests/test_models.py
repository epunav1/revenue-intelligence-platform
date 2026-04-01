"""Tests for the ensemble model and explainer."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import FraudEnsemble
from src.models.explainer import FraudExplainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_classification_data(n: int = 2000, n_features: int = 20, fraud_rate: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (rng.random(n) < fraud_rate).astype(int)
    # Make it linearly separable enough to train
    X[y == 1, :3] += 2.5
    return X, y


@pytest.fixture(scope="module")
def trained_ensemble():
    X, y = make_classification_data()
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    model = FraudEnsemble(n_estimators=50)
    model.fit(X_tr, y_tr, feature_names=[f"feat_{i}" for i in range(X.shape[1])])
    return model, X, y, X_val, y_val


# ---------------------------------------------------------------------------
# Ensemble model tests
# ---------------------------------------------------------------------------

class TestFraudEnsemble:
    def test_fit_produces_models(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        assert model.xgb_model_ is not None
        assert model.lgb_model_ is not None

    def test_predict_proba_shape(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)

    def test_probabilities_sum_to_one(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        proba = model.predict_proba(X[:50])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_in_range(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        preds = model.predict(X[:20])
        assert set(preds).issubset({0, 1})

    def test_roc_auc_above_baseline(self, trained_ensemble):
        from sklearn.metrics import roc_auc_score
        model, X, y, X_val, y_val = trained_ensemble
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        assert auc > 0.6, f"AUC {auc:.3f} is too low — model should beat random"

    def test_evaluate_returns_all_keys(self, trained_ensemble):
        model, X, y, X_val, y_val = trained_ensemble
        metrics = model.evaluate(X_val, y_val)
        for key in ["roc_auc", "avg_precision", "f1", "precision", "recall", "tp", "fp", "fn", "tn"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_custom_threshold(self, trained_ensemble):
        model, X, y, X_val, y_val = trained_ensemble
        preds_low = model.predict(X_val, threshold=0.1)
        preds_high = model.predict(X_val, threshold=0.9)
        # Lower threshold → more predicted positives
        assert preds_low.sum() >= preds_high.sum()

    def test_save_load_roundtrip(self, trained_ensemble, tmp_path):
        model, X, y, _, _ = trained_ensemble
        path = str(tmp_path / "test_model.joblib")
        model.save(path)
        loaded = FraudEnsemble.load(path)
        np.testing.assert_allclose(
            model.predict_proba(X[:5]),
            loaded.predict_proba(X[:5]),
            atol=1e-5,
        )

    def test_feature_names_stored(self, trained_ensemble):
        model, _, _, _, _ = trained_ensemble
        assert len(model.feature_names_) == 20
        assert model.feature_names_[0] == "feat_0"

    def test_classes_attribute(self, trained_ensemble):
        model, _, _, _, _ = trained_ensemble
        np.testing.assert_array_equal(model.classes_, [0, 1])

    def test_xgb_weight_blend(self):
        """Changing xgb_weight shifts the ensemble score."""
        X, y = make_classification_data(n=500, seed=99)
        m1 = FraudEnsemble(n_estimators=30, xgb_weight=0.1)
        m2 = FraudEnsemble(n_estimators=30, xgb_weight=0.9)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict_proba(X[:10])[:, 1]
        p2 = m2.predict_proba(X[:10])[:, 1]
        assert not np.allclose(p1, p2), "Different weights should produce different scores"


# ---------------------------------------------------------------------------
# Explainer tests
# ---------------------------------------------------------------------------

class TestFraudExplainer:
    @pytest.fixture(scope="class")
    def explainer_setup(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        proba = model.predict_proba(X[:5])[:, 1]
        txn_ids = [f"txn_{i}" for i in range(5)]
        results = explainer.explain(X[:5], txn_ids, proba.tolist())
        return explainer, results, X, proba

    def test_explain_returns_correct_count(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        proba = model.predict_proba(X[:3])[:, 1]
        results = explainer.explain(X[:3], ["a", "b", "c"], proba.tolist())
        assert len(results) == 3

    def test_top_factors_count(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        proba = model.predict_proba(X[:1])[:, 1]
        results = explainer.explain(X[:1], ["txn_0"], proba.tolist(), top_k=5)
        assert len(results[0].top_factors) == 5

    def test_shap_directions_correct(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        proba = model.predict_proba(X[:5])[:, 1]
        results = explainer.explain(X[:5], [f"t{i}" for i in range(5)], proba.tolist())
        for r in results:
            for f in r.top_factors:
                expected = "increases_risk" if f.shap_value > 0 else "decreases_risk"
                assert f.direction == expected

    def test_explain_single(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        score = float(model.predict_proba(X[:1])[:, 1][0])
        result = explainer.explain_single(X[0], "txn_single", score)
        assert result.transaction_id == "txn_single"
        assert result.fraud_score == score

    def test_to_dict(self, trained_ensemble):
        model, X, y, _, _ = trained_ensemble
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        explainer = FraudExplainer(model, feature_names)
        explainer.build()
        score = float(model.predict_proba(X[:1])[:, 1][0])
        result = explainer.explain_single(X[0], "t1", score)
        d = result.to_dict()
        assert "transaction_id" in d
        assert "top_factors" in d
        assert "fraud_score" in d
