"""Tests for the FastAPI fraud detection endpoints."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Mock model + transformer so tests run without trained artefacts
# ---------------------------------------------------------------------------

def _make_mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1]])
    model.xgb_weight = 0.5
    model.feature_names_ = ["f1", "f2"]
    return model


def _make_mock_transformer():
    t = MagicMock()
    t.transform.return_value = np.zeros((1, 38), dtype=np.float32)
    t.get_feature_names_out.return_value = [f"feat_{i}" for i in range(38)]
    return t


@pytest.fixture
def app_with_mock():
    """Import app and inject mocked state before each test."""
    from src.api.main import app, app_state

    app_state["model"] = _make_mock_model()
    app_state["transformer"] = _make_mock_transformer()
    app_state["feature_names"] = [f"feat_{i}" for i in range(38)]
    app_state["explainer"] = None
    app_state["drift_monitor"] = None
    app_state["request_log"] = []

    from datetime import datetime
    app_state["start_time"] = datetime.utcnow()
    return app


SAMPLE_TXN = {
    "user_id": "user_001",
    "merchant_id": "merch_99",
    "merchant_category": "electronics",
    "merchant_country": "NG",
    "amount": 1299.99,
    "is_online": True,
    "device_fingerprint": "abc123",
    "user_home_country": "US",
    "credit_limit": 5000.0,
    "account_age_days": 120,
    "txn_count_1h": 6,
    "amt_sum_1h": 3000.0,
    "txn_count_24h": 12,
    "amt_sum_24h": 5000.0,
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("healthy", "degraded")


@pytest.mark.asyncio
async def test_health_model_loaded(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_info(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "fraud_threshold" in data


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_returns_200(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=SAMPLE_TXN)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_predict_response_schema(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=SAMPLE_TXN)
    data = r.json()
    required_keys = [
        "transaction_id", "fraud_score", "risk_level",
        "is_fraud", "decision", "threshold_used", "inference_ms", "timestamp",
    ]
    for k in required_keys:
        assert k in data, f"Missing key: {k}"


@pytest.mark.asyncio
async def test_predict_fraud_score_range(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=SAMPLE_TXN)
    score = r.json()["fraud_score"]
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_predict_valid_risk_levels(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=SAMPLE_TXN)
    assert r.json()["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@pytest.mark.asyncio
async def test_predict_valid_decisions(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=SAMPLE_TXN)
    assert r.json()["decision"] in ("APPROVE", "REVIEW", "DECLINE")


@pytest.mark.asyncio
async def test_predict_invalid_amount_rejected(app_with_mock):
    bad = {**SAMPLE_TXN, "amount": -100}
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_required_field(app_with_mock):
    bad = {k: v for k, v in SAMPLE_TXN.items() if k != "user_id"}
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict", json=bad)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_predict(app_with_mock):
    batch = {"transactions": [SAMPLE_TXN] * 5}
    mock_model = app_with_mock.state  # unused
    from src.api.main import app_state
    app_state["transformer"].transform.return_value = np.zeros((5, 38), dtype=np.float32)
    app_state["model"].predict_proba.return_value = np.tile([0.9, 0.1], (5, 1))

    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict/batch", json=batch)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5
    assert "results" in data
    assert len(data["results"]) == 5


@pytest.mark.asyncio
async def test_batch_flagged_count(app_with_mock):
    from src.api.main import app_state
    # Mock: all transactions score 0.9 (above threshold → fraud)
    app_state["model"].predict_proba.return_value = np.array([[0.1, 0.9]] * 3)
    app_state["transformer"].transform.return_value = np.zeros((3, 38), dtype=np.float32)

    batch = {"transactions": [SAMPLE_TXN] * 3}
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.post("/predict/batch", json=batch)
    assert r.json()["flagged"] == 3


# ---------------------------------------------------------------------------
# Drift status + recent transactions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drift_status(app_with_mock):
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.get("/drift/status")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_recent_transactions_log(app_with_mock):
    from src.api.main import app_state
    app_state["request_log"] = [
        {"transaction_id": f"t{i}", "fraud_score": 0.1 * i} for i in range(10)
    ]
    async with AsyncClient(transport=ASGITransport(app=app_with_mock), base_url="http://test") as client:
        r = await client.get("/transactions/recent?limit=5")
    data = r.json()
    assert len(data["transactions"]) == 5
    assert data["total_logged"] == 10
