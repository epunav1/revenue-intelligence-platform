"""
FastAPI Fraud Detection Service

Endpoints:
  POST /predict          — Score a single transaction
  POST /predict/batch    — Score a batch (up to 1000)
  GET  /health           — Health check
  GET  /metrics          — Prometheus metrics (if enabled)
  GET  /model/info       — Model metadata
  GET  /drift/status     — Latest drift check summary
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.features.engineering import FraudFeatureTransformer
from src.models.ensemble import FraudEnsemble
from src.models.explainer import FraudExplainer
from src.monitoring.drift import (
    DriftMonitor,
    FRAUD_SCORE_HIST,
    FRAUD_ALERT_COUNTER,
    MODEL_LATENCY,
    TRANSACTIONS_PROCESSED,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "./models_store/ensemble_model.joblib")
PIPELINE_PATH = os.getenv("FEATURE_PIPELINE_PATH", "./models_store/feature_pipeline.joblib")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.8"))
DRIFT_REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "./models_store/reference_data.parquet")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model artefacts...")
    try:
        app_state["model"] = FraudEnsemble.load(MODEL_PATH)
        app_state["transformer"] = joblib.load(PIPELINE_PATH)
        app_state["feature_names"] = app_state["transformer"].get_feature_names_out()
        app_state["explainer"] = FraudExplainer(
            app_state["model"], app_state["feature_names"]
        )
        app_state["explainer"].build()
        app_state["drift_monitor"] = DriftMonitor(reference_path=DRIFT_REFERENCE_PATH)
        logger.info("All artefacts loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load model artefacts: {e}. Running in demo mode.")
        app_state["model"] = None
        app_state["transformer"] = None

    app_state["request_log"] = []  # rolling window for drift
    app_state["start_time"] = datetime.utcnow()
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Detection Engine",
    description="Real-time transaction fraud scoring with SHAP explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    merchant_id: str
    merchant_category: str
    merchant_country: str
    amount: float = Field(gt=0, le=1_000_000)
    timestamp: Optional[str] = None
    is_online: bool = False
    device_fingerprint: Optional[str] = None
    user_home_country: str = "US"
    credit_limit: float = Field(default=10_000.0, gt=0)
    account_age_days: int = Field(default=365, ge=0)
    # Optional pre-computed velocity features (for low-latency path)
    txn_count_1h: float = 0
    txn_count_6h: float = 0
    txn_count_24h: float = 0
    txn_count_7d: float = 0
    amt_sum_1h: float = 0
    amt_sum_6h: float = 0
    amt_sum_24h: float = 0
    amt_sum_7d: float = 0
    amt_mean_1h: float = 0
    amt_mean_24h: float = 0
    amt_mean_7d: float = 0
    amt_max_24h: float = 0
    amt_max_7d: float = 0
    amt_std_7d: float = 0
    online_count_1h: float = 0
    online_count_24h: float = 0
    unique_merchants_24h: float = 0
    unique_merchants_7d: float = 0

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if v is None:
            return datetime.utcnow().isoformat()
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": "user_12345",
            "merchant_id": "merch_99887",
            "merchant_category": "electronics",
            "merchant_country": "NG",
            "amount": 1299.99,
            "is_online": True,
            "device_fingerprint": "abc123xyz456",
            "user_home_country": "US",
            "credit_limit": 5000.0,
            "account_age_days": 180,
            "txn_count_1h": 8,
            "amt_sum_1h": 4500.0,
        }
    }}


class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    is_fraud: bool
    decision: str    # APPROVE | REVIEW | DECLINE
    threshold_used: float
    explanation: Optional[dict] = None
    inference_ms: float
    timestamp: str


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(max_length=1000)


class BatchResponse(BaseModel):
    results: list[FraudPrediction]
    total: int
    flagged: int
    processing_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score < 0.3:
        return "LOW"
    elif score < 0.5:
        return "MEDIUM"
    elif score < 0.8:
        return "HIGH"
    return "CRITICAL"


def _decision(score: float) -> str:
    if score < FRAUD_THRESHOLD:
        return "APPROVE"
    elif score < HIGH_RISK_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def _build_feature_df(req: TransactionRequest) -> pd.DataFrame:
    ts = pd.Timestamp(req.timestamp) if req.timestamp else pd.Timestamp.utcnow()
    row = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "merchant_id": req.merchant_id,
        "merchant_category": req.merchant_category,
        "merchant_country": req.merchant_country,
        "amount": req.amount,
        "timestamp": ts,
        "is_online": int(req.is_online),
        "device_fingerprint": req.device_fingerprint or "unknown",
        "user_home_country": req.user_home_country,
        "credit_limit": req.credit_limit,
        "account_age_days": req.account_age_days,
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_night": int(ts.hour >= 22 or ts.hour <= 5),
        "is_cross_border": int(req.merchant_country != req.user_home_country),
        "is_high_risk_country": int(req.merchant_country in {"NG", "RU", "OTHER"}),
        # Velocity (caller-provided or zero)
        "txn_count_1h": req.txn_count_1h,
        "txn_count_6h": req.txn_count_6h,
        "txn_count_24h": req.txn_count_24h,
        "txn_count_7d": req.txn_count_7d,
        "amt_sum_1h": req.amt_sum_1h,
        "amt_sum_6h": req.amt_sum_6h,
        "amt_sum_24h": req.amt_sum_24h,
        "amt_sum_7d": req.amt_sum_7d,
        "amt_mean_1h": req.amt_mean_1h,
        "amt_mean_24h": req.amt_mean_24h,
        "amt_mean_7d": req.amt_mean_7d,
        "amt_max_24h": req.amt_max_24h,
        "amt_max_7d": req.amt_max_7d,
        "amt_std_7d": req.amt_std_7d,
        "online_count_1h": req.online_count_1h,
        "online_count_24h": req.online_count_24h,
        "unique_merchants_24h": req.unique_merchants_24h,
        "unique_merchants_7d": req.unique_merchants_7d,
        # Derived behavioral
        "is_fraud": 0,  # placeholder for transformer
    }
    # Add derived features needed by transformer
    row["amt_zscore_7d"] = ((req.amount - row["amt_mean_7d"]) / row["amt_std_7d"]
                            if row["amt_std_7d"] > 0 else 0.0)
    row["amt_ratio_24h"] = (req.amount / row["amt_mean_24h"]
                            if row["amt_mean_24h"] > 0 else req.amount)
    row["burst_ratio_1h"] = (req.txn_count_1h / (req.txn_count_24h / 24 + 1e-6)
                              if req.txn_count_24h > 0 else req.txn_count_1h)
    row["credit_utilization_24h"] = (req.amt_sum_24h / req.credit_limit
                                     if req.credit_limit > 0 else 0.0)
    row["is_new_device"] = int(req.txn_count_7d == 0)
    row["time_pressure_flag"] = int(req.txn_count_1h >= 5)
    row["merchant_fraud_rate"] = 0.0  # unknown at inference time
    row["merchant_txn_count"] = 1
    row["device_user_count"] = 1
    row["device_is_shared"] = 0
    row["is_high_risk_category"] = int(req.merchant_category in
                                       {"jewelry", "electronics", "atm", "transfer"})
    return pd.DataFrame([row])


def _score_transaction(req: TransactionRequest, explain: bool = False) -> FraudPrediction:
    t0 = time.perf_counter()
    model = app_state.get("model")
    transformer = app_state.get("transformer")

    if model is None or transformer is None:
        # Demo mode: return mock score
        score = float(np.clip(np.random.beta(1, 10) + (0.6 if req.amount > 900 else 0), 0, 1))
    else:
        df = _build_feature_df(req)
        X = transformer.transform(df)
        score = float(model.predict_proba(X)[0, 1])

    elapsed_ms = (time.perf_counter() - t0) * 1000
    FRAUD_SCORE_HIST.observe(score)
    TRANSACTIONS_PROCESSED.inc()
    MODEL_LATENCY.observe(elapsed_ms / 1000)

    risk = _risk_level(score)
    FRAUD_ALERT_COUNTER.labels(risk_level=risk).inc()

    explanation_payload = None
    if explain and model is not None:
        explainer = app_state.get("explainer")
        if explainer:
            df = _build_feature_df(req)
            X = transformer.transform(df)
            feature_names = app_state["feature_names"]
            result = explainer.explain_single(X[0], req.transaction_id, score)
            explanation_payload = {
                "base_value": result.base_value,
                "top_factors": [
                    {
                        "feature": c.feature,
                        "value": c.value,
                        "shap_value": c.shap_value,
                        "direction": c.direction,
                    }
                    for c in result.top_factors
                ],
            }

    # Store in rolling log
    log_entry = {
        "transaction_id": req.transaction_id,
        "fraud_score": score,
        "risk_level": risk,
        "timestamp": datetime.utcnow().isoformat(),
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        "merchant_country": req.merchant_country,
    }
    app_state["request_log"].append(log_entry)
    if len(app_state["request_log"]) > 10_000:
        app_state["request_log"] = app_state["request_log"][-5_000:]

    return FraudPrediction(
        transaction_id=req.transaction_id,
        fraud_score=round(score, 6),
        risk_level=risk,
        is_fraud=score >= FRAUD_THRESHOLD,
        decision=_decision(score),
        threshold_used=FRAUD_THRESHOLD,
        explanation=explanation_payload,
        inference_ms=round(elapsed_ms, 3),
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    model_loaded = app_state.get("model") is not None
    uptime = (datetime.utcnow() - app_state["start_time"]).total_seconds()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime, 1),
        "version": "1.0.0",
    }


@app.get("/model/info", tags=["ops"])
async def model_info():
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "FraudEnsemble (XGBoost + LightGBM)",
        "xgb_weight": model.xgb_weight,
        "feature_count": len(app_state.get("feature_names", [])),
        "fraud_threshold": FRAUD_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
    }


@app.post("/predict", response_model=FraudPrediction, tags=["scoring"])
async def predict(request: TransactionRequest, explain: bool = False):
    """
    Score a single transaction. Set `explain=true` for SHAP breakdown.
    """
    try:
        result = _score_transaction(request, explain=explain)
        logger.info("prediction", **{
            "txn_id": result.transaction_id,
            "score": result.fraud_score,
            "decision": result.decision,
        })
        return result
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
async def predict_batch(batch: BatchRequest):
    """Score up to 1000 transactions in one request."""
    t0 = time.perf_counter()
    results = [_score_transaction(req) for req in batch.transactions]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    flagged = sum(1 for r in results if r.is_fraud)
    return BatchResponse(
        results=results,
        total=len(results),
        flagged=flagged,
        processing_ms=round(elapsed_ms, 3),
    )


@app.get("/drift/status", tags=["monitoring"])
async def drift_status():
    monitor = app_state.get("drift_monitor")
    if monitor is None:
        return {"status": "monitor_not_initialized"}

    history = monitor.get_drift_history()
    latest = history[-1] if history else None

    # Score distribution summary from rolling log
    recent_scores = [r["fraud_score"] for r in app_state.get("request_log", [])]

    return {
        "drift_check_count": len(history),
        "latest_check": latest,
        "score_stats": {
            "count": len(recent_scores),
            "mean": round(float(np.mean(recent_scores)), 4) if recent_scores else None,
            "p95": round(float(np.percentile(recent_scores, 95)), 4) if recent_scores else None,
            "fraud_rate": round(
                sum(1 for s in recent_scores if s >= FRAUD_THRESHOLD) / max(1, len(recent_scores)), 4
            ),
        },
    }


@app.get("/transactions/recent", tags=["monitoring"])
async def recent_transactions(limit: int = 50):
    log = app_state.get("request_log", [])
    return {"transactions": log[-limit:], "total_logged": len(log)}
