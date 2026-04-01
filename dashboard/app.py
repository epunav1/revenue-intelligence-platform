"""
Live Fraud Detection Dashboard — Streamlit

Panels:
  1. Real-time transaction feed with fraud alerts
  2. Fraud score distribution (rolling window)
  3. Risk-level breakdown donut
  4. Top fraud features (SHAP bar chart)
  5. Model drift status
  6. KPI cards: total txns, fraud rate, avg latency, alerts today
"""

from __future__ import annotations

import os
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_SEC", "3"))
MAX_LOG_ROWS = 500

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "retail", "online_retail",
    "travel", "entertainment", "healthcare", "utility", "atm",
    "jewelry", "electronics", "transfer",
]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "NG", "RU", "CN", "BR", "MX"]

RISK_COLORS = {
    "LOW":      "#2ecc71",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e67e22",
    "CRITICAL": "#e74c3c",
}

st.set_page_config(
    page_title="Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0e6ff;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta-pos { color: #2ecc71; font-size: 0.8rem; }
    .kpi-delta-neg { color: #e74c3c; font-size: 0.8rem; }
    .alert-badge-CRITICAL {
        background: #e74c3c22; border: 1px solid #e74c3c;
        color: #e74c3c; border-radius: 6px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 700;
    }
    .alert-badge-HIGH {
        background: #e67e2222; border: 1px solid #e67e22;
        color: #e67e22; border-radius: 6px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 700;
    }
    .alert-badge-MEDIUM {
        background: #f39c1222; border: 1px solid #f39c12;
        color: #f39c12; border-radius: 6px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 700;
    }
    .alert-badge-LOW {
        background: #2ecc7122; border: 1px solid #2ecc71;
        color: #2ecc71; border-radius: 6px; padding: 2px 8px;
        font-size: 0.75rem; font-weight: 700;
    }
    .stMetric { background: #1e2130; border-radius: 8px; padding: 10px; }
    div[data-testid="stMetricValue"] > div { color: #e0e6ff !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state():
    if "txn_log" not in st.session_state:
        st.session_state.txn_log = []
    if "score_history" not in st.session_state:
        st.session_state.score_history = []
    if "total_txns" not in st.session_state:
        st.session_state.total_txns = 0
    if "total_frauds" not in st.session_state:
        st.session_state.total_frauds = 0
    if "latencies" not in st.session_state:
        st.session_state.latencies = []
    if "alerts_today" not in st.session_state:
        st.session_state.alerts_today = 0
    if "last_shap" not in st.session_state:
        st.session_state.last_shap = None
    if "api_online" not in st.session_state:
        st.session_state.api_online = False


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _random_transaction() -> dict:
    """Generate a synthetic transaction payload for the live feed."""
    is_fraud_attempt = random.random() < 0.04  # 4% fraud injection
    if is_fraud_attempt:
        return {
            "user_id": f"user_{random.randint(1, 500):05d}",
            "merchant_id": f"merch_{random.randint(1, 100):04d}",
            "merchant_category": random.choice(["electronics", "jewelry", "atm", "transfer"]),
            "merchant_country": random.choice(["NG", "RU", "OTHER"]),
            "amount": round(random.uniform(800, 4999), 2),
            "is_online": True,
            "device_fingerprint": str(uuid.uuid4())[:8],
            "user_home_country": "US",
            "credit_limit": 5000.0,
            "account_age_days": random.randint(10, 180),
            "txn_count_1h": random.randint(5, 25),
            "amt_sum_1h": round(random.uniform(2000, 8000), 2),
            "txn_count_24h": random.randint(10, 40),
            "amt_sum_24h": round(random.uniform(3000, 12000), 2),
        }
    else:
        cat = random.choice(MERCHANT_CATEGORIES[:8])
        return {
            "user_id": f"user_{random.randint(1, 5000):05d}",
            "merchant_id": f"merch_{random.randint(1, 1000):04d}",
            "merchant_category": cat,
            "merchant_country": random.choices(COUNTRIES, weights=[70,5,4,3,2,2,1,1,2,2,2], k=1)[0],
            "amount": round(random.lognormvariate(3.5, 0.9), 2),
            "is_online": cat == "online_retail" or random.random() < 0.2,
            "device_fingerprint": f"dev_{random.randint(1, 2000):04d}",
            "user_home_country": "US",
            "credit_limit": random.choice([2000, 5000, 10000, 25000]),
            "account_age_days": random.randint(90, 3000),
            "txn_count_1h": random.randint(0, 3),
            "amt_sum_1h": round(random.uniform(0, 200), 2),
            "txn_count_24h": random.randint(0, 10),
            "amt_sum_24h": round(random.uniform(0, 800), 2),
        }


def _call_api(payload: dict) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            params={"explain": "true"},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _simulate_score(payload: dict) -> dict:
    """Fallback simulation when API is offline."""
    score_base = 0.02
    if payload.get("merchant_country") in {"NG", "RU", "OTHER"}:
        score_base += 0.35
    if payload.get("merchant_category") in {"electronics", "jewelry", "atm"}:
        score_base += 0.15
    if payload.get("txn_count_1h", 0) > 8:
        score_base += 0.30
    if payload.get("amount", 0) > 1000:
        score_base += 0.10
    score = float(np.clip(score_base + np.random.beta(1.5, 10) * 0.2, 0, 1))
    risk = "LOW" if score < 0.3 else "MEDIUM" if score < 0.5 else "HIGH" if score < 0.8 else "CRITICAL"
    decision = "APPROVE" if score < 0.5 else "REVIEW" if score < 0.8 else "DECLINE"

    # Simulate SHAP explanation
    features = [
        "txn_count_1h", "merchant_country_enc", "amount", "is_high_risk_country",
        "merchant_category_enc", "amt_sum_1h", "account_age_days", "is_online",
    ]
    shap_vals = np.random.normal(0, 0.08, len(features))
    shap_vals[0] += payload.get("txn_count_1h", 0) * 0.02
    shap_vals[2] += (payload.get("amount", 0) - 200) * 0.0001

    explanation = {
        "base_value": 0.015,
        "top_factors": [
            {
                "feature": f,
                "value": round(float(np.random.uniform(0, 5)), 3),
                "shap_value": round(float(sv), 4),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            }
            for f, sv in sorted(zip(features, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        ],
    }

    return {
        "transaction_id": payload.get("user_id", "") + "_" + str(uuid.uuid4())[:6],
        "fraud_score": round(score, 4),
        "risk_level": risk,
        "is_fraud": score >= 0.5,
        "decision": decision,
        "threshold_used": 0.5,
        "explanation": explanation,
        "inference_ms": round(random.uniform(2, 15), 2),
        "timestamp": datetime.utcnow().isoformat(),
        "_simulated": True,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8892b0", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)


def score_histogram(scores: list[float]) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=40,
        marker=dict(
            color=scores,
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1.0, "#e74c3c"]],
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
        ),
        opacity=0.85,
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c", annotation_text="threshold")
    fig.update_layout(
        title="Fraud Score Distribution",
        xaxis_title="Score",
        yaxis_title="Count",
        **CHART_LAYOUT,
    )
    return fig


def risk_donut(log: list[dict]) -> go.Figure:
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in log:
        counts[r.get("risk_level", "LOW")] += 1
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.65,
        marker=dict(colors=[RISK_COLORS[k] for k in counts]),
        textinfo="label+percent",
        textfont_size=12,
    ))
    fig.update_layout(
        title="Risk Level Breakdown",
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


def shap_bar(explanation: dict) -> go.Figure:
    factors = explanation.get("top_factors", [])
    if not factors:
        return go.Figure()
    names = [f["feature"].replace("_", " ") for f in factors]
    vals = [f["shap_value"] for f in factors]
    colors = [RISK_COLORS["CRITICAL"] if v > 0 else RISK_COLORS["LOW"] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#8892b0", line_width=1)
    fig.update_layout(
        title="SHAP Feature Contributions (latest flagged txn)",
        xaxis_title="SHAP Value",
        yaxis=dict(autorange="reversed"),
        **CHART_LAYOUT,
    )
    return fig


def score_timeline(log: list[dict]) -> go.Figure:
    df = pd.DataFrame(log[-200:])
    if df.empty or "timestamp" not in df.columns:
        return go.Figure()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["color"] = df["risk_level"].map(RISK_COLORS).fillna("#8892b0")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["fraud_score"],
        mode="markers",
        marker=dict(
            color=df["color"],
            size=6,
            opacity=0.8,
            line=dict(width=0),
        ),
        text=df.apply(lambda r: f"${r.get('amount',0):.2f} | {r.get('merchant_category','')} | {r.get('risk_level','')}", axis=1),
        hovertemplate="%{text}<br>Score: %{y:.3f}<br>%{x}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", opacity=0.5)
    fig.update_layout(
        title="Live Score Timeline",
        yaxis=dict(range=[0, 1], title="Fraud Score"),
        xaxis_title="Time",
        **CHART_LAYOUT,
    )
    return fig


def category_fraud_bar(log: list[dict]) -> go.Figure:
    df = pd.DataFrame(log)
    if df.empty or "merchant_category" not in df.columns:
        return go.Figure()
    grp = df.groupby("merchant_category").agg(
        fraud_rate=("is_fraud", "mean"),
        count=("is_fraud", "count"),
    ).reset_index().sort_values("fraud_rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=grp["fraud_rate"],
        y=grp["merchant_category"],
        orientation="h",
        marker=dict(
            color=grp["fraud_rate"],
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
        ),
        text=[f"{r:.1%}" for r in grp["fraud_rate"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Fraud Rate by Category",
        xaxis_title="Fraud Rate",
        **CHART_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Engine")
        st.markdown("---")

        api_ok = st.session_state.api_online
        status_color = "#2ecc71" if api_ok else "#e74c3c"
        status_text = "API Online" if api_ok else "Simulation Mode"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{status_color};'
            f'box-shadow:0 0 6px {status_color};"></div>'
            f'<span style="color:{status_color};font-size:0.9rem;">{status_text}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### Settings")
        speed = st.slider("Transactions/refresh", 1, 20, 5)
        show_only_fraud = st.checkbox("Show only fraud alerts", value=False)
        fraud_threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)
        st.markdown("---")

        st.markdown("### Model Info")
        st.caption(f"XGBoost + LightGBM Ensemble")
        st.caption(f"Threshold: {fraud_threshold:.2f}")
        st.caption(f"Features: ~38 engineered")

        st.markdown("---")
        if st.button("🔄 Reset Stats"):
            st.session_state.txn_log = []
            st.session_state.score_history = []
            st.session_state.total_txns = 0
            st.session_state.total_frauds = 0
            st.session_state.latencies = []
            st.session_state.alerts_today = 0

    return speed, show_only_fraud, fraud_threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()
    speed, show_only_fraud, fraud_threshold = render_sidebar()

    # Check API status
    st.session_state.api_online = _check_api()

    # Header
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.markdown("# 🛡️ Real-Time Fraud Detection Engine")
        st.caption("Live transaction monitoring • Ensemble ML • SHAP Explainability")
    with col_time:
        st.markdown(f"<br><p style='color:#8892b0;text-align:right;'>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>", unsafe_allow_html=True)

    # ---- Simulate / fetch new transactions ----
    for _ in range(speed):
        payload = _random_transaction()
        if st.session_state.api_online:
            result = _call_api(payload)
        else:
            result = None
        if result is None:
            result = _simulate_score(payload)

        result["amount"] = payload.get("amount", 0)
        result["merchant_category"] = payload.get("merchant_category", "")
        result["merchant_country"] = payload.get("merchant_country", "")
        result["is_fraud"] = result.get("fraud_score", 0) >= fraud_threshold

        st.session_state.txn_log.append(result)
        st.session_state.score_history.append(result["fraud_score"])
        st.session_state.total_txns += 1
        if result["is_fraud"]:
            st.session_state.total_frauds += 1
            st.session_state.alerts_today += 1
        if result.get("explanation") and result["is_fraud"]:
            st.session_state.last_shap = result["explanation"]
        if result.get("inference_ms"):
            st.session_state.latencies.append(result["inference_ms"])

    # Trim logs
    if len(st.session_state.txn_log) > MAX_LOG_ROWS:
        st.session_state.txn_log = st.session_state.txn_log[-MAX_LOG_ROWS:]
        st.session_state.score_history = st.session_state.score_history[-MAX_LOG_ROWS:]

    # ---- KPI Cards ----
    total = st.session_state.total_txns
    frauds = st.session_state.total_frauds
    fraud_rate = frauds / max(1, total) * 100
    avg_latency = np.mean(st.session_state.latencies[-100:]) if st.session_state.latencies else 0
    alerts = st.session_state.alerts_today

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-label">Transactions Scored</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        color = "#e74c3c" if fraud_rate > 3 else "#f39c12" if fraud_rate > 1 else "#2ecc71"
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:{color};">{fraud_rate:.2f}%</p>
            <p class="kpi-label">Fraud Rate</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">{avg_latency:.1f}ms</p>
            <p class="kpi-label">Avg Inference Latency</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#e74c3c;">{alerts:,}</p>
            <p class="kpi-label">Fraud Alerts</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts Row 1 ----
    c1, c2 = st.columns([2, 1])
    with c1:
        if st.session_state.score_history:
            st.plotly_chart(score_timeline(st.session_state.txn_log), use_container_width=True)
    with c2:
        if st.session_state.txn_log:
            st.plotly_chart(risk_donut(st.session_state.txn_log), use_container_width=True)

    # ---- Charts Row 2 ----
    c3, c4 = st.columns([1, 1])
    with c3:
        if st.session_state.score_history:
            st.plotly_chart(score_histogram(st.session_state.score_history), use_container_width=True)
    with c4:
        if st.session_state.last_shap:
            st.plotly_chart(shap_bar(st.session_state.last_shap), use_container_width=True)
        else:
            st.info("SHAP explanation will appear after the first flagged transaction.")

    # ---- Charts Row 3 ----
    c5, c6 = st.columns([1, 1])
    with c5:
        if st.session_state.txn_log:
            st.plotly_chart(category_fraud_bar(st.session_state.txn_log), use_container_width=True)
    with c6:
        # Country risk heatmap (bar)
        df_log = pd.DataFrame(st.session_state.txn_log)
        if not df_log.empty and "merchant_country" in df_log.columns:
            country_fraud = (
                df_log.groupby("merchant_country")
                .agg(fraud_rate=("is_fraud", "mean"), count=("is_fraud", "count"))
                .reset_index()
                .sort_values("fraud_rate", ascending=False)
                .head(10)
            )
            fig_country = px.bar(
                country_fraud, x="merchant_country", y="fraud_rate",
                color="fraud_rate",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                title="Fraud Rate by Country",
                text=[f"{r:.1%}" for r in country_fraud["fraud_rate"]],
            )
            fig_country.update_layout(**CHART_LAYOUT, xaxis_title="Country", yaxis_title="Fraud Rate")
            st.plotly_chart(fig_country, use_container_width=True)

    # ---- Transaction Feed ----
    st.markdown("### 📋 Live Transaction Feed")

    feed = st.session_state.txn_log.copy()
    if show_only_fraud:
        feed = [r for r in feed if r.get("is_fraud")]
    feed = list(reversed(feed[-50:]))

    if feed:
        rows = []
        for r in feed:
            rows.append({
                "Time": r.get("timestamp", "")[:19].replace("T", " "),
                "Transaction ID": r.get("transaction_id", "")[:16] + "...",
                "Amount": f"${r.get('amount', 0):,.2f}",
                "Category": r.get("merchant_category", ""),
                "Country": r.get("merchant_country", ""),
                "Score": f"{r.get('fraud_score', 0):.4f}",
                "Risk": r.get("risk_level", ""),
                "Decision": r.get("decision", ""),
                "Latency": f"{r.get('inference_ms', 0):.1f}ms",
            })
        df_feed = pd.DataFrame(rows)

        def color_risk(val):
            colors_map = {
                "CRITICAL": "background-color: #e74c3c22; color: #e74c3c; font-weight: bold",
                "HIGH": "background-color: #e67e2222; color: #e67e22; font-weight: bold",
                "MEDIUM": "background-color: #f39c1222; color: #f39c12",
                "LOW": "background-color: #2ecc7122; color: #2ecc71",
            }
            return colors_map.get(val, "")

        def color_decision(val):
            if val == "DECLINE":
                return "background-color: #e74c3c33; color: #e74c3c; font-weight: bold"
            elif val == "REVIEW":
                return "background-color: #f39c1233; color: #f39c12"
            return "background-color: #2ecc7122; color: #2ecc71"

        styled = (
            df_feed.style
            .applymap(color_risk, subset=["Risk"])
            .applymap(color_decision, subset=["Decision"])
        )
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.info("No transactions yet. Refresh will populate the feed.")

    # ---- Drift Status ----
    st.markdown("### 📊 Model Drift Monitor")
    d1, d2, d3 = st.columns(3)
    scores = st.session_state.score_history
    if scores:
        with d1:
            st.metric("Score Mean", f"{np.mean(scores[-500:]):.4f}")
        with d2:
            st.metric("Score P95", f"{np.percentile(scores[-500:], 95):.4f}")
        with d3:
            recent_fraud = sum(1 for s in scores[-500:] if s >= fraud_threshold)
            st.metric("Recent Fraud Rate", f"{recent_fraud/len(scores[-500:])*100:.2f}%")

        # Mini score trend
        trend_df = pd.DataFrame({"score": scores[-200:]})
        trend_df["idx"] = range(len(trend_df))
        trend_df["rolling_mean"] = trend_df["score"].rolling(20, min_periods=1).mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df["idx"], y=trend_df["score"],
            mode="lines", line=dict(color="#2d3250", width=1), name="Score",
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_df["idx"], y=trend_df["rolling_mean"],
            mode="lines", line=dict(color="#4fc3f7", width=2), name="20-txn MA",
        ))
        fig_trend.add_hline(y=fraud_threshold, line_dash="dash", line_color="#e74c3c", opacity=0.6)
        fig_trend.update_layout(
            title="Score Trend (last 200 transactions)",
            height=220, showlegend=True,
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
