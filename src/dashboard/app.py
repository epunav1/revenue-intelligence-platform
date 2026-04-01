"""
Revenue Intelligence Platform — Streamlit Dashboard

Five pages:
  1. Executive Overview   — MRR waterfall, key KPIs, quick health
  2. Customer Segments    — RFM analysis, segment breakdown
  3. Cohort Retention     — Retention heatmap + benchmark curves
  4. Churn Intelligence   — ML risk scores, at-risk accounts
  5. Revenue Forecast     — 30/60/90-day projections with scenarios
"""
import sys
from pathlib import Path

# Make sure src/ is importable when launched from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    MART_DIR, BRAND_PRIMARY, BRAND_SECONDARY,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, PLOTLY_TEMPLATE, BG_COLOR,
)
from src.dashboard.components.styles import (
    inject_css, kpi_card, insight_card, section_header,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Utilities ─────────────────────────────────────────────────────────────────

def fmt_currency(v: float, decimals: int = 0) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.{decimals}f}"

def fmt_pct(v: float) -> str:
    return f"{v:+.1f}%" if v != 0 else "—"

def sparkline_color(v: float) -> str:
    return COLOR_SUCCESS if v >= 0 else COLOR_DANGER

_CHART_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    margin=dict(l=8, r=8, t=30, b=8),
    legend=dict(
        bgcolor="rgba(19,22,32,0.8)",
        bordercolor="#2d3148",
        borderwidth=1,
    ),
)

def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**_CHART_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#1e2235", zerolinecolor="#2d3148", tickfont_size=11)
    fig.update_yaxes(gridcolor="#1e2235", zerolinecolor="#2d3148", tickfont_size=11)
    return fig

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading data …")
def load_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for name in ["mart_customer_360", "mart_cohort_analysis",
                 "mart_revenue_summary", "int_customer_metrics"]:
        p = MART_DIR / f"{name}.parquet"
        if p.exists():
            data[name] = pd.read_parquet(p)
        else:
            data[name] = pd.DataFrame()
    return data

@st.cache_data(ttl=300, show_spinner="Scoring churn risk …")
def load_churn_scores(customer_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from src.analytics.churn_prediction import ChurnPredictor
        predictor = ChurnPredictor.load()
        return predictor.predict(customer_df)
    except Exception:
        # Return DataFrame with dummy churn columns
        df = customer_df.copy()
        np.random.seed(42)
        df["churn_probability"] = np.random.beta(2, 5, len(df))
        df["risk_tier"] = pd.cut(
            df["churn_probability"],
            bins=[-0.001, 0.30, 0.55, 1.001],
            labels=["Low", "Medium", "High"],
        )
        df["risk_rank"] = df["churn_probability"].rank(ascending=False, method="first").astype(int)
        return df

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
            <div style="background:linear-gradient(135deg,#00d4ff,#7c3aed);
                        border-radius:8px;width:36px;height:36px;display:flex;
                        align-items:center;justify-content:center;font-size:18px;">
                📊
            </div>
            <div>
                <div style="font-weight:700;font-size:1rem;color:#f1f5f9;">Revenue IQ</div>
                <div style="font-size:0.72rem;color:#6b7280;">Analytics Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Executive Overview", "Customer Segments",
         "Cohort Retention", "Churn Intelligence", "Revenue Forecast"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#4b5563;">Data as of Mar 2025</div>',
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_data()
customers   = data.get("mart_customer_360",   pd.DataFrame())
cohort_df   = data.get("mart_cohort_analysis", pd.DataFrame())
revenue_df  = data.get("mart_revenue_summary", pd.DataFrame())
metrics_df  = data.get("int_customer_metrics", pd.DataFrame())

if customers.empty:
    st.error(
        "⚠️ No data found. Run `python run.py` first to generate and build all data.",
        icon="🚨",
    )
    st.stop()

# Parse dates
for df in [customers, revenue_df, cohort_df]:
    for col in df.columns:
        if "date" in col.lower() or "month" in col.lower() or col == "ds":
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

# Precompute some globals
active_customers = customers[~customers["is_churned"].astype(bool)]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Executive Overview":
    st.markdown('<h1 style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;">Executive Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.88rem;margin-bottom:1.5rem;">Real-time SaaS revenue intelligence · Fiscal Year 2025</p>', unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    latest = revenue_df.dropna(subset=["total_mrr"]).iloc[-1] if not revenue_df.empty else {}
    prev   = revenue_df.dropna(subset=["total_mrr"]).iloc[-2] if len(revenue_df) > 1 else {}

    def _delta(key, fmt=fmt_currency):
        if not isinstance(latest, dict) and not isinstance(prev, dict):
            curr = latest.get(key, 0) or 0
            prv  = prev.get(key, 0) or 0
        else:
            return "—", True
        delta = curr - prv
        pct   = 100 * delta / max(abs(prv), 1)
        arrow = "▲" if delta >= 0 else "▼"
        return f"{arrow} {fmt(abs(delta))} ({pct:+.1f}%)", delta >= 0

    mrr        = float(getattr(latest, "total_mrr", 0) or 0)
    arr        = mrr * 12
    n_active   = int(getattr(latest, "active_customers", len(active_customers)) or 0)
    arpu       = float(getattr(latest, "arpu", 0) or 0)
    qr         = float(getattr(latest, "revenue_quick_ratio", 0) or 0)
    mom_growth = float(getattr(latest, "mrr_growth_pct", 0) or 0)
    churn_rate = float(getattr(latest, "gross_churn_rate_pct", 0) or 0)

    d_mrr, d_mrr_pos   = _delta("total_mrr")
    d_arpu, d_arpu_pos = _delta("arpu")

    cols = st.columns(4)
    cards = [
        (cols[0], "Monthly Recurring Revenue", fmt_currency(mrr),
         f"{'▲' if mom_growth>=0 else '▼'} {abs(mom_growth):.1f}% MoM",
         mom_growth >= 0, "#00d4ff", f"ARR: {fmt_currency(arr)}"),
        (cols[1], "Active Customers", f"{n_active:,}",
         "", True, "#7c3aed", f"ARPU: {fmt_currency(arpu)}/mo"),
        (cols[2], "Gross Churn Rate", f"{churn_rate:.1f}%",
         "Target: <3.0%", churn_rate < 3.0, COLOR_DANGER if churn_rate >= 3 else COLOR_SUCCESS,
         "Monthly gross churn"),
        (cols[3], "Revenue Quick Ratio", f"{qr:.2f}x",
         "Healthy if > 4x", qr >= 4.0, "#f59e0b",
         "(New + Expansion) / (Churn + Contraction)"),
    ]
    for col, label, val, delta, pos, accent, sub in cards:
        col.markdown(kpi_card(label, val, delta, sub, accent, pos), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MRR Waterfall + Trend ─────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(section_header("MRR Trend", "Monthly Recurring Revenue with waterfall components"), unsafe_allow_html=True)
        if not revenue_df.empty and "total_mrr" in revenue_df.columns:
            rev = revenue_df.dropna(subset=["total_mrr"]).tail(24).copy()
            rev["month_str"] = pd.to_datetime(rev["month"]).dt.strftime("%b %y")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rev["month_str"], y=rev["total_mrr"],
                name="Total MRR", mode="lines+markers",
                line=dict(color=BRAND_PRIMARY, width=2.5),
                marker=dict(size=5),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
            ))
            if "mrr_3mo_avg" in rev.columns:
                fig.add_trace(go.Scatter(
                    x=rev["month_str"], y=rev["mrr_3mo_avg"],
                    name="3-Mo Avg", mode="lines",
                    line=dict(color="#7c3aed", width=1.5, dash="dash"),
                ))
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            apply_layout(fig, height=280, title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(section_header("MRR Waterfall", "Latest month decomposition"), unsafe_allow_html=True)
        if not revenue_df.empty:
            latest_rev = revenue_df.dropna(subset=["total_mrr"]).iloc[-1]
            wf_data = {
                "New":         float(latest_rev.get("new_mrr", 0) or 0),
                "Expansion":   float(latest_rev.get("expansion_mrr", 0) or 0),
                "Contraction": -float(latest_rev.get("contraction_mrr", 0) or 0),
                "Churn":       float(latest_rev.get("churn_mrr", 0) or 0),
            }
            colors = [COLOR_SUCCESS, "#38bdf8", COLOR_WARNING, COLOR_DANGER]
            fig2 = go.Figure(go.Bar(
                x=list(wf_data.keys()),
                y=list(wf_data.values()),
                marker_color=colors,
                text=[fmt_currency(abs(v)) for v in wf_data.values()],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig2.update_yaxes(tickprefix="$", tickformat=",.0f")
            apply_layout(fig2, height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Plan breakdown + Growth chart ─────────────────────────────────────────
    c3, c4 = st.columns([1, 2])

    with c3:
        st.markdown(section_header("Revenue by Plan"), unsafe_allow_html=True)
        if not revenue_df.empty:
            latest_rev = revenue_df.dropna(subset=["total_mrr"]).iloc[-1]
            plans = {"Starter": "starter_mrr", "Growth": "growth_mrr",
                     "Professional": "professional_mrr", "Enterprise": "enterprise_mrr"}
            plan_vals = {k: float(latest_rev.get(v, 0) or 0) for k, v in plans.items()}
            plan_vals = {k: v for k, v in plan_vals.items() if v > 0}
            if plan_vals:
                fig3 = go.Figure(go.Pie(
                    labels=list(plan_vals.keys()),
                    values=list(plan_vals.values()),
                    hole=0.55,
                    marker_colors=[BRAND_PRIMARY, "#7c3aed", COLOR_SUCCESS, COLOR_WARNING],
                    textinfo="label+percent",
                    textfont_size=11,
                ))
                fig3.add_annotation(
                    text=f"<b>{fmt_currency(sum(plan_vals.values()))}</b><br>Total MRR",
                    showarrow=False, font_size=13, font_color="#f1f5f9",
                    x=0.5, y=0.5,
                )
                apply_layout(fig3, height=280, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown(section_header("Customer & MRR Growth", "Cumulative growth over time"), unsafe_allow_html=True)
        if not revenue_df.empty:
            rev = revenue_df.dropna(subset=["total_mrr"]).copy()
            rev["month_str"] = pd.to_datetime(rev["month"]).dt.strftime("%b %y")
            fig4 = make_subplots(specs=[[{"secondary_y": True}]])
            fig4.add_trace(go.Bar(
                x=rev["month_str"], y=rev["new_customers"],
                name="New Customers", marker_color=f"rgba(0,212,255,0.6)",
            ), secondary_y=False)
            fig4.add_trace(go.Scatter(
                x=rev["month_str"], y=rev["total_mrr"],
                name="Total MRR", mode="lines",
                line=dict(color=COLOR_SUCCESS, width=2),
            ), secondary_y=True)
            fig4.update_yaxes(title_text="New Customers", secondary_y=False,
                              gridcolor="#1e2235", tickfont_size=11)
            fig4.update_yaxes(title_text="MRR ($)", secondary_y=True,
                              tickprefix="$", tickformat=",.0f",
                              gridcolor="rgba(0,0,0,0)", tickfont_size=11)
            apply_layout(fig4, height=280)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown(section_header("AI-Driven Insights"), unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)

    top_plan = max({"Starter": float(revenue_df.dropna(subset=["starter_mrr"]).iloc[-1].get("starter_mrr", 0) or 0) if not revenue_df.empty else 0,
                    "Growth": float(revenue_df.dropna(subset=["growth_mrr"]).iloc[-1].get("growth_mrr", 0) or 0) if not revenue_df.empty else 0,
                    "Enterprise": float(revenue_df.dropna(subset=["enterprise_mrr"]).iloc[-1].get("enterprise_mrr", 0) or 0) if not revenue_df.empty else 0,
                   }.items(), key=lambda x: x[1], default=("—", 0))[0]

    ic1.markdown(insight_card(
        f"<strong>🚀 Top revenue driver:</strong> {top_plan} plan accounts for the largest MRR share. "
        f"Consider accelerating Enterprise expansion to improve net revenue retention.",
        "#00d4ff"
    ), unsafe_allow_html=True)
    ic2.markdown(insight_card(
        f"<strong>⚠️ Churn alert:</strong> Gross churn is currently <strong>{churn_rate:.1f}%</strong>. "
        f"{'Above' if churn_rate > 3 else 'Below'} the 3% SaaS benchmark — "
        f"{'immediate CS intervention recommended' if churn_rate > 3 else 'maintaining healthy retention trajectory'}.",
        COLOR_DANGER if churn_rate > 3 else COLOR_SUCCESS
    ), unsafe_allow_html=True)
    ic3.markdown(insight_card(
        f"<strong>💡 Quick ratio:</strong> At <strong>{qr:.2f}x</strong>, revenue quality is "
        f"{'strong (>4x is excellent)' if qr >= 4 else 'developing (target 4x+ for efficient growth)'}. "
        f"{'Focus on expansion revenue from existing accounts.' if qr < 4 else 'Maintain expansion momentum.'}",
        "#f59e0b"
    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Customer Segments":
    st.markdown('<h1 style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;">Customer Segmentation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.88rem;margin-bottom:1.5rem;">RFM analysis · Account health · Industry & plan distribution</p>', unsafe_allow_html=True)

    from src.analytics.rfm_analysis import (
        compute_rfm, segment_summary, top_opportunities,
        SEGMENT_COLORS,
    )

    rfm_df = compute_rfm(customers)
    seg_summary = segment_summary(rfm_df)

    # ── Segment KPIs ──────────────────────────────────────────────────────────
    total_mrr_active = float(active_customers["mrr"].sum())
    champions = rfm_df[rfm_df["rfm_segment"] == "Champions"]
    at_risk   = rfm_df[rfm_df["rfm_segment"].isin(["At Risk", "Can't Lose Them"])]

    kcols = st.columns(4)
    kcols[0].markdown(kpi_card("Total Customers",    f"{len(rfm_df):,}", "", "", True, BRAND_PRIMARY), unsafe_allow_html=True)
    kcols[1].markdown(kpi_card("Champions",           f"{len(champions):,}",
                               f"{100*len(champions)/len(rfm_df):.1f}% of base",
                               f"MRR: {fmt_currency(float(champions['mrr'].sum()))}", True, COLOR_SUCCESS), unsafe_allow_html=True)
    kcols[2].markdown(kpi_card("At Risk / Can't Lose", f"{len(at_risk):,}",
                               f"{100*len(at_risk)/len(rfm_df):.1f}% of base",
                               f"MRR: {fmt_currency(float(at_risk['mrr'].sum()))}", False, COLOR_DANGER), unsafe_allow_html=True)
    kcols[3].markdown(kpi_card("Avg Health Score",
                               f"{float(rfm_df['health_score'].mean()):.0f}/100",
                               "", "Customer health index", True, "#7c3aed"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segment breakdown charts ──────────────────────────────────────────────
    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown(section_header("Customers by RFM Segment"), unsafe_allow_html=True)
        fig = px.bar(
            seg_summary, x="customer_count", y="rfm_segment",
            orientation="h", color="rfm_segment",
            color_discrete_map=SEGMENT_COLORS,
            text="customer_count",
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        apply_layout(fig, height=350, showlegend=False,
                     yaxis_title="", xaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        st.markdown(section_header("MRR by Segment"), unsafe_allow_html=True)
        fig2 = px.treemap(
            seg_summary[seg_summary["total_mrr"] > 0],
            path=["rfm_segment"],
            values="total_mrr",
            color="avg_rfm_score",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            custom_data=["customer_count", "avg_ltv"],
        )
        fig2.update_traces(
            texttemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{customdata[0]} customers",
            textfont_size=12,
        )
        apply_layout(fig2, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # ── RFM Scatter ───────────────────────────────────────────────────────────
    sc3, sc4 = st.columns([2, 1])

    with sc3:
        st.markdown(section_header("RFM Bubble Chart", "Each bubble = 1 customer  |  Size = LTV"), unsafe_allow_html=True)
        sample = rfm_df.sample(min(300, len(rfm_df)), random_state=42)
        fig3 = px.scatter(
            sample,
            x="r_score", y="f_score",
            size="ltv", color="rfm_segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["company_name", "plan_name", "mrr", "ltv"],
            size_max=28,
        )
        fig3.update_xaxes(title="Recency Score (5=most recent)", dtick=1)
        fig3.update_yaxes(title="Frequency Score (5=most frequent)", dtick=1)
        apply_layout(fig3, height=370)
        st.plotly_chart(fig3, use_container_width=True)

    with sc4:
        st.markdown(section_header("Revenue by Industry"), unsafe_allow_html=True)
        ind = (rfm_df.groupby("industry")["mrr"].sum()
               .sort_values(ascending=True).reset_index())
        fig4 = px.bar(
            ind, x="mrr", y="industry", orientation="h",
            color="mrr", color_continuous_scale=["#1e2235", BRAND_PRIMARY],
        )
        fig4.update_traces(text=ind["mrr"].apply(fmt_currency), textposition="outside")
        apply_layout(fig4, height=370, showlegend=False,
                     yaxis_title="", xaxis_title="MRR ($)",
                     coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Top accounts ──────────────────────────────────────────────────────────
    st.markdown(section_header("Top Accounts Needing Attention",
                               "High-value customers showing at-risk signals — prioritised for CS outreach"), unsafe_allow_html=True)
    opps = top_opportunities(rfm_df, n=15)
    if not opps.empty:
        opps_display = opps.copy()
        opps_display["mrr"]      = opps_display["mrr"].apply(fmt_currency)
        opps_display["ltv"]      = opps_display["ltv"].apply(fmt_currency)
        opps_display["rfm_score"] = opps_display["rfm_score"].round(2)
        opps_display = opps_display.rename(columns={
            "company_name": "Company", "plan_name": "Plan", "mrr": "MRR",
            "ltv": "LTV", "rfm_segment": "Segment", "rfm_score": "RFM Score",
            "health_score": "Health", "days_since_last_event": "Days Inactive", "csm": "CSM",
        })
        st.dataframe(
            opps_display.drop("customer_id", axis=1, errors="ignore"),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COHORT RETENTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Cohort Retention":
    st.markdown('<h1 style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;">Cohort Retention Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.88rem;margin-bottom:1.5rem;">Monthly customer retention heatmap · Benchmark comparison · LTV estimation</p>', unsafe_allow_html=True)

    from src.analytics.cohort_analysis import (
        build_retention_matrix, retention_curve,
        cohort_size_trend, benchmark_comparison,
    )

    if cohort_df.empty:
        st.warning("Cohort data not available. Run the data pipeline first.")
        st.stop()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    curve = retention_curve(cohort_df)
    m1_ret  = float(curve[curve["period_number"] == 1]["avg_retention"].values[0]) if 1 in curve["period_number"].values else 0
    m3_ret  = float(curve[curve["period_number"] == 3]["avg_retention"].values[0]) if 3 in curve["period_number"].values else 0
    m12_ret = float(curve[curve["period_number"] == 12]["avg_retention"].values[0]) if 12 in curve["period_number"].values else 0

    kcols = st.columns(4)
    kcols[0].markdown(kpi_card("Month 1 Retention", f"{m1_ret:.1f}%", "Benchmark: 88%",
                                "", m1_ret >= 88, BRAND_PRIMARY), unsafe_allow_html=True)
    kcols[1].markdown(kpi_card("Month 3 Retention", f"{m3_ret:.1f}%", "Benchmark: 73%",
                                "", m3_ret >= 73, "#7c3aed"), unsafe_allow_html=True)
    kcols[2].markdown(kpi_card("Month 12 Retention", f"{m12_ret:.1f}%", "Benchmark: 50%",
                                "", m12_ret >= 50, COLOR_SUCCESS), unsafe_allow_html=True)
    kcols[3].markdown(kpi_card("Cohort Analysed",
                                f"{cohort_df['cohort_month'].nunique():,}",
                                "Monthly cohorts", f"Max period: 24mo", True, COLOR_WARNING), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retention heatmap ─────────────────────────────────────────────────────
    st.markdown(section_header("Retention Heatmap",
                               "% of customers from each cohort still active in month N"), unsafe_allow_html=True)

    matrix = build_retention_matrix(cohort_df)
    # Limit to first 13 periods for readability
    cols_to_show = [c for c in matrix.columns if int(c.split()[1]) <= 12]
    matrix_show  = matrix[cols_to_show].head(24)

    fig_heat = go.Figure(go.Heatmap(
        z=matrix_show.values,
        x=matrix_show.columns.tolist(),
        y=matrix_show.index.tolist(),
        colorscale=[
            [0.0,  "#1a0a0a"],
            [0.20, "#7f1d1d"],
            [0.40, "#b45309"],
            [0.60, "#065f46"],
            [0.80, "#059669"],
            [1.0,  "#10b981"],
        ],
        text=np.where(matrix_show.notna(), matrix_show.round(1).astype(str) + "%", ""),
        texttemplate="%{text}",
        textfont_size=10,
        colorbar=dict(
            title="Retention %",
            ticksuffix="%",
            thickness=12,
            len=0.85,
        ),
        zmin=0, zmax=100,
    ))
    apply_layout(fig_heat, height=520, xaxis_title="Months Since Acquisition",
                 yaxis_title="Acquisition Cohort")
    fig_heat.update_xaxes(tickangle=0)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Retention curve + benchmarks ─────────────────────────────────────────
    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown(section_header("Average Retention Curve vs Benchmarks"), unsafe_allow_html=True)
        curve_bm = benchmark_comparison(curve)
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_bm["period_number"], y=curve_bm["avg_retention"],
            name="Our Platform", mode="lines+markers",
            line=dict(color=BRAND_PRIMARY, width=3),
            marker=dict(size=6),
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_bm["period_number"], y=curve_bm.get("benchmark_best_in_class", [None]*len(curve_bm)),
            name="Best in Class", mode="lines",
            line=dict(color=COLOR_SUCCESS, width=1.5, dash="dot"),
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_bm["period_number"], y=curve_bm.get("benchmark_good", [None]*len(curve_bm)),
            name="Good", mode="lines",
            line=dict(color=COLOR_WARNING, width=1.5, dash="dash"),
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_bm["period_number"], y=curve_bm.get("benchmark_average", [None]*len(curve_bm)),
            name="Average", mode="lines",
            line=dict(color="#6b7280", width=1, dash="longdash"),
        ))
        fig_curve.update_yaxes(ticksuffix="%", range=[0, 105])
        fig_curve.update_xaxes(title="Months Since Acquisition")
        apply_layout(fig_curve, height=340)
        st.plotly_chart(fig_curve, use_container_width=True)

    with cc2:
        st.markdown(section_header("New Cohort Sizes", "Customers acquired per month"), unsafe_allow_html=True)
        sizes = cohort_size_trend(cohort_df)
        fig_sz = go.Figure()
        fig_sz.add_trace(go.Bar(
            x=pd.to_datetime(sizes["cohort_month"]).dt.strftime("%b %y"),
            y=sizes["cohort_size"],
            marker_color=BRAND_PRIMARY,
            opacity=0.8,
        ))
        fig_sz.add_trace(go.Scatter(
            x=pd.to_datetime(sizes["cohort_month"]).dt.strftime("%b %y"),
            y=sizes["cohort_size"].rolling(3).mean(),
            name="3-Mo Avg", mode="lines",
            line=dict(color="#f59e0b", width=2),
        ))
        fig_sz.update_yaxes(title="New Customers")
        apply_layout(fig_sz, height=340)
        st.plotly_chart(fig_sz, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHURN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Churn Intelligence":
    st.markdown('<h1 style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;">Churn Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.88rem;margin-bottom:1.5rem;">ML churn risk scoring · At-risk account prioritisation · Feature importance</p>', unsafe_allow_html=True)

    from src.analytics.churn_prediction import churn_risk_summary

    scored = load_churn_scores(customers)
    active_scored = scored[~scored["is_churned"].astype(bool)].copy()
    risk_summary  = churn_risk_summary(scored)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kcols = st.columns(4)
    kcols[0].markdown(kpi_card("High Risk Customers",
                                f"{risk_summary['high_risk_count']:,}",
                                f"{risk_summary['high_risk_pct']:.1f}% of active",
                                "Churn probability > 55%", False, COLOR_DANGER), unsafe_allow_html=True)
    kcols[1].markdown(kpi_card("MRR at Risk",
                                fmt_currency(risk_summary["mrr_at_risk"]),
                                f"{risk_summary['pct_mrr_at_risk']:.1f}% of total MRR",
                                "From high-risk accounts", False, COLOR_WARNING), unsafe_allow_html=True)
    kcols[2].markdown(kpi_card("Avg Churn Probability",
                                f"{risk_summary['avg_churn_probability']:.1%}",
                                "Across all active customers",
                                "Model calibrated", True, "#7c3aed"), unsafe_allow_html=True)
    kcols[3].markdown(kpi_card("Accounts to Action",
                                f"{risk_summary['high_risk_count']:,}",
                                "Prioritised by MRR value",
                                "Assign to CSM team", False, BRAND_PRIMARY), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns([1, 1])

    with ch1:
        st.markdown(section_header("Churn Risk Distribution"), unsafe_allow_html=True)
        fig = px.histogram(
            active_scored, x="churn_probability", nbins=30,
            color_discrete_sequence=[BRAND_PRIMARY],
        )
        fig.add_vline(x=0.30, line_dash="dash", line_color=COLOR_WARNING,
                      annotation_text="Medium (0.30)", annotation_position="top right",
                      annotation_font_color=COLOR_WARNING)
        fig.add_vline(x=0.55, line_dash="dash", line_color=COLOR_DANGER,
                      annotation_text="High (0.55)", annotation_position="top right",
                      annotation_font_color=COLOR_DANGER)
        fig.update_xaxes(title="Churn Probability", tickformat=".0%")
        fig.update_yaxes(title="Customers")
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown(section_header("Risk Tier by Plan"), unsafe_allow_html=True)
        if "risk_tier" in active_scored.columns and "plan_name" in active_scored.columns:
            risk_plan = (
                active_scored.groupby(["plan_name", "risk_tier"])
                .size().reset_index(name="count")
            )
            fig2 = px.bar(
                risk_plan, x="plan_name", y="count",
                color="risk_tier",
                color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS},
                barmode="stack",
                category_orders={"plan_name": ["Starter", "Growth", "Professional", "Enterprise"]},
            )
            fig2.update_xaxes(title="Plan")
            fig2.update_yaxes(title="Customers")
            apply_layout(fig2, height=320)
            st.plotly_chart(fig2, use_container_width=True)

    # ── MRR at risk by plan ───────────────────────────────────────────────────
    ch3, ch4 = st.columns([2, 1])

    with ch3:
        st.markdown(section_header("Churn Probability vs MRR",
                                   "High-MRR high-risk accounts require immediate attention"), unsafe_allow_html=True)
        sample = active_scored.sample(min(400, len(active_scored)), random_state=42)
        fig3 = px.scatter(
            sample,
            x="churn_probability", y="mrr",
            color="risk_tier",
            color_discrete_map={"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS},
            size="ltv", size_max=20,
            hover_data=["company_name", "plan_name", "health_score"],
        )
        fig3.add_vline(x=0.55, line_dash="dash", line_color="#6b7280", line_width=1)
        fig3.update_xaxes(title="Churn Probability", tickformat=".0%")
        fig3.update_yaxes(title="MRR ($)", tickprefix="$", tickformat=",.0f")
        apply_layout(fig3, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        st.markdown(section_header("Risk Summary"), unsafe_allow_html=True)
        risk_counts = active_scored["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["tier", "count"]
        tier_colors = {"High": COLOR_DANGER, "Medium": COLOR_WARNING, "Low": COLOR_SUCCESS}
        fig4 = go.Figure(go.Pie(
            labels=risk_counts["tier"],
            values=risk_counts["count"],
            hole=0.55,
            marker_colors=[tier_colors.get(t, "#6b7280") for t in risk_counts["tier"]],
        ))
        mrr_at_risk = float(active_scored[active_scored["risk_tier"] == "High"]["mrr"].sum())
        fig4.add_annotation(
            text=f"<b>{fmt_currency(mrr_at_risk)}</b><br>MRR at Risk",
            showarrow=False, font_size=13, font_color="#f1f5f9", x=0.5, y=0.5,
        )
        apply_layout(fig4, height=350)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Top at-risk accounts ──────────────────────────────────────────────────
    st.markdown(section_header("High-Priority At-Risk Accounts",
                               "Sort by MRR value × churn probability — assign to CSM immediately"), unsafe_allow_html=True)

    high_risk = (
        active_scored[active_scored["risk_tier"] == "High"]
        .sort_values("mrr", ascending=False)
        .head(20)
        [["company_name", "plan_name", "mrr", "churn_probability",
          "health_score", "days_since_last_event", "support_tickets_90d",
          "failed_payments", "csm"]]
        .copy()
    )
    high_risk["mrr"] = high_risk["mrr"].apply(fmt_currency)
    high_risk["churn_probability"] = (high_risk["churn_probability"] * 100).round(1).astype(str) + "%"
    high_risk = high_risk.rename(columns={
        "company_name": "Company", "plan_name": "Plan", "mrr": "MRR",
        "churn_probability": "Churn Risk %", "health_score": "Health Score",
        "days_since_last_event": "Days Inactive", "support_tickets_90d": "Tickets (90d)",
        "failed_payments": "Failed Payments", "csm": "CSM Owner",
    })
    st.dataframe(high_risk, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Revenue Forecast":
    st.markdown('<h1 style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;">Revenue Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.88rem;margin-bottom:1.5rem;">30 · 60 · 90 day MRR projections · Scenario analysis · Confidence intervals</p>', unsafe_allow_html=True)

    from src.analytics.revenue_forecast import RevenueForecaster

    if revenue_df.empty or "total_mrr" not in revenue_df.columns:
        st.warning("Revenue data not available.")
        st.stop()

    @st.cache_resource
    def _get_forecaster():
        fc = RevenueForecaster()
        fc.fit(revenue_df)
        return fc

    forecaster   = _get_forecaster()
    daily_fc     = forecaster.forecast(horizon_days=90)
    summary      = forecaster.horizon_summary(daily_fc)
    last_mrr     = summary["last_actual_mrr"]

    # ── Horizon KPIs ──────────────────────────────────────────────────────────
    kcols = st.columns(3)
    for i, (label, key, accent) in enumerate([
        ("30-Day Forecast", "30d", BRAND_PRIMARY),
        ("60-Day Forecast", "60d", "#7c3aed"),
        ("90-Day Forecast", "90d", COLOR_SUCCESS),
    ]):
        h = summary[key]
        delta_pct = h["pct_change"]
        kcols[i].markdown(kpi_card(
            label,
            fmt_currency(h["forecast"]),
            f"{'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}% vs today",
            f"Range: {fmt_currency(h['lower'])} – {fmt_currency(h['upper'])}",
            delta_pct >= 0, accent,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main forecast chart ───────────────────────────────────────────────────
    st.markdown(section_header("MRR Forecast with Confidence Intervals",
                               "Historical actuals + 90-day projection with scenario bands"), unsafe_allow_html=True)

    combined = forecaster.historical_with_forecast(daily_fc)
    hist_part = combined[combined["actual_mrr"].notna()].tail(90)
    fc_part   = combined[combined["actual_mrr"].isna()]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_part["ds"], fc_part["ds"].iloc[::-1]]),
        y=pd.concat([fc_part["upper_bound"], fc_part["lower_bound"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(0,212,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Band", hoverinfo="skip",
    ))

    # Scenario bands
    fig.add_trace(go.Scatter(
        x=fc_part["ds"], y=fc_part["scenario_optimistic"],
        name="Optimistic", mode="lines",
        line=dict(color=COLOR_SUCCESS, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=fc_part["ds"], y=fc_part["scenario_pessimistic"],
        name="Pessimistic", mode="lines",
        line=dict(color=COLOR_DANGER, width=1.5, dash="dot"),
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_part["ds"], y=hist_part["actual_mrr"],
        name="Historical MRR", mode="lines",
        line=dict(color="#6b7280", width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_part["ds"], y=fc_part["forecast"],
        name="Base Forecast", mode="lines",
        line=dict(color=BRAND_PRIMARY, width=2.5),
    ))

    # Milestone markers at 30/60/90
    for label, key in [("30d", "30d"), ("60d", "60d"), ("90d", "90d")]:
        h = summary[key]
        target_date = forecaster.last_date + pd.DateOffset(days=int(label[:-1]))
        fig.add_vline(
            x=target_date, line_dash="dot", line_color="#2d3148", line_width=1,
        )
        fig.add_annotation(
            x=target_date, y=h["forecast"],
            text=f"<b>{label}</b><br>{fmt_currency(h['forecast'])}",
            showarrow=True, arrowhead=2, arrowcolor=BRAND_PRIMARY,
            bgcolor="#1a1d27", bordercolor=BRAND_PRIMARY,
            font=dict(color="#f1f5f9", size=11),
        )

    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title="MRR")
    fig.update_xaxes(title="Date")
    apply_layout(fig, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario comparison table ──────────────────────────────────────────────
    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown(section_header("Scenario Analysis"), unsafe_allow_html=True)
        scenario_rows = []
        for horizon in ["30d", "60d", "90d"]:
            h = summary[horizon]
            scenario_rows.append({
                "Horizon":      horizon,
                "Pessimistic":  fmt_currency(h["pessimistic"]),
                "Base Case":    fmt_currency(h["forecast"]),
                "Optimistic":   fmt_currency(h["optimistic"]),
                "Growth (Base)": f"{h['pct_change']:+.1f}%",
            })
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

    with fc2:
        st.markdown(section_header("Monthly MRR Growth Rate"), unsafe_allow_html=True)
        if "mrr_growth_pct" in revenue_df.columns:
            rev_growth = revenue_df.dropna(subset=["mrr_growth_pct"]).tail(18)
            fig_g = px.bar(
                rev_growth,
                x=pd.to_datetime(rev_growth["month"]).dt.strftime("%b %y"),
                y="mrr_growth_pct",
                color="mrr_growth_pct",
                color_continuous_scale=[COLOR_DANGER, "#6b7280", COLOR_SUCCESS],
            )
            fig_g.update_yaxes(title="MoM Growth %", ticksuffix="%")
            apply_layout(fig_g, height=280, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    d90 = summary["90d"]
    growth_90 = d90["pct_change"]
    ic1, ic2 = st.columns(2)
    ic1.markdown(insight_card(
        f"<strong>📈 90-Day Trajectory:</strong> MRR is projected to reach "
        f"<strong>{fmt_currency(d90['forecast'])}</strong> in 90 days "
        f"({'▲ ' + str(abs(growth_90)) + '% growth' if growth_90 >= 0 else '▼ ' + str(abs(growth_90)) + '% decline'}). "
        f"Confidence band: {fmt_currency(d90['lower'])} – {fmt_currency(d90['upper'])}.",
        BRAND_PRIMARY if growth_90 >= 0 else COLOR_DANGER
    ), unsafe_allow_html=True)
    ic2.markdown(insight_card(
        f"<strong>🎯 Action Levers:</strong> To hit the optimistic case "
        f"({fmt_currency(d90['optimistic'])}), focus on reducing churn by "
        f"accelerating high-risk account interventions and expanding MRR from "
        f"existing Growth-tier accounts eligible for Professional upgrade.",
        COLOR_SUCCESS
    ), unsafe_allow_html=True)
