"""
Global CSS styles injected into the Streamlit app.
"""

DARK_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117 !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background-color: #131620 !important;
    border-right: 1px solid #1e2235 !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── KPI Cards ────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: #3d4168; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #00d4ff);
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.kpi-delta-positive { color: #10b981; font-size: 0.82rem; font-weight: 500; }
.kpi-delta-negative { color: #ef4444; font-size: 0.82rem; font-weight: 500; }
.kpi-delta-neutral  { color: #6b7280; font-size: 0.82rem; font-weight: 500; }
.kpi-subtext { color: #9ca3af; font-size: 0.78rem; margin-top: 0.2rem; }

/* ── Section headers ──────────────────────────────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2d3148;
}
.section-sub {
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

/* ── Insight cards ────────────────────────────────────────────────── */
.insight-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-left: 4px solid var(--accent, #00d4ff);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.insight-card strong { color: #f1f5f9; }

/* ── Risk badge ───────────────────────────────────────────────────── */
.badge-high   { background:#ef444420; color:#ef4444; border:1px solid #ef4444; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.badge-medium { background:#f59e0b20; color:#f59e0b; border:1px solid #f59e0b; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.badge-low    { background:#10b98120; color:#10b981; border:1px solid #10b981; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #131620;
    border-radius: 8px;
    padding: 0.2rem;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 6px;
    color: #9ca3af;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #1e2235 !important;
    color: #00d4ff !important;
}

/* ── Dataframes ───────────────────────────────────────────────────── */
.stDataFrame { border: 1px solid #2d3148 !important; border-radius: 8px; }

/* ── Misc ─────────────────────────────────────────────────────────── */
hr { border-color: #2d3148 !important; }
.stSpinner > div { color: #00d4ff !important; }
</style>
"""


def inject_css():
    """Call inside a Streamlit app to apply the dark theme."""
    import streamlit as st
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def kpi_card(label: str,
             value: str,
             delta: str = "",
             subtext: str = "",
             accent: str = "#00d4ff",
             delta_positive: bool = True) -> str:
    delta_class = ("kpi-delta-positive" if delta_positive else "kpi-delta-negative") if delta else "kpi-delta-neutral"
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    sub_html   = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""
    return f"""
    <div class="kpi-card" style="--accent:{accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """


def insight_card(text: str, accent: str = "#00d4ff") -> str:
    return f'<div class="insight-card" style="--accent:{accent};">{text}</div>'


def section_header(title: str, subtitle: str = "") -> str:
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-header">{title}</div>{sub_html}'
