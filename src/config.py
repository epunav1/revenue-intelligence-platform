"""
Central configuration for the Revenue Intelligence Platform.
"""
import os
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
STG_DIR    = DATA_DIR / "staging"
INT_DIR    = DATA_DIR / "intermediate"
MART_DIR   = DATA_DIR / "mart"
MODELS_DIR = ROOT_DIR / "src" / "models"
ASSETS_DIR = ROOT_DIR / "assets"

DB_PATH    = DATA_DIR / "revenue_intelligence.duckdb"

# ensure directories exist
for _d in [RAW_DIR, STG_DIR, INT_DIR, MART_DIR, ASSETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Simulation window ────────────────────────────────────────────────────────
SIM_START   = datetime(2022, 1, 1)
SIM_END     = datetime(2025, 3, 31)
RANDOM_SEED = 42

# ── Subscription plans ───────────────────────────────────────────────────────
PLANS: dict = {
    "Starter": {
        "price_monthly":  299,
        "price_annual":  2990,   # ~2 months free
        "churn_rate":    0.075,  # monthly
        "upsell_rate":   0.02,
        "weight":        0.35,
        "seats_range":   (1, 5),
    },
    "Growth": {
        "price_monthly":  799,
        "price_annual":  7990,
        "churn_rate":    0.045,
        "upsell_rate":   0.025,
        "weight":        0.33,
        "seats_range":   (3, 15),
    },
    "Professional": {
        "price_monthly": 1499,
        "price_annual": 14990,
        "churn_rate":    0.025,
        "upsell_rate":   0.015,
        "weight":        0.21,
        "seats_range":   (10, 50),
    },
    "Enterprise": {
        "price_monthly": 2999,
        "price_annual": 29990,
        "churn_rate":    0.012,
        "upsell_rate":   0.01,
        "weight":        0.11,
        "seats_range":   (25, 200),
    },
}

PLAN_ORDER = ["Starter", "Growth", "Professional", "Enterprise"]

INDUSTRIES = [
    "SaaS", "E-commerce", "FinTech", "Healthcare", "Marketing Tech",
    "EdTech", "Manufacturing", "Real Estate", "Legal Tech", "Consulting",
    "Logistics", "Media & Entertainment",
]

ACQUISITION_CHANNELS = [
    "organic_search", "paid_search", "referral", "social_media",
    "partner", "direct", "content_marketing", "outbound_sales",
]

COUNTRIES = {
    "US":  0.52,
    "UK":  0.12,
    "Canada": 0.08,
    "Australia": 0.06,
    "Germany": 0.05,
    "France": 0.04,
    "Netherlands": 0.03,
    "Sweden": 0.02,
    "Singapore": 0.04,
    "Japan": 0.04,
}

CSM_NAMES = [
    "Alex Rivera", "Jordan Kim", "Sam Patel", "Taylor Brooks",
    "Morgan Chen", "Casey Zhang", "Drew Okafor", "Quinn Nakamura",
]

# ── Analytics parameters ─────────────────────────────────────────────────────
RFM_QUANTILES     = 5       # quintile scoring
CHURN_THRESHOLD   = 0.55    # probability → high risk
FORECAST_HORIZON  = 90      # days
COHORT_MONTHS     = 24      # look-back for cohort analysis

# ── Dashboard ────────────────────────────────────────────────────────────────
BRAND_PRIMARY   = "#00d4ff"
BRAND_SECONDARY = "#7c3aed"
COLOR_SUCCESS   = "#10b981"
COLOR_WARNING   = "#f59e0b"
COLOR_DANGER    = "#ef4444"
COLOR_NEUTRAL   = "#6b7280"

PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR        = "#0f1117"
CARD_BG         = "#1a1d27"
BORDER_COLOR    = "#2d3148"
