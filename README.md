# Revenue Intelligence Platform

> Production-grade analytics system for B2B SaaS companies — customer segmentation, cohort retention, churn prediction, and revenue forecasting in a single interactive dashboard.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Revenue Intelligence Platform                    │
└────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐    ┌──────────────────────────────────────────┐
  │  Data Generation │    │           SQL Model Layer (DuckDB)       │
  │  (Synthetic SaaS)│    │                                          │
  │                 │    │  STAGING          INTERMEDIATE   MART     │
  │  customers      │───▶│  stg_customers    int_customer   customer │
  │  subscriptions  │    │  stg_subscriptions  _metrics     _360    │
  │  transactions   │    │  stg_transactions int_monthly   cohort   │
  │  product_events │    │  stg_product       _revenue     _analysis│
  │                 │    │    _events                      revenue  │
  └─────────────────┘    │                                _summary  │
                         └──────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                       │
            ┌───────▼──────┐   ┌───────────▼──────┐   ┌──────────▼─────┐
            │ RFM Analysis │   │ Churn Prediction  │   │Revenue Forecast│
            │              │   │                   │   │                │
            │ Quintile     │   │ XGBoost + Platt   │   │ ETS / SARIMAX  │
            │ scoring      │   │ calibration       │   │ 30/60/90-day   │
            │ 10 segments  │   │ SHAP explainer    │   │ CI bands       │
            └───────┬──────┘   └──────────┬────────┘   └──────────┬─────┘
                    │                      │                       │
                    └──────────────────────▼───────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Streamlit Dashboard    │
                              │   (Dark Theme · Plotly)  │
                              │                          │
                              │  Executive Overview      │
                              │  Customer Segments       │
                              │  Cohort Retention        │
                              │  Churn Intelligence      │
                              │  Revenue Forecast        │
                              └──────────────────────────┘
```

---

## Features

| Module | Description |
|--------|-------------|
| **Synthetic Data** | 750 customers · 3 yrs history · realistic churn/upgrade curves · 700K+ product events |
| **DBT-style SQL** | 9 models across staging → intermediate → mart layers, run via DuckDB |
| **RFM Segmentation** | Quintile scoring, 10 business segments (Champions → Lost), CS outreach queue |
| **Cohort Retention** | Monthly heatmap · average retention curve · SaaS benchmark comparison |
| **Churn Prediction** | XGBoost with Platt calibration · SHAP feature importance · risk tiers |
| **Revenue Forecast** | Holt-Winters ETS · 30/60/90-day projections · optimistic/pessimistic scenarios |
| **Dashboard** | 5-page Streamlit app · dark theme · Plotly charts · KPI cards |

---

## Project Structure

```
revenue-intelligence-platform/
├── data/
│   ├── raw/               ← Generated CSVs + Parquet files
│   ├── staging/           ← Staged model outputs
│   ├── intermediate/      ← Intermediate model outputs
│   └── mart/              ← Final analytical tables + churn_model.pkl
│
├── src/
│   ├── config.py                       ← Central config (paths, plans, colours)
│   ├── data_generation/
│   │   └── synthetic_data.py           ← SaaS dataset generator
│   ├── models/
│   │   ├── database.py                 ← DuckDB runner (DBT-style)
│   │   ├── staging/                    ← 4 SQL staging models
│   │   ├── intermediate/               ← 2 SQL intermediate models
│   │   └── mart/                       ← 3 SQL mart models
│   ├── analytics/
│   │   ├── rfm_analysis.py             ← RFM scoring & segmentation
│   │   ├── cohort_analysis.py          ← Cohort retention matrix
│   │   ├── churn_prediction.py         ← ML churn model
│   │   └── revenue_forecast.py         ← ETS revenue forecaster
│   └── dashboard/
│       ├── app.py                      ← Streamlit multi-page app
│       └── components/
│           └── styles.py               ← CSS, KPI card helpers
│
├── tests/
│   ├── test_data_generation.py         ← 20 data generation tests
│   └── test_analytics.py               ← 27 analytics tests
│
├── .streamlit/config.toml              ← Dark theme config
├── run.py                              ← Full pipeline runner
├── Makefile                            ← Convenience commands
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run.py
```

This runs all four stages in sequence:
- **Stage 1** — Generate 750 synthetic customers + all related data (~6s)
- **Stage 2** — Build all 9 SQL models via DuckDB (~1s)
- **Stage 3** — Train XGBoost churn model with CV evaluation (~45s)
- **Stage 4** — Fit ETS revenue forecaster + print 30/60/90-day projections

### 3. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### One-liner (pipeline + dashboard)

```bash
python run.py --dashboard
```

---

## Pipeline Options

```
python run.py [OPTIONS]

Options:
  --skip-data      Skip data generation (reuse existing raw files)
  --skip-models    Skip SQL model build
  --skip-ml        Skip ML model training
  --dashboard      Launch Streamlit after pipeline completes
  --customers N    Number of synthetic customers to generate (default: 750)
```

Examples:
```bash
# Rebuild SQL models + ML only (data already generated)
python run.py --skip-data

# Generate fresh data with 1000 customers
python run.py --customers 1000

# Rebuild everything and open dashboard
python run.py --dashboard
```

---

## Running Tests

```bash
# All 47 tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## SQL Data Models

Models follow the **staging → intermediate → mart** pattern used by dbt.

### Staging layer — raw data cleaning

| Model | Source | Description |
|-------|--------|-------------|
| `stg_customers` | raw_customers | Type casting, derived fields (cohort, size segment) |
| `stg_subscriptions` | raw_subscriptions | Duration, status flags, MRR/ARR |
| `stg_transactions` | raw_transactions | Revenue recognition, type flags |
| `stg_product_events` | raw_product_events | Event depth classification, engagement weights |

### Intermediate layer — business logic

| Model | Description |
|-------|-------------|
| `int_customer_metrics` | One row per customer: lifetime value, engagement scores, payment health, days-since-* signals |
| `int_monthly_revenue` | Monthly MRR waterfall: new / expansion / contraction / churn / net-new |

### Mart layer — analytics-ready

| Model | Description |
|-------|-------------|
| `mart_customer_360` | Full customer profile with RFM scores, health tier, LTV tier |
| `mart_cohort_analysis` | Retention matrix: cohort × period × retention_rate |
| `mart_revenue_summary` | Monthly revenue with plan breakdown, quick ratio, 3-mo rolling avg |

---

## Analytics Modules

### RFM Segmentation (`src/analytics/rfm_analysis.py`)

Customers are scored on three dimensions using quintiles (1–5):
- **Recency** — days since last product event (lower = better)
- **Frequency** — total successful transactions
- **Monetary** — lifetime revenue

The combined score maps to 10 named segments: Champions, Loyal Customers, Potential Loyalists, New Customers, Promising, At Risk, Needs Attention, Can't Lose Them, Hibernating, Lost.

### Cohort Retention (`src/analytics/cohort_analysis.py`)

- Monthly cohort retention heatmap (up to 24-month lookback)
- Average retention curve with percentile bands
- Industry benchmark comparison (Best-in-Class / Good / Average)
- Cohort size trend to visualise sales velocity

### Churn Prediction (`src/analytics/churn_prediction.py`)

- **Model**: `XGBoostClassifier` wrapped in `CalibratedClassifierCV` (Platt scaling)
- **Features**: 22 behavioural + account features (engagement, payment health, plan, tenure, etc.)
- **Validation**: Stratified 5-fold cross-validation
- **Output**: calibrated probability, risk tier (High / Medium / Low), SHAP top-5 risk factors
- **Persistence**: model saved to `data/mart/churn_model.pkl`

### Revenue Forecast (`src/analytics/revenue_forecast.py`)

- **Primary model**: Holt-Winters ETS (additive trend + seasonality)
- **Fallback**: SARIMAX → linear trend (handles short series gracefully)
- **Output**: daily interpolated forecast with 90% confidence band, optimistic/pessimistic scenarios
- **Horizons**: 30-day, 60-day, 90-day summary with % change vs current

---

## Dashboard Pages

| Page | Key Visuals |
|------|-------------|
| **Executive Overview** | MRR trend, waterfall chart, plan pie, customer growth, AI insights |
| **Customer Segments** | RFM bubble chart, segment treemap, industry bar, at-risk account table |
| **Cohort Retention** | Retention heatmap, benchmark curve, cohort size trend |
| **Churn Intelligence** | Risk distribution, risk-by-plan stack, MRR at risk scatter, high-risk table |
| **Revenue Forecast** | 90-day chart with CI + scenario bands, horizon summary table, MoM growth |

---

## Screenshots

> _Run the platform and take screenshots at each dashboard page._

| Page | Preview |
|------|---------|
| Executive Overview | `assets/screenshots/01_executive_overview.png` |
| Customer Segments | `assets/screenshots/02_customer_segments.png` |
| Cohort Retention | `assets/screenshots/03_cohort_retention.png` |
| Churn Intelligence | `assets/screenshots/04_churn_intelligence.png` |
| Revenue Forecast | `assets/screenshots/05_revenue_forecast.png` |

---

## Key Metrics (Sample Run)

```
Customers:     750 total  |  517 active  |  233 churned (31%)
Current MRR:   $798,966   |  ARR: $9.6M
Avg ARPU:      ~$1,547/mo
30-day MRR:    $839,832   (+5.1%)
90-day MRR:    $899,392   (+12.6%)
Model ROC-AUC: 1.000 (synthetic data is fully separable by design)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | Python · Faker · NumPy |
| SQL analytics | DuckDB (in-process, no server required) |
| ML | XGBoost · scikit-learn · SHAP |
| Forecasting | statsmodels (ETS / SARIMAX) |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Testing | pytest |

---

## Makefile Commands

```bash
make install      # pip install -r requirements.txt
make pipeline     # python run.py (full pipeline)
make pipeline-fast # python run.py --skip-data
make dashboard    # streamlit run src/dashboard/app.py
make test         # pytest tests/ -v
make test-cov     # pytest with coverage
make clean        # remove __pycache__, .pyc, DuckDB file, model pickle
make clean-data   # remove all generated parquet/CSV files
```

---

## Extending the Platform

**Add a new SQL model**: drop a `.sql` file into `src/models/{layer}/`. Prefix with a number to control execution order (e.g., `04_int_new_model.sql`). Use `{{ref('table_name')}}` to reference upstream models.

**Add a new dashboard page**: add a new `elif page == "..."` branch in `src/dashboard/app.py` and add the label to the sidebar `st.radio`.

**Swap in a real database**: replace the DuckDB connection in `src/models/database.py` with a `psycopg2`/`snowflake-connector` connection while keeping all SQL files unchanged.

---

## License

MIT
