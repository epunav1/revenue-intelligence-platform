"""
Revenue Intelligence Platform — Full Pipeline Runner
=====================================================

Usage
-----
  python run.py               # run full pipeline (data gen + models + ML)
  python run.py --skip-data   # skip data generation (use existing raw files)
  python run.py --skip-models # skip SQL model build
  python run.py --skip-ml     # skip ML model training
  python run.py --dashboard   # launch Streamlit after pipeline
  python run.py --help
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# ── ensure src/ is on path when running from project root ─────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import colorlog

# ── Logging setup ─────────────────────────────────────────────────────────────
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    },
))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)
log = logging.getLogger(__name__)


# ── Pipeline stages ────────────────────────────────────────────────────────────

def stage_data_generation(n_customers: int = 750):
    log.info("=" * 60)
    log.info("STAGE 1 — Synthetic Data Generation")
    log.info("=" * 60)
    from src.data_generation.synthetic_data import run as gen_run
    t0 = time.time()
    datasets = gen_run(n_customers)
    for name, df in datasets.items():
        log.info("  %-18s %d rows", name, len(df))
    log.info("Data generation complete in %.1fs", time.time() - t0)
    return datasets


def stage_sql_models():
    log.info("=" * 60)
    log.info("STAGE 2 — SQL Model Build (DBT-style)")
    log.info("=" * 60)
    from src.models.database import build_all, get_connection
    t0 = time.time()
    conn = build_all()
    log.info("SQL models complete in %.1fs", time.time() - t0)
    return conn


def stage_ml_training(conn=None):
    log.info("=" * 60)
    log.info("STAGE 3 — ML Model Training")
    log.info("=" * 60)
    t0 = time.time()

    import pandas as pd
    from src.config import MART_DIR, INT_DIR

    # Load customer metrics for training
    metrics_path = INT_DIR / "int_customer_metrics.parquet"
    if not metrics_path.exists():
        log.warning("int_customer_metrics.parquet not found — skipping ML training")
        return

    df = pd.read_parquet(metrics_path)
    log.info("  Training data: %d customers  |  Churn rate: %.1f%%",
             len(df), 100 * df["is_churned"].mean())

    from src.analytics.churn_prediction import ChurnPredictor
    predictor = ChurnPredictor()
    metrics   = predictor.train(df)

    log.info("  ┌─────────────────────────────────")
    log.info("  │  ROC-AUC:         %.4f", metrics["roc_auc"])
    log.info("  │  Avg Precision:   %.4f", metrics["avg_precision"])
    log.info("  │  CV ROC-AUC:      %.4f ± %.4f",
             metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"])
    log.info("  └─────────────────────────────────")
    log.info("ML training complete in %.1fs", time.time() - t0)


def stage_revenue_forecast():
    log.info("=" * 60)
    log.info("STAGE 4 — Revenue Forecast")
    log.info("=" * 60)
    t0 = time.time()

    import pandas as pd
    from src.config import MART_DIR
    from src.analytics.revenue_forecast import RevenueForecaster

    rev_path = MART_DIR / "mart_revenue_summary.parquet"
    if not rev_path.exists():
        log.warning("mart_revenue_summary.parquet not found — skipping forecast")
        return

    rev_df = pd.read_parquet(rev_path)
    fc = RevenueForecaster()
    fc.fit(rev_df)
    daily = fc.forecast(horizon_days=90)
    summary = fc.horizon_summary(daily)

    last = summary["last_actual_mrr"]
    log.info("  Current MRR:  $%s  (ARR: $%s)", f"{last:,.0f}", f"{last*12:,.0f}")
    for horizon, label in [("30d", "30-day"), ("60d", "60-day"), ("90d", "90-day")]:
        h = summary[horizon]
        log.info("  %s forecast: $%s  (%+.1f%%)",
                 label, f"{h['forecast']:,.0f}", h["pct_change"])

    log.info("Forecast complete in %.1fs", time.time() - t0)


def launch_dashboard():
    import subprocess, os
    app_path = ROOT / "src" / "dashboard" / "app.py"
    log.info("Launching Streamlit dashboard → %s", app_path)
    os.execvp("streamlit", ["streamlit", "run", str(app_path),
                             "--server.headless", "false"])


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Revenue Intelligence Platform — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-data",   action="store_true", help="Skip data generation")
    parser.add_argument("--skip-models", action="store_true", help="Skip SQL model build")
    parser.add_argument("--skip-ml",     action="store_true", help="Skip ML training")
    parser.add_argument("--dashboard",   action="store_true", help="Launch dashboard after pipeline")
    parser.add_argument("--customers",   type=int, default=750, help="Number of synthetic customers (default: 750)")
    args = parser.parse_args()

    wall_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     Revenue Intelligence Platform — Pipeline             ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    if not args.skip_data:
        stage_data_generation(args.customers)

    conn = None
    if not args.skip_models:
        conn = stage_sql_models()

    if not args.skip_ml:
        stage_ml_training(conn)

    stage_revenue_forecast()

    log.info("=" * 60)
    log.info("Pipeline complete in %.1fs", time.time() - wall_start)
    log.info("Run:  streamlit run src/dashboard/app.py")
    log.info("=" * 60)

    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
