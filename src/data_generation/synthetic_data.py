"""
Synthetic SaaS dataset generator.

Produces four CSVs that mimic a real B2B SaaS company's operational data:
  - customers.csv
  - subscriptions.csv
  - transactions.csv
  - product_events.csv

Data characteristics are tuned to match typical $10M ARR SaaS metrics:
  ~750 customers, 3-year history, ~$1.1M MRR at peak.
"""
import uuid
import random
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    RAW_DIR, SIM_START, SIM_END, RANDOM_SEED,
    PLANS, PLAN_ORDER, INDUSTRIES, ACQUISITION_CHANNELS,
    COUNTRIES, CSM_NAMES,
)

log = logging.getLogger(__name__)
fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── helpers ──────────────────────────────────────────────────────────────────

def _weighted_choice(options: dict) -> str:
    keys   = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]

def _rand_date(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def _employee_count(plan: str) -> int:
    ranges = {
        "Starter": (1, 20),
        "Growth": (10, 100),
        "Professional": (50, 500),
        "Enterprise": (200, 5000),
    }
    lo, hi = ranges[plan]
    # log-normal so most land in lower range
    return max(lo, min(hi, int(np.random.lognormal(
        np.log((lo + hi) / 2), 0.5
    ))))

def _mrr(plan: str, billing: str, seat_count: int) -> float:
    base = PLANS[plan]["price_monthly"]
    if billing == "annual":
        base = PLANS[plan]["price_annual"] / 12
    # small per-seat multiplier for larger plans
    if plan in ("Professional", "Enterprise"):
        extra = max(0, seat_count - PLANS[plan]["seats_range"][0]) * random.uniform(15, 40)
        base += extra
    return round(base, 2)

# ── customer generation ───────────────────────────────────────────────────────

def generate_customers(n: int = 750) -> pd.DataFrame:
    """
    Generate n customers with realistic attributes.
    Signup dates follow an S-curve (slow start → growth → slight maturity dip).
    """
    log.info("Generating %d customers …", n)

    # weighted signup date distribution – growth acceleration over time
    total_days = (SIM_END - SIM_START).days
    day_weights = np.exp(np.linspace(-1.5, 1.5, total_days))  # exponential growth
    day_weights /= day_weights.sum()
    signup_offsets = np.random.choice(total_days, size=n, replace=True, p=day_weights)

    plan_names   = list(PLANS.keys())
    plan_weights = [PLANS[p]["weight"] for p in plan_names]
    country_keys = list(COUNTRIES.keys())
    country_weights = list(COUNTRIES.values())

    rows = []
    for i, offset in enumerate(signup_offsets):
        signup_dt  = SIM_START + timedelta(days=int(offset))
        plan       = random.choices(plan_names, weights=plan_weights, k=1)[0]
        billing    = random.choices(["monthly", "annual"], weights=[0.65, 0.35], k=1)[0]
        seats      = random.randint(*PLANS[plan]["seats_range"])
        country    = random.choices(country_keys, weights=country_weights, k=1)[0]

        # churn simulation: some fraction of customers leave before SIM_END
        churned_at = None
        monthly_churn = PLANS[plan]["churn_rate"]
        # survival function: probability of still active at each month
        months_alive = relativedelta(SIM_END, signup_dt).months + \
                       relativedelta(SIM_END, signup_dt).years * 12
        # draw time-to-churn from geometric distribution
        if random.random() < (1 - (1 - monthly_churn) ** months_alive):
            churn_month = np.random.geometric(monthly_churn)
            churn_date  = signup_dt + relativedelta(months=min(churn_month, months_alive))
            if churn_date < SIM_END:
                churned_at = churn_date

        rows.append({
            "customer_id":     str(uuid.uuid4()),
            "company_name":    fake.company(),
            "industry":        random.choice(INDUSTRIES),
            "country":         country,
            "plan":            plan,
            "billing_cycle":   billing,
            "seats":           seats,
            "employee_count":  _employee_count(plan),
            "signup_date":     signup_dt.date(),
            "churned_at":      churned_at.date() if churned_at else None,
            "csm":             random.choice(CSM_NAMES),
            "acquisition_channel": random.choice(ACQUISITION_CHANNELS),
            "health_score":    round(random.gauss(72, 18), 1),   # 0-100
            "nps_score":       random.choice([None, None, random.randint(-100, 100)]),
        })

    df = pd.DataFrame(rows)
    # clip health score
    df["health_score"] = df["health_score"].clip(0, 100)
    df["is_churned"]   = df["churned_at"].notna().astype(int)
    log.info("  Active: %d  |  Churned: %d", (~df["is_churned"].astype(bool)).sum(),
             df["is_churned"].sum())
    return df

# ── subscription generation ───────────────────────────────────────────────────

def generate_subscriptions(customers: pd.DataFrame) -> pd.DataFrame:
    """
    One subscription record per customer (initial).  Some customers have
    upgrade events that produce additional subscription records.
    """
    log.info("Generating subscriptions …")
    rows = []
    for _, c in customers.iterrows():
        sub_start = c["signup_date"]
        plan      = c["plan"]
        billing   = c["billing_cycle"]
        seats     = c["seats"]
        mrr       = _mrr(plan, billing, seats)

        sub_end   = c["churned_at"]

        # potential upsell: ~15% of non-churned customers upgrade mid-life
        upgrades = []
        if not c["is_churned"] and plan != "Enterprise":
            idx = PLAN_ORDER.index(plan)
            if random.random() < 0.18:
                upgrade_month = random.randint(3, 18)
                upgrade_date  = (pd.Timestamp(sub_start) +
                                 relativedelta(months=upgrade_month)).date()
                if upgrade_date < SIM_END.date():
                    new_plan  = PLAN_ORDER[idx + 1]
                    new_mrr   = _mrr(new_plan, billing, seats)
                    upgrades.append((upgrade_date, new_plan, new_mrr))

        # build subscription segments
        sid = str(uuid.uuid4())
        rows.append({
            "subscription_id": sid,
            "customer_id":     c["customer_id"],
            "plan":            plan,
            "billing_cycle":   billing,
            "mrr":             mrr,
            "start_date":      sub_start,
            "end_date":        upgrades[0][0] if upgrades else sub_end,
            "status":          "upgraded" if upgrades else ("churned" if sub_end else "active"),
        })

        for i, (upg_date, upg_plan, upg_mrr) in enumerate(upgrades):
            next_end = upgrades[i + 1][0] if i + 1 < len(upgrades) else sub_end
            rows.append({
                "subscription_id": str(uuid.uuid4()),
                "customer_id":     c["customer_id"],
                "plan":            upg_plan,
                "billing_cycle":   billing,
                "mrr":             upg_mrr,
                "start_date":      upg_date,
                "end_date":        next_end,
                "status":          "active" if not next_end else "churned",
            })

    return pd.DataFrame(rows)

# ── transaction generation ────────────────────────────────────────────────────

def generate_transactions(customers: pd.DataFrame,
                          subscriptions: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly billing transactions, contraction/expansion events, and refunds.
    """
    log.info("Generating transactions …")
    rows = []

    for _, sub in subscriptions.iterrows():
        start = pd.Timestamp(sub["start_date"])
        end   = pd.Timestamp(sub["end_date"]) if sub["end_date"] else pd.Timestamp(SIM_END.date())
        mrr   = sub["mrr"]
        billing = sub["billing_cycle"]

        current = start
        while current <= end:
            # billing event
            if billing == "annual":
                # annual invoice once a year
                if current.month == start.month:
                    amount = mrr * 12
                    tx_type = "annual_subscription"
                else:
                    current += relativedelta(months=1)
                    continue
            else:
                amount  = mrr
                tx_type = "monthly_subscription"

            # occasional payment failure (~2%)
            status = "failed" if random.random() < 0.02 else "success"
            # rare refund on success (~0.3%)
            if status == "success" and random.random() < 0.003:
                status = "refunded"

            rows.append({
                "transaction_id":   str(uuid.uuid4()),
                "customer_id":      sub["customer_id"],
                "subscription_id":  sub["subscription_id"],
                "transaction_date": current.date(),
                "amount":           round(amount, 2),
                "currency":         "USD",
                "transaction_type": tx_type,
                "status":           status,
            })
            current += relativedelta(months=1)

        # churn / cancellation event
        if sub["status"] in ("churned", "upgraded") and sub["end_date"]:
            rows.append({
                "transaction_id":   str(uuid.uuid4()),
                "customer_id":      sub["customer_id"],
                "subscription_id":  sub["subscription_id"],
                "transaction_date": sub["end_date"],
                "amount":           0.0,
                "currency":         "USD",
                "transaction_type": "churn" if sub["status"] == "churned" else "upgrade",
                "status":           "success",
            })

    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    log.info("  %d transactions generated", len(df))
    return df

# ── product events ────────────────────────────────────────────────────────────

_EVENT_TYPES = [
    ("login",              0.30),
    ("dashboard_view",     0.18),
    ("report_generated",   0.12),
    ("api_call",           0.10),
    ("integration_added",  0.04),
    ("export_csv",         0.07),
    ("share_report",       0.05),
    ("support_ticket",     0.06),
    ("feature_flag_used",  0.04),
    ("billing_viewed",     0.04),
]
_EVENT_NAMES   = [e[0] for e in _EVENT_TYPES]
_EVENT_WEIGHTS = [e[1] for e in _EVENT_TYPES]


def generate_product_events(customers: pd.DataFrame,
                            n_events: int = 120_000) -> pd.DataFrame:
    """
    Product usage event log.  Active customers generate more events;
    customers close to churn generate progressively fewer.
    """
    log.info("Generating %d product events …", n_events)
    rows = []

    active_custs = customers[customers["is_churned"] == 0].copy()
    churned_custs = customers[customers["is_churned"] == 1].copy()

    def _emit(cid: str, start: datetime, end: datetime, base_rate: int):
        d = start
        while d < end:
            day_events = max(0, int(np.random.poisson(base_rate / 30)))
            for _ in range(day_events):
                rows.append({
                    "event_id":    str(uuid.uuid4()),
                    "customer_id": cid,
                    "event_type":  random.choices(_EVENT_NAMES, weights=_EVENT_WEIGHTS, k=1)[0],
                    "event_date":  d.date(),
                })
            d += timedelta(days=1)

    # active customers — ongoing usage
    for _, c in active_custs.iterrows():
        base = {"Starter": 40, "Growth": 90, "Professional": 150, "Enterprise": 280}[c["plan"]]
        _emit(c["customer_id"],
              datetime.combine(c["signup_date"], datetime.min.time()),
              SIM_END, base)

    # churned customers — declining usage before churn
    for _, c in churned_custs.iterrows():
        churn_dt = datetime.combine(c["churned_at"], datetime.min.time())
        base = {"Starter": 35, "Growth": 80, "Professional": 130, "Enterprise": 250}[c["plan"]]
        # normal usage until 60 days before churn
        cutoff = churn_dt - timedelta(days=60)
        _emit(c["customer_id"],
              datetime.combine(c["signup_date"], datetime.min.time()),
              cutoff, base)
        # declining phase
        _emit(c["customer_id"], cutoff, churn_dt, max(2, int(base * 0.20)))

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    log.info("  %d events generated", len(df))
    return df

# ── main entry ────────────────────────────────────────────────────────────────

def run(n_customers: int = 750) -> dict[str, pd.DataFrame]:
    """Generate all datasets and write to RAW_DIR as parquet + CSV."""
    customers     = generate_customers(n_customers)
    subscriptions = generate_subscriptions(customers)
    transactions  = generate_transactions(customers, subscriptions)
    events        = generate_product_events(customers)

    datasets = {
        "customers":     customers,
        "subscriptions": subscriptions,
        "transactions":  transactions,
        "product_events": events,
    }

    for name, df in datasets.items():
        csv_path     = RAW_DIR / f"{name}.csv"
        parquet_path = RAW_DIR / f"{name}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        log.info("Saved %s → %s (%d rows)", name, csv_path.name, len(df))

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
