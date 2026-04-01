"""
Synthetic Financial Transaction Dataset Generator

Generates realistic transaction data with embedded fraud patterns:
- Card testing attacks (small sequential transactions)
- Account takeover (geo-velocity violations)
- CNP fraud (card-not-present patterns)
- Bust-out fraud (sudden spend spike before default)
- Money mule transactions (unusual recipient patterns)
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MERCHANT_CATEGORIES = {
    "grocery": 0.20,
    "gas_station": 0.08,
    "restaurant": 0.12,
    "retail": 0.15,
    "online_retail": 0.10,
    "travel": 0.05,
    "entertainment": 0.06,
    "healthcare": 0.04,
    "utility": 0.05,
    "atm": 0.07,
    "jewelry": 0.01,
    "electronics": 0.04,
    "transfer": 0.03,
}

COUNTRIES = {
    "US": 0.70, "CA": 0.05, "GB": 0.04, "DE": 0.03,
    "FR": 0.02, "AU": 0.02, "NG": 0.01, "RU": 0.01,
    "CN": 0.02, "BR": 0.02, "MX": 0.02, "OTHER": 0.06,
}

HIGH_RISK_COUNTRIES = {"NG", "RU", "OTHER"}

HOUR_WEIGHTS = [
    0.3, 0.2, 0.15, 0.12, 0.1, 0.15,  # 00-05
    0.3, 0.6, 0.9, 1.0, 1.0, 1.0,       # 06-11
    1.0, 1.0, 1.0, 1.0, 1.0, 1.1,       # 12-17
    1.2, 1.1, 1.0, 0.9, 0.7, 0.5,       # 18-23
]


@dataclass
class UserProfile:
    user_id: str
    home_country: str
    avg_txn_amount: float
    std_txn_amount: float
    preferred_categories: list[str]
    credit_limit: float
    account_age_days: int
    risk_score: float  # 0-1, intrinsic fraud likelihood
    is_compromised: bool = False
    compromise_date: Optional[datetime] = None


@dataclass
class MerchantProfile:
    merchant_id: str
    category: str
    country: str
    avg_ticket: float
    is_high_risk: bool = False


def _weighted_choice(options: dict) -> str:
    keys = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]


def generate_users(n: int = 10_000) -> list[UserProfile]:
    users = []
    categories = list(MERCHANT_CATEGORIES.keys())
    for _ in range(n):
        home = _weighted_choice(COUNTRIES)
        avg_amt = np.random.lognormal(mean=3.5, sigma=1.0)
        users.append(UserProfile(
            user_id=str(uuid.uuid4()),
            home_country=home,
            avg_txn_amount=avg_amt,
            std_txn_amount=avg_amt * np.random.uniform(0.3, 0.8),
            preferred_categories=random.sample(categories, k=random.randint(3, 7)),
            credit_limit=round(random.uniform(1_000, 50_000), -2),
            account_age_days=random.randint(30, 3650),
            risk_score=np.random.beta(1.5, 15),
        ))
    return users


def generate_merchants(n: int = 5_000) -> list[MerchantProfile]:
    merchants = []
    for _ in range(n):
        cat = _weighted_choice(MERCHANT_CATEGORIES)
        country = _weighted_choice(COUNTRIES)
        merchants.append(MerchantProfile(
            merchant_id=str(uuid.uuid4()),
            category=cat,
            country=country,
            avg_ticket=np.random.lognormal(
                mean=3.0 if cat not in ("jewelry", "electronics", "travel") else 5.0,
                sigma=0.8,
            ),
            is_high_risk=(cat in ("jewelry", "electronics", "atm", "transfer")
                          or country in HIGH_RISK_COUNTRIES),
        ))
    return merchants


# ---------------------------------------------------------------------------
# Fraud pattern injectors
# ---------------------------------------------------------------------------

def _card_testing_burst(
    user: UserProfile,
    merchants: list[MerchantProfile],
    anchor_time: datetime,
) -> list[dict]:
    """10-30 small transactions in rapid succession to test stolen card."""
    txns = []
    n = random.randint(10, 30)
    for i in range(n):
        m = random.choice(merchants)
        txns.append({
            "amount": round(random.uniform(0.50, 5.00), 2),
            "merchant_id": m.merchant_id,
            "merchant_category": m.category,
            "merchant_country": m.country,
            "timestamp": anchor_time + timedelta(seconds=i * random.randint(5, 60)),
            "is_online": True,
            "device_fingerprint": hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16],
        })
    return txns


def _geo_velocity_attack(
    user: UserProfile,
    merchants: list[MerchantProfile],
    anchor_time: datetime,
) -> list[dict]:
    """Transactions in geographically impossible locations within minutes."""
    foreign_country = random.choice([c for c in COUNTRIES if c != user.home_country])
    txns = []
    for i in range(random.randint(2, 5)):
        m = random.choice([m for m in merchants if m.country == foreign_country] or merchants)
        txns.append({
            "amount": round(np.random.lognormal(5.0, 0.5), 2),
            "merchant_id": m.merchant_id,
            "merchant_category": m.category,
            "merchant_country": foreign_country,
            "timestamp": anchor_time + timedelta(minutes=i * 15),
            "is_online": False,
            "device_fingerprint": hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16],
        })
    return txns


def _bust_out_spend(
    user: UserProfile,
    merchants: list[MerchantProfile],
    anchor_time: datetime,
) -> list[dict]:
    """Sudden large transactions approaching credit limit."""
    txns = []
    target_spend = user.credit_limit * random.uniform(0.80, 0.99)
    remaining = target_spend
    for i in range(random.randint(3, 8)):
        chunk = min(remaining, random.uniform(target_spend * 0.1, target_spend * 0.4))
        remaining -= chunk
        m = random.choice([m for m in merchants if m.category in ("jewelry", "electronics", "travel")] or merchants)
        txns.append({
            "amount": round(chunk, 2),
            "merchant_id": m.merchant_id,
            "merchant_category": m.category,
            "merchant_country": m.country,
            "timestamp": anchor_time + timedelta(hours=i * 2),
            "is_online": random.random() > 0.5,
            "device_fingerprint": hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16],
        })
        if remaining <= 0:
            break
    return txns


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_transactions(
    n_transactions: int = 500_000,
    fraud_rate: float = 0.015,
    n_users: int = 10_000,
    n_merchants: int = 5_000,
    start_date: datetime = datetime(2023, 1, 1),
    end_date: datetime = datetime(2024, 1, 1),
    random_seed: int = 42,
) -> pd.DataFrame:
    random.seed(random_seed)
    np.random.seed(random_seed)
    Faker.seed(random_seed)

    print("Generating users and merchants...")
    users = generate_users(n_users)
    merchants = generate_merchants(n_merchants)
    merchant_by_category: dict[str, list] = {}
    for m in merchants:
        merchant_by_category.setdefault(m.category, []).append(m)

    date_range_seconds = int((end_date - start_date).total_seconds())
    records = []
    fraud_txns_target = int(n_transactions * fraud_rate)

    # Mark some users as compromised
    compromised_users = random.sample(users, k=max(1, int(n_users * 0.02)))
    for u in compromised_users:
        u.is_compromised = True
        u.compromise_date = start_date + timedelta(
            seconds=random.randint(0, date_range_seconds)
        )

    print(f"Generating {n_transactions:,} transactions ({fraud_rate*100:.1f}% fraud rate)...")

    fraud_injected = 0
    pbar = tqdm(total=n_transactions)

    # Pre-generate fraud burst records from compromised users
    fraud_records: list[dict] = []
    for u in compromised_users:
        if u.compromise_date is None:
            continue
        pattern = random.choice(["card_testing", "geo_velocity", "bust_out"])
        if pattern == "card_testing":
            burst = _card_testing_burst(u, merchants, u.compromise_date)
        elif pattern == "geo_velocity":
            burst = _geo_velocity_attack(u, merchants, u.compromise_date)
        else:
            burst = _bust_out_spend(u, merchants, u.compromise_date)

        for t in burst:
            t.update({
                "transaction_id": str(uuid.uuid4()),
                "user_id": u.user_id,
                "is_fraud": 1,
                "fraud_type": pattern,
            })
            fraud_records.append(t)

    # Pad fraud to target rate with random injection
    while len(fraud_records) < fraud_txns_target:
        u = random.choice(users)
        m = random.choice(merchants)
        ts = start_date + timedelta(seconds=random.randint(0, date_range_seconds))
        fraud_records.append({
            "transaction_id": str(uuid.uuid4()),
            "user_id": u.user_id,
            "merchant_id": m.merchant_id,
            "merchant_category": m.category,
            "merchant_country": m.country,
            "amount": round(np.random.lognormal(5.5, 1.2), 2),
            "timestamp": ts,
            "is_online": random.random() > 0.3,
            "device_fingerprint": hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16],
            "is_fraud": 1,
            "fraud_type": "misc_fraud",
        })

    fraud_set = set(random.sample(range(n_transactions), k=min(len(fraud_records), fraud_txns_target)))
    fraud_iter = iter(fraud_records)

    for i in range(n_transactions):
        if i in fraud_set:
            try:
                rec = next(fraud_iter)
            except StopIteration:
                fraud_set.discard(i)
                rec = None
        else:
            rec = None

        if rec is None:
            u = random.choice(users)
            cat = random.choice(u.preferred_categories)
            cat_merchants = merchant_by_category.get(cat, merchants)
            m = random.choice(cat_merchants)
            ts = start_date + timedelta(seconds=random.randint(0, date_range_seconds))
            hour = ts.hour
            # Apply hour-of-day weight for legit traffic
            if random.random() > HOUR_WEIGHTS[hour] / 1.2:
                ts = ts.replace(hour=random.randint(9, 20))

            amt = max(0.01, np.random.normal(u.avg_txn_amount, u.std_txn_amount))
            rec = {
                "transaction_id": str(uuid.uuid4()),
                "user_id": u.user_id,
                "merchant_id": m.merchant_id,
                "merchant_category": m.category,
                "merchant_country": m.country,
                "amount": round(amt, 2),
                "timestamp": ts,
                "is_online": m.category == "online_retail" or random.random() < 0.25,
                "device_fingerprint": hashlib.md5((u.user_id + str(ts.date())).encode()).hexdigest()[:16],
                "is_fraud": 0,
                "fraud_type": None,
            }

        # Ensure required fields
        rec.setdefault("transaction_id", str(uuid.uuid4()))
        rec.setdefault("merchant_id", random.choice(merchants).merchant_id)

        # Enrich with user profile
        u_lookup = next((u for u in users if u.user_id == rec["user_id"]), random.choice(users))
        rec["user_home_country"] = u_lookup.home_country
        rec["credit_limit"] = u_lookup.credit_limit
        rec["account_age_days"] = u_lookup.account_age_days

        records.append(rec)
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add derived raw fields
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour_of_day"].between(22, 6).astype(int)
    df["is_cross_border"] = (df["merchant_country"] != df["user_home_country"]).astype(int)
    df["is_high_risk_country"] = df["merchant_country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    fraud_count = df["is_fraud"].sum()
    print(f"\nDataset generated: {len(df):,} transactions | "
          f"Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    return df


if __name__ == "__main__":
    df = generate_transactions(n_transactions=100_000)
    out = "/Users/ebubeepuna/fraud-detection-engine/data/transactions.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")
    print(df.head())
    print(df["is_fraud"].value_counts())
