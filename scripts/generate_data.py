#!/usr/bin/env python3
"""
Generate synthetic transaction dataset and save to data/transactions.parquet

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n 500000 --fraud-rate 0.015 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_transactions


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--n", type=int, default=200_000, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.015, help="Fraud rate (0-1)")
    parser.add_argument("--users", type=int, default=10_000, help="Number of unique users")
    parser.add_argument("--merchants", type=int, default=5_000, help="Number of unique merchants")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/transactions.parquet",
        help="Output path (.parquet)",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(
        n_transactions=args.n,
        fraud_rate=args.fraud_rate,
        n_users=args.users,
        n_merchants=args.merchants,
        random_seed=args.seed,
    )

    df.to_parquet(args.output, index=False)
    print(f"\nSaved {len(df):,} transactions → {args.output}")
    print(f"Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Columns: {list(df.columns)}")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
