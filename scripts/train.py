#!/usr/bin/env python3
"""
Full training pipeline: generate data → engineer features → tune → train → evaluate

Usage:
    python scripts/train.py
    python scripts/train.py --data ./data/transactions.parquet --trials 50
    python scripts/train.py --generate --n 200000   # also regenerate data
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--data", type=str, default="./data/transactions.parquet")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data before training",
    )
    parser.add_argument("--n", type=int, default=200_000, help="Rows to generate (if --generate)")
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    args = parser.parse_args()

    console.print(Panel.fit("🛡️  Fraud Detection Engine — Training Pipeline", style="bold blue"))

    # Step 1: Generate data if requested or missing
    if args.generate or not Path(args.data).exists():
        console.print("\n[bold cyan]Step 1/4 — Generating synthetic dataset...[/bold cyan]")
        from data.generator import generate_transactions
        df = generate_transactions(n_transactions=args.n, fraud_rate=args.fraud_rate)
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.data, index=False)
        console.print(f"[green]✓ Dataset saved: {args.data} ({len(df):,} rows)[/green]")
    else:
        console.print(f"\n[cyan]Step 1/4 — Using existing dataset: {args.data}[/cyan]")

    # Step 2: Feature engineering (baked into trainer via FraudFeatureTransformer)
    console.print("\n[bold cyan]Step 2/4 — Feature engineering will run inside trainer...[/bold cyan]")
    console.print("[yellow]Note: velocity features require sorted-by-time data (handled automatically)[/yellow]")

    # Step 3: Train
    console.print("\n[bold cyan]Step 3/4 — Hyperparameter tuning + model training...[/bold cyan]")
    t0 = time.time()

    from src.models.trainer import train
    metrics = train(
        data_path=args.data,
        n_trials=args.trials,
        fraud_threshold=args.threshold,
    )

    elapsed = time.time() - t0
    console.print(f"[green]✓ Training complete in {elapsed:.1f}s[/green]")

    # Step 4: Print results
    console.print("\n[bold cyan]Step 4/4 — Test Set Results[/bold cyan]")
    table = Table(title="Model Performance", style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    metric_display = {
        "roc_auc": ("ROC-AUC", ".4f"),
        "avg_precision": ("Avg Precision (AUCPR)", ".4f"),
        "f1": ("F1 Score", ".4f"),
        "precision": ("Precision", ".4f"),
        "recall": ("Recall", ".4f"),
        "tp": ("True Positives", "d"),
        "fp": ("False Positives", "d"),
        "fn": ("False Negatives", "d"),
        "tn": ("True Negatives", "d"),
    }
    for key, (label, fmt) in metric_display.items():
        if key in metrics:
            val = metrics[key]
            formatted = f"{val:{fmt}}" if isinstance(val, float) else str(val)
            table.add_row(label, formatted)

    console.print(table)
    console.print(f"\n[bold green]Artefacts saved to ./models_store/[/bold green]")
    console.print("  • ensemble_model.joblib")
    console.print("  • feature_pipeline.joblib")
    console.print("  • best_params.json")
    console.print("  • metrics.json")
    console.print("  • reference_data.parquet")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  uvicorn src.api.main:app --reload   # Start API")
    console.print("  streamlit run dashboard/app.py      # Start dashboard")


if __name__ == "__main__":
    main()
