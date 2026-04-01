#!/usr/bin/env python3
"""
Evaluate a trained model on a dataset and print a detailed report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --data ./data/transactions.parquet --threshold 0.45
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud model")
    parser.add_argument("--data", type=str, default="./data/transactions.parquet")
    parser.add_argument("--model", type=str, default="./models_store/ensemble_model.joblib")
    parser.add_argument("--pipeline", type=str, default="./models_store/feature_pipeline.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample", type=int, default=50_000, help="Sample rows for speed")
    args = parser.parse_args()

    from src.models.ensemble import FraudEnsemble

    console.print("[cyan]Loading model and data...[/cyan]")
    model = FraudEnsemble.load(args.model)
    transformer = joblib.load(args.pipeline)

    df = pd.read_parquet(args.data)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)
    console.print(f"Evaluating on {len(df):,} transactions (fraud rate: {df['is_fraud'].mean()*100:.2f}%)")

    X = transformer.transform(df)
    y = df["is_fraud"].values
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    table = Table(title=f"Evaluation (threshold={args.threshold})", style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("ROC-AUC", f"{roc_auc_score(y, proba):.4f}")
    table.add_row("Avg Precision (AUCPR)", f"{average_precision_score(y, proba):.4f}")
    table.add_row("F1 Score", f"{f1_score(y, preds, zero_division=0):.4f}")
    table.add_row("Precision", f"{tp / max(1, tp+fp):.4f}")
    table.add_row("Recall", f"{tp / max(1, tp+fn):.4f}")
    table.add_row("False Positive Rate", f"{fp / max(1, fp+tn):.4f}")
    table.add_row("True Positives", str(tp))
    table.add_row("False Positives", str(fp))
    table.add_row("False Negatives", str(fn))
    table.add_row("True Negatives", str(tn))

    console.print(table)

    # Threshold sweep
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sweep_table = Table(title="Threshold Sweep", style="yellow")
    sweep_table.add_column("Threshold")
    sweep_table.add_column("Precision", justify="right")
    sweep_table.add_column("Recall", justify="right")
    sweep_table.add_column("F1", justify="right")
    sweep_table.add_column("Fraud Caught", justify="right")

    for t in thresholds:
        p = (proba >= t).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y, p).ravel()
        prec = _tp / max(1, _tp + _fp)
        rec = _tp / max(1, _tp + _fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        sweep_table.add_row(
            f"{t:.2f}",
            f"{prec:.3f}",
            f"{rec:.3f}",
            f"{f1:.3f}",
            f"{_tp}/{y.sum()} ({_tp/max(1,y.sum())*100:.1f}%)",
        )
    console.print(sweep_table)


if __name__ == "__main__":
    main()
