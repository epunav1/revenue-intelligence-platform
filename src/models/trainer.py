"""
Model Trainer with Optuna Hyperparameter Tuning

Workflow:
1. Load feature-engineered dataset
2. Stratified train/val/test split
3. Optuna study optimizes XGB + LGB hyperparams jointly
4. Train final ensemble on train+val
5. Evaluate on held-out test set
6. Save model artefacts
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb

from src.features.engineering import FraudFeatureTransformer, FEATURE_COLS
from src.models.ensemble import FraudEnsemble

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path("./models_store")
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Objective function for Optuna
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, X_tr, y_tr, X_val, y_val) -> float:
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 800),
        "max_depth": trial.suggest_int("xgb_max_depth", 4, 9),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("xgb_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
    }
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_lambda", 1e-3, 10.0, log=True),
    }
    xgb_weight = trial.suggest_float("xgb_weight", 0.3, 0.7)

    ensemble = FraudEnsemble(
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        xgb_weight=xgb_weight,
    )
    ensemble.fit(X_tr, y_tr)
    proba = ensemble.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path: str = "./data/transactions.parquet",
    n_trials: int = 50,
    test_size: float = 0.15,
    val_size: float = 0.15,
    fraud_threshold: float = 0.5,
) -> dict:
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows. Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    # Feature transformer
    transformer = FraudFeatureTransformer()

    # Split before fitting transformer (prevent leakage)
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, stratify=df["is_fraud"], random_state=42
    )
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size / (1 - test_size),
        stratify=df_trainval["is_fraud"], random_state=42
    )

    print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    # Fit transformer on train only
    print("Fitting feature transformer...")
    transformer.fit(df_train)
    joblib.dump(transformer, MODELS_DIR / "feature_pipeline.joblib")
    print(f"Feature transformer saved.")

    X_train = transformer.transform(df_train)
    X_val = transformer.transform(df_val)
    X_test = transformer.transform(df_test)
    y_train = df_train["is_fraud"].values
    y_val = df_val["is_fraud"].values
    y_test = df_test["is_fraud"].values
    feature_names = transformer.get_feature_names_out()

    # Optuna hyperparameter search
    print(f"\nRunning Optuna hyperparameter search ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        study_name="fraud_ensemble",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    print(f"Best AUCPR: {study.best_value:.4f}")

    # Save best params
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({"best_params": best_params, "best_aucpr": study.best_value}, f, indent=2)

    # Train final model on train+val
    print("\nTraining final ensemble on train+val...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_") and k != "xgb_weight"}
    lgb_params = {k.replace("lgb_", ""): v for k, v in best_params.items() if k.startswith("lgb_")}
    xgb_weight = best_params.get("xgb_weight", 0.5)

    final_model = FraudEnsemble(
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        xgb_weight=xgb_weight,
    )
    final_model.fit(X_trainval, y_trainval, feature_names=feature_names)
    final_model.save(MODELS_DIR / "ensemble_model.joblib")

    # Evaluate on test set
    print("\nEvaluating on held-out test set...")
    metrics = final_model.evaluate(X_test, y_test, threshold=fraud_threshold)

    print("\n" + "=" * 50)
    print("TEST SET PERFORMANCE")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")

    # Save metrics
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reference data for drift monitoring
    ref_sample = df_test.sample(min(5000, len(df_test)), random_state=42)
    ref_sample.to_parquet(MODELS_DIR / "reference_data.parquet", index=False)
    print(f"\nReference data saved for drift monitoring.")

    return metrics


if __name__ == "__main__":
    metrics = train(n_trials=30)
