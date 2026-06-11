# src/models/train.py
#
# Training entry point for the process classifier.
# Loads features_v2.csv, trains XGBoost with stratified k-fold CV,
# calibrates probabilities with Platt scaling, saves model artifact.
#
# Run from repo root:
#   python -m src.models.train
#   python -m src.models.train --data data/processed/features_v2.csv --output outputs/model_v2.pkl

import argparse
import os
import pickle
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import xgboost as xgb

from ..features.definitions import FEATURE_COLUMNS
from ..models.classifier import PROCESS_CLASSES
from ..evaluation.metrics import top_k_accuracy, per_class_report


def load_data(data_path: str):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    X = df[FEATURE_COLUMNS].values
    label_encoder = LabelEncoder()
    label_encoder.fit(PROCESS_CLASSES)
    y = label_encoder.transform(df["process_code"].values)

    # Report class distribution
    from collections import Counter
    counts = Counter(df["process_code"].tolist())
    print(f"\nClass distribution ({df['process_code'].nunique()} classes):")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(n / max(counts.values()) * 20)
        print(f"  {cls:<30} {n:>4}  {bar}")
    if len(counts) > 8:
        print(f"  ... and {len(counts) - 8} more classes")

    return X, y, label_encoder


def train(data_path: str, output_path: str, seed: int = 42):
    print("=" * 60)
    print("Process Classifier — Training Run")
    print("=" * 60)
    t_start = time.time()

    X, y, label_encoder = load_data(data_path)

    # Impute missing values (e.g. hole_diameter_min when hole_count=0)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # XGBoost with class weighting for imbalanced classes
    xgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": len(PROCESS_CLASSES),
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    }

    base_clf = xgb.XGBClassifier(**xgb_params)

    # Stratified k-fold cross-validation
    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_top1, fold_top3 = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y), 1):
        X_tr, X_val = X_imputed[train_idx], X_imputed[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        base_clf.fit(X_tr, y_tr)
        proba = base_clf.predict_proba(X_val)

        top1 = top_k_accuracy(y_val, proba, k=1)
        top3 = top_k_accuracy(y_val, proba, k=3)
        fold_top1.append(top1)
        fold_top3.append(top3)
        print(f"  Fold {fold}: top-1={top1:.3f}  top-3={top3:.3f}")

    print(f"\nCV results: top-1={np.mean(fold_top1):.3f} ± {np.std(fold_top1):.3f}")
    print(f"            top-3={np.mean(fold_top3):.3f} ± {np.std(fold_top3):.3f}")

    # Final model on full training data, then Platt-calibrate
    print("\nFitting final model on full dataset...")
    base_clf.fit(X_imputed, y)

    print("Calibrating probabilities with Platt scaling (cv=3)...")
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    calibrated.fit(X_imputed, y)

    # Sanity check on full training set (not a held-out metric — just for log)
    train_proba = calibrated.predict_proba(X_imputed)
    train_top3 = top_k_accuracy(y, train_proba, k=3)
    print(f"Training set top-3 accuracy (calibrated): {train_top3:.3f}")

    # Save artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": calibrated,
        "imputer": imputer,
        "calibrated": True,
        "feature_columns": FEATURE_COLUMNS,
        "process_classes": PROCESS_CLASSES,
        "cv_top1_mean": float(np.mean(fold_top1)),
        "cv_top3_mean": float(np.mean(fold_top3)),
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    elapsed = time.time() - t_start
    print(f"\nModel saved to {output_path}  ({elapsed:.1f}s)")
    print("=" * 60)
    return calibrated, imputer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the process classifier")
    parser.add_argument("--data", default="data/processed/features_v2.csv")
    parser.add_argument("--output", default="outputs/model_v2.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.data, args.output, seed=args.seed)
