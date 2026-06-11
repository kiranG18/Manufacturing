# src/models/train_per_process.py
#
# Training entry point for per-process cost decomposition models.
# Fits one XGBoost regressor per process code, one per cost component.
# All targets are log-transformed before fitting to handle skewed cost distributions.
#
# Run from repo root:
#   python -m src.models.train_per_process
#   python -m src.models.train_per_process --data data/processed/training_data_v2.csv \
#                                           --output outputs/models/

import argparse
import math
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import xgboost as xgb

from ..models.cost_regressor import COST_COMPONENTS


# Features consumed by each per-process model.
# Geometry features + process context + material pricing.
COST_FEATURE_COLUMNS = [
    "bounding_box_x", "bounding_box_y", "bounding_box_z",
    "volume", "surface_area",
    "wall_thickness_min", "wall_thickness_avg",
    "aspect_ratio", "hole_count", "hole_diameter_min",
    "undercut_flag", "thin_wall_flag",
    "curvature_complexity", "symmetry_score",
    "quantity", "log_quantity", "batch_size",
    "material_code_encoded", "material_price_per_kg",
    "surface_finish_ra", "tolerance_index",
    "volume_x_price",
]

# Hyperparameters tuned via grid search — one set for all component models.
# Per-process hyperparameter overrides are in configs/hyperparams.yaml.
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.06,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 4,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # Exclude estimated labels from evaluation (label quality is lower)
    if "cost_label_source" in df.columns:
        df = df[df["cost_label_source"] != "estimated"].copy()
    print(f"Loaded {len(df)} rows (estimated labels excluded)")
    return df


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE, ignoring near-zero true values that would produce huge percentages."""
    mask = y_true > 1.0  # ignore sub-dollar costs
    if mask.sum() == 0:
        return float("nan")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask])) * 100


def train_process_models(
    df: pd.DataFrame,
    process_code: str,
    output_dir: str,
    params: dict = None,
):
    """Train one model per cost component for a given process."""
    df_proc = df[df["process_code"] == process_code].copy()
    n = len(df_proc)
    if n < 20:
        print(f"  Skipping {process_code}: only {n} samples (need ≥ 20)")
        return None

    print(f"\n  {process_code} (n={n})")
    params = params or DEFAULT_PARAMS

    # Features
    available_cols = [c for c in COST_FEATURE_COLUMNS if c in df_proc.columns]
    X = df_proc[available_cols].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    component_models = {
        "_feature_order": available_cols,
        "_n_training_samples": n,
        "_imputer": imputer,
    }

    for component in COST_COMPONENTS:
        cost_col = f"cost_{component.replace('_cost', '')}"
        if cost_col not in df_proc.columns:
            component = component
            # try alternate column naming
            cost_col = component
        if cost_col not in df_proc.columns:
            continue

        y_raw = df_proc[cost_col].values.astype(float)
        # Remove rows where cost is null or implausibly low
        valid = (y_raw > 0.01) & ~np.isnan(y_raw)
        if valid.sum() < 10:
            continue

        y_log = np.log1p(y_raw[valid])
        X_valid = X_imp[valid]

        reg = xgb.XGBRegressor(**params)
        reg.fit(X_valid, y_log)

        # Quick train-set MAPE as sanity check
        y_pred = np.expm1(reg.predict(X_valid))
        mape = compute_mape(y_raw[valid], y_pred)
        print(f"    {component:<22} train MAPE: {mape:5.1f}%  (n={valid.sum()})")

        component_models[component] = reg

    # Serialize
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{process_code}_v2.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(component_models, f)

    return component_models


def train(data_path: str, output_dir: str):
    print("=" * 65)
    print("Cost Decomposition Models — Per-Process Training")
    print("=" * 65)
    t_start = time.time()

    import yaml
    
    hyperparams = {}
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "hyperparams.yaml")
    if not os.path.exists(config_path):
        config_path = "configs/hyperparams.yaml"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                hyperparams = yaml.safe_load(f) or {}
            print(f"Loaded hyperparameters from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load hyperparams config: {e}")

    default_cfg = hyperparams.get("default", DEFAULT_PARAMS)
    overrides_cfg = hyperparams.get("overrides", {})

    df = load_data(data_path)
    processes = df["process_code"].unique().tolist()
    print(f"\nTraining models for {len(processes)} processes:")

    results = {}
    for proc in sorted(processes):
        proc_params = default_cfg.copy()
        if proc in overrides_cfg:
            proc_params.update(overrides_cfg[proc])
            print(f"  Applying parameter overrides for {proc}")
        result = train_process_models(df, proc, output_dir, params=proc_params)
        if result:
            results[proc] = result

    elapsed = time.time() - t_start
    print(f"\nDone. Trained {len(results)} process models in {elapsed:.1f}s")
    print(f"Artifacts in: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/training_data_v2.csv")
    parser.add_argument("--output", default="outputs/models/")
    args = parser.parse_args()
    train(args.data, args.output)
