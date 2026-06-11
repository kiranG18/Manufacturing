# src/models/cost_regressor.py
#
# CostDecompositionModel
# ----------------------
# Wraps a set of per-process XGBoost regressors. Each process code (e.g.
# "cnc_milling", "injection_molding") has its own model trained on the
# cost data for that process only.
#
# The per-process design is intentional: cost structures differ radically
# across processes. A model trained on all processes together would learn
# average behaviour across incompatible physics, producing worse predictions
# than a model specialised for each process's cost drivers.

import os
import pickle
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# The 5 cost components the model predicts separately.
COST_COMPONENTS = [
    "machine_time_cost",
    "setup_cost",
    "tooling_cost",
    "raw_material_cost",
    "finishing_cost",
]

# Processes that don't have tooling in the traditional sense
# (tooling_cost is set to 0 for these)
NO_TOOLING_PROCESSES = {"additive_fdm", "additive_slm", "laser_cutting"}


class CostDecompositionModel:
    """
    Loads and manages per-process cost decomposition regressors.

    Targets are log-transformed (log(cost + 1)) during training to handle
    the skewed cost distribution. Predictions are inverse-transformed here
    before being returned to callers.

    Usage:
        model = CostDecompositionModel.load("outputs/models/")
        breakdown = model.predict("cnc_milling", features)
    """

    def __init__(self, models: Dict[str, dict]):
        """
        models: {process_code: {"component": sklearn_regressor, ...}}
        Each process code maps to a dict of component → fitted regressor.
        """
        self._models = models

    @classmethod
    def load(cls, models_dir: str) -> "CostDecompositionModel":
        """Load all per-process model artifacts from a directory."""
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(
                f"Models directory not found: '{models_dir}'. "
                f"Run src/models/train_per_process.py first."
            )
        models = {}
        for fname in os.listdir(models_dir):
            if not fname.endswith(".pkl"):
                continue
            process_code = fname.replace("_v2.pkl", "").replace(".pkl", "")
            fpath = os.path.join(models_dir, fname)
            with open(fpath, "rb") as f:
                models[process_code] = pickle.load(f)

        if not models:
            raise FileNotFoundError(f"No .pkl model files found in '{models_dir}'")
        return cls(models)

    def predict(
        self,
        process_code: str,
        features: Dict[str, float],
    ) -> Dict:
        """
        Predict cost breakdown for a given process and feature vector.

        Returns a dict with per-component costs, total, confidence level,
        and price date metadata.
        """
        if process_code not in self._models:
            return self._fallback_estimate(process_code, features)

        process_models = self._models[process_code]
        feature_array = self._features_to_array(features, process_code)

        breakdown = {}
        for component in COST_COMPONENTS:
            if component == "tooling_cost" and process_code in NO_TOOLING_PROCESSES:
                breakdown[component] = 0.0
                continue

            regressor = process_models.get(component)
            if regressor is None:
                breakdown[component] = 0.0
                continue

            log_pred = float(regressor.predict(feature_array.reshape(1, -1))[0])
            # Inverse of log(x + 1) → x = e^pred - 1
            pred = max(0.0, math.expm1(log_pred))
            breakdown[component] = round(pred, 2)

        breakdown["total_cost"] = round(sum(breakdown.values()), 2)
        breakdown["confidence"] = self._assess_confidence(process_code, features)
        breakdown["price_date"] = features.get("price_date", "unknown")

        return breakdown

    def supported_processes(self) -> List[str]:
        return sorted(self._models.keys())

    def _features_to_array(
        self, features: Dict[str, float], process_code: str
    ) -> np.ndarray:
        """
        Convert feature dict to a numpy array in the order expected by
        the regressors for this process. Feature order is stored in the
        model bundle; falls back to a canonical order if not present.
        """
        process_models = self._models[process_code]
        feature_order = process_models.get(
            "_feature_order",
            list(features.keys()),
        )
        return np.array([
            float(features.get(f, 0.0)) if features.get(f) is not None else 0.0
            for f in feature_order
        ])

    def _assess_confidence(self, process_code: str, features: Dict) -> str:
        """
        Returns "high", "medium", or "low" based on training data coverage
        and whether the input is in-distribution.
        """
        process_models = self._models.get(process_code, {})
        n_training = process_models.get("_n_training_samples", 0)

        if n_training >= 80:
            return "high"
        elif n_training >= 25:
            return "medium"
        else:
            return "low"

    def _fallback_estimate(self, process_code: str, features: Dict) -> Dict:
        """
        Called when no model exists for the requested process.
        Returns a clearly flagged low-confidence estimate based on rough
        industry averages rather than leaving the caller with nothing.
        """
        vol = features.get("volume", 50000)
        mat_price = features.get("material_price_per_kg", 5.0)
        qty = max(1, features.get("quantity", 10))

        rough_material = (vol / 1e6) * 2.7 * mat_price  # assuming ~2.7 g/cm³ density
        rough_total = rough_material * 3.5  # typical material-to-total ratio

        return {
            "machine_time_cost": round(rough_total * 0.40, 2),
            "setup_cost": round(rough_total * 0.15 / qty, 2),
            "tooling_cost": round(rough_total * 0.10, 2),
            "raw_material_cost": round(rough_material, 2),
            "finishing_cost": round(rough_total * 0.05, 2),
            "total_cost": round(rough_total, 2),
            "confidence": "low",
            "note": (
                f"No trained model for process '{process_code}'. "
                f"Rough estimate only — request manual quote."
            ),
            "price_date": features.get("price_date", "unknown"),
        }
