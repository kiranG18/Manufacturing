# app/services/cost_service.py
import os
import pickle
import math
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton cost model instance
_model = None
_ready: bool = False


# The 5 cost components the model predicts separately.
COST_COMPONENTS = [
    "machine_time_cost",
    "setup_cost",
    "tooling_cost",
    "raw_material_cost",
    "finishing_cost",
]

# Processes that don't have tooling in the traditional sense
NO_TOOLING_PROCESSES = {"additive_fdm", "additive_slm", "laser_cutting"}


class CostDecompositionModel:
    """
    Loads and manages per-process cost decomposition regressors.
    """

    def __init__(self, models: Dict[str, dict]):
        self._models = models

    @classmethod
    def load(cls, models_dir: str) -> "CostDecompositionModel":
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: '{models_dir}'.")
        
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

    def predict(self, process_code: str, features: Dict[str, float]) -> Dict:
        if process_code not in self._models:
            return self._fallback_estimate(process_code, features)

        process_models = self._models[process_code]
        feature_array = self._features_to_array(features, process_code)

        import numpy as np
        breakdown = {}
        for component in COST_COMPONENTS:
            if component == "tooling_cost" and process_code in NO_TOOLING_PROCESSES:
                breakdown[component] = 0.0
                continue

            regressor = process_models.get(component)
            if regressor is None:
                breakdown[component] = 0.0
                continue

            # Run prediction and inverse-log
            log_pred = float(regressor.predict(feature_array.reshape(1, -1))[0])
            pred = max(0.0, math.expm1(log_pred))
            breakdown[component] = round(pred, 2)

        breakdown["total_cost"] = round(sum(breakdown.values()), 2)
        breakdown["confidence"] = self._assess_confidence(process_code, features)
        breakdown["price_date"] = features.get("price_date", "unknown")

        return breakdown

    def supported_processes(self) -> List[str]:
        return sorted(self._models.keys())

    def _features_to_array(self, features: Dict[str, float], process_code: str):
        process_models = self._models[process_code]
        feature_order = process_models.get("_feature_order", list(features.keys()))
        
        import numpy as np
        return np.array([
            float(features.get(f, 0.0)) if features.get(f) is not None else 0.0
            for f in feature_order
        ])

    def _assess_confidence(self, process_code: str, features: Dict) -> str:
        process_models = self._models.get(process_code, {})
        n_training = process_models.get("_n_training_samples", 0)

        if n_training >= 80:
            return "high"
        elif n_training >= 25:
            return "medium"
        else:
            return "low"

    def _fallback_estimate(self, process_code: str, features: Dict) -> Dict:
        vol = features.get("volume", 50000.0)
        mat_price = features.get("material_price_per_kg", 5.0)
        qty = max(1, features.get("quantity", 10))

        rough_material = (vol / 1e6) * 2.7 * mat_price
        rough_total = rough_material * 3.5

        return {
            "machine_time_cost": round(rough_total * 0.40, 2),
            "setup_cost": round(rough_total * 0.15 / qty, 2),
            "tooling_cost": round(rough_total * 0.10, 2),
            "raw_material_cost": round(rough_material, 2),
            "finishing_cost": round(rough_total * 0.05, 2),
            "total_cost": round(rough_total, 2),
            "confidence": "low",
            "note": f"No trained model for process '{process_code}'. Fallback estimate used.",
            "price_date": features.get("price_date", "unknown"),
        }


def initialise(models_dir: str) -> None:
    global _model, _ready
    try:
        _model = CostDecompositionModel.load(models_dir)
        _ready = True
        logger.info(f"Cost models loaded from {models_dir} ({len(_model.supported_processes())} processes)")
    except Exception as e:
        logger.error(f"Failed to load cost models from {models_dir}: {e}")
        # Initialize an empty model so it runs fallbacks
        _model = CostDecompositionModel({})
        _ready = True


def is_ready() -> bool:
    return _ready


def predict(process_code: str, features: Dict[str, float]) -> Optional[Dict]:
    if not _ready:
        raise RuntimeError("Cost service not initialised.")
    return _model.predict(process_code, features)


estimate = predict


def supported_processes() -> List[str]:
    if not _ready:
        return []
    return _model.supported_processes()
