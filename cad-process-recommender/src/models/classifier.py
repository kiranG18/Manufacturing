# src/models/classifier.py

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import SimpleImputer
import xgboost as xgb

from ..features.definitions import FEATURE_COLUMNS


# The 18 process classes the model was trained on, in class-index order.
# This list is the ground truth — it must match label_map.json exactly.
PROCESS_CLASSES = [
    "cnc_milling",
    "cnc_turning",
    "sheet_metal_bending",
    "sheet_metal_stamping",
    "injection_molding",
    "die_casting",
    "sand_casting",
    "investment_casting",
    "additive_fdm",
    "additive_slm",
    "laser_cutting",
    "forging",
    "extrusion",
    "5axis_cnc",
    "turning_milling",
    "edm",
    "vacuum_forming",
    "powder_coating",
]

PROCESS_DISPLAY_NAMES = {
    "cnc_milling": "CNC Milling",
    "cnc_turning": "CNC Turning",
    "sheet_metal_bending": "Sheet Metal Bending",
    "sheet_metal_stamping": "Sheet Metal Stamping",
    "injection_molding": "Injection Molding",
    "die_casting": "Die Casting",
    "sand_casting": "Sand Casting",
    "investment_casting": "Investment Casting",
    "additive_fdm": "Additive Manufacturing (FDM)",
    "additive_slm": "Additive Manufacturing (SLM)",
    "laser_cutting": "Laser Cutting",
    "forging": "Forging",
    "extrusion": "Extrusion",
    "5axis_cnc": "5-Axis CNC Machining",
    "turning_milling": "CNC Turning + Milling",
    "edm": "EDM (Electrical Discharge Machining)",
    "vacuum_forming": "Vacuum Forming",
    "powder_coating": "Powder Coating",
}


class ProcessClassifier:
    """
    Wrapper around the trained XGBoost process classifier.

    The underlying model is a multi-class XGBoost classifier with softmax
    objective, trained on 13 geometry features and calibrated with Platt
    scaling to improve probability reliability for downstream confidence display.

    Usage:
        model = ProcessClassifier.load("outputs/model_v2.pkl")
        recommendations = model.predict_top_k(features, k=3)
        # -> [("CNC Milling", 0.72), ("Sheet Metal Bending", 0.18), ...]
    """

    def __init__(self, xgb_model, imputer: SimpleImputer, calibrated: bool = True):
        self._model = xgb_model
        self._imputer = imputer
        self._calibrated = calibrated

    @classmethod
    def load(cls, model_path: str) -> "ProcessClassifier":
        """Load a serialized classifier from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                f"Run src/models/train.py to generate it."
            )
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return cls(
            xgb_model=bundle["model"],
            imputer=bundle["imputer"],
            calibrated=bundle.get("calibrated", True),
        )

    def save(self, model_path: str) -> None:
        """Serialize classifier to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        bundle = {
            "model": self._model,
            "imputer": self._imputer,
            "calibrated": self._calibrated,
            "feature_columns": FEATURE_COLUMNS,
            "process_classes": PROCESS_CLASSES,
        }
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)

    def predict_proba(self, features: Dict[str, float]) -> np.ndarray:
        """
        Returns a probability distribution over all 18 process classes.
        Features are passed as a dict; missing values are median-imputed.
        """
        x = self._dict_to_array(features)
        x_imputed = self._imputer.transform(x.reshape(1, -1))
        proba = self._model.predict_proba(x_imputed)[0]
        return proba

    def predict_top_k(
        self,
        features: Dict[str, float],
        k: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns the top-k process recommendations.

        Returns list of (process_code, display_name, probability) tuples,
        sorted descending by probability.
        """
        proba = self.predict_proba(features)
        top_indices = np.argsort(proba)[::-1][:k]
        results = []
        for idx in top_indices:
            code = PROCESS_CLASSES[idx]
            name = PROCESS_DISPLAY_NAMES.get(code, code)
            results.append((code, name, float(proba[idx])))
        return results

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(col, np.nan) for col in FEATURE_COLUMNS])

    @property
    def n_classes(self) -> int:
        return len(PROCESS_CLASSES)

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURE_COLUMNS)
