# app/services/classifier_service.py
#
# Module-level service wrapper for the process classifier.
# Provides a clean interface that the route handlers can call without
# needing to know anything about pickle loading or model internals.
# This also makes the service easy to mock in tests.

import logging
import pickle
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_imputer = None
_process_classes: List[str] = []
_feature_columns: List[str] = []
_model_version_tag: str = "unknown"
_ready: bool = False

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
    "edm": "EDM",
    "vacuum_forming": "Vacuum Forming",
    "powder_coating": "Powder Coating",
}


def initialise(model_path: str) -> None:
    """Load the model from disk. Called once at startup."""
    global _model, _imputer, _process_classes, _feature_columns, _model_version_tag, _ready

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    _model = bundle["model"]
    _imputer = bundle["imputer"]
    _process_classes = bundle.get("process_classes", [])
    _feature_columns = bundle.get("feature_columns", [])
    _model_version_tag = model_path.split("/")[-1].replace(".pkl", "")
    _ready = True
    logger.info(f"Classifier loaded from {model_path} ({len(_process_classes)} classes)")


def is_ready() -> bool:
    return _ready


def model_version() -> str:
    return _model_version_tag


def predict(features: Dict[str, float], top_k: int = 3) -> List[Dict]:
    """
    Run inference and return top-k ranked predictions.
    """
    if not _ready:
        raise RuntimeError("Classifier service not initialised. Call initialise() first.")

    x = np.array([
        float(features.get(col, np.nan)) if features.get(col) is not None else np.nan
        for col in _feature_columns
    ])
    x_imp = _imputer.transform(x.reshape(1, -1))
    proba = _model.predict_proba(x_imp)[0]

    top_indices = np.argsort(proba)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        code = _process_classes[idx]
        prob = float(proba[idx])
        results.append({
            "rank": rank,
            "process_code": code,
            "process_name": PROCESS_DISPLAY_NAMES.get(code, code),
            "probability": round(prob, 4),
            "confidence": _prob_to_confidence(prob),
        })
    return results


def _prob_to_confidence(prob: float) -> str:
    if prob >= 0.55:
        return "high"
    elif prob >= 0.25:
        return "medium"
    return "low"
