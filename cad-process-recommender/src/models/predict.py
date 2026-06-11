# src/models/predict.py
#
# Inference module for the process classifier.
# Loads a saved model artifact, accepts a feature dict, and returns
# a ranked process recommendation list.

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from .classifier import ProcessClassifier, PROCESS_CLASSES, PROCESS_DISPLAY_NAMES
from ..features.definitions import FEATURE_COLUMNS


_CACHED_MODEL: Optional[ProcessClassifier] = None
_CACHED_MODEL_PATH: Optional[str] = None


def load_model(model_path: str, force_reload: bool = False) -> ProcessClassifier:
    """
    Loads and caches the classifier. Subsequent calls with the same path
    return the cached instance without re-reading from disk.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == model_path and not force_reload:
        return _CACHED_MODEL

    _CACHED_MODEL = ProcessClassifier.load(model_path)
    _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL


def predict_top_k(
    features: Dict[str, float],
    model_path: str = "outputs/model_v2.pkl",
    k: int = 3,
) -> List[Dict]:
    """
    Main inference function. Takes a feature dict and returns a ranked list
    of process recommendations.

    Returns:
        List of dicts, each with:
            process_code    str    e.g. "cnc_milling"
            process_name    str    e.g. "CNC Milling"
            probability     float  calibrated softmax probability
            confidence      str    "high" / "medium" / "low"
    """
    model = load_model(model_path)
    raw_recommendations = model.predict_top_k(features, k=k)

    results = []
    for rank, (code, name, prob) in enumerate(raw_recommendations, 1):
        confidence = _probability_to_confidence(prob, rank)
        results.append({
            "rank": rank,
            "process_code": code,
            "process_name": name,
            "probability": round(prob, 4),
            "confidence": confidence,
        })
    return results


def _probability_to_confidence(prob: float, rank: int) -> str:
    """
    Maps a calibrated probability to a human-readable confidence level.

    Thresholds derived from calibration analysis on validation set:
    - P > 0.55 → predictions were correct in top-1 ~82% of the time
    - P 0.25–0.55 → correct in top-1 ~58% of the time
    - P < 0.25 → lower confidence; surface with caveat
    """
    if prob >= 0.55:
        return "high"
    elif prob >= 0.25:
        return "medium"
    else:
        return "low"
