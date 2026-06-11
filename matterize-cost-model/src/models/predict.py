# src/models/predict.py
#
# Cost model inference entry point.
# Loads per-process models and returns cost breakdowns.

import os
from typing import Dict, Optional

from .cost_regressor import CostDecompositionModel


_CACHED_MODEL: Optional[CostDecompositionModel] = None
_CACHED_DIR: Optional[str] = None


def load_models(models_dir: str, force_reload: bool = False) -> CostDecompositionModel:
    """Load and cache the per-process model bundle."""
    global _CACHED_MODEL, _CACHED_DIR
    if _CACHED_MODEL is not None and _CACHED_DIR == models_dir and not force_reload:
        return _CACHED_MODEL
    _CACHED_MODEL = CostDecompositionModel.load(models_dir)
    _CACHED_DIR = models_dir
    return _CACHED_MODEL


def predict_cost_breakdown(
    process_code: str,
    features: Dict[str, float],
    models_dir: str = "outputs/models/",
) -> Dict:
    """
    Main cost inference function.

    Args:
        process_code: e.g. "cnc_milling"
        features:     Combined geometry + cost context feature dict
        models_dir:   Directory containing per-process .pkl files

    Returns:
        Cost breakdown dict including machine_time_cost, setup_cost,
        tooling_cost, raw_material_cost, finishing_cost, total_cost,
        confidence, and price_date.
    """
    model = load_models(models_dir)
    return model.predict(process_code, features)
