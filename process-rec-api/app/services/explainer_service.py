# app/services/explainer_service.py
import logging
from typing import Dict, List, Any
import numpy as np
from . import classifier_service

logger = logging.getLogger(__name__)


def explain_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes SHAP feature contributions for the top predicted manufacturing process.
    Falls back to a deterministic heuristic explanation if SHAP calculation fails.
    """
    if not classifier_service.is_ready():
        raise RuntimeError("Classifier service is not ready.")

    # 1. Run classifier predict to get the top class and probabilities
    preds = classifier_service.predict(features, top_k=1)
    if not preds:
        raise RuntimeError("Model did not return any process recommendations.")
        
    top_pred = preds[0]
    pred_process = top_pred["process_code"]
    prob = top_pred["probability"]

    # Retrieve classifier components
    model = classifier_service._model
    imputer = classifier_service._imputer
    process_classes = classifier_service._process_classes
    feature_cols = classifier_service._feature_columns

    pred_class_idx = 0
    if pred_process in process_classes:
        pred_class_idx = process_classes.index(pred_process)

    # Prepare input array
    x = np.array([
        float(features.get(col, np.nan)) if features.get(col) is not None else np.nan
        for col in feature_cols
    ])
    x_imp = imputer.transform(x.reshape(1, -1))

    shap_contributions = {}
    explanation_used = "shap"

    try:
        import shap
        # SHAP TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(x_imp)
        
        # For multi-class, shap_values can be a list (one array per class) or a 3D array
        if isinstance(raw_shap, list):
            # Length equal to classes, grab target class array of shape (1, n_features)
            class_shap = raw_shap[pred_class_idx][0]
        else:
            # Shape (1, n_features, n_classes) or (1, n_features)
            if len(raw_shap.shape) == 3:
                class_shap = raw_shap[0, :, pred_class_idx]
            else:
                class_shap = raw_shap[0]
                
        for i, col in enumerate(feature_cols):
            shap_contributions[col] = float(class_shap[i])
    except Exception as e:
        logger.warning(f"Failed to compute SHAP values via shap library: {e}. Using heuristic fallback.")
        explanation_used = "heuristic"
        # Deterministic fallback heuristics for explainability if SHAP fails
        # Assign values that physically represent why a process is chosen
        for col in feature_cols:
            val = float(features.get(col, 0.0))
            # Standard heuristic contributions based on process logic
            if pred_process == "cnc_milling":
                if col == "wall_thickness_min":
                    shap_contributions[col] = 0.25 if val >= 2.0 else -0.15
                elif col == "hole_count":
                    shap_contributions[col] = 0.15 if val > 0 else 0.0
                elif col == "undercut_flag":
                    shap_contributions[col] = -0.3 if val == 1 else 0.1
                else:
                    shap_contributions[col] = 0.01
            elif pred_process == "injection_molding":
                if col == "quantity":
                    shap_contributions[col] = 0.40 if val >= 500 else -0.50
                elif col == "undercut_flag":
                    shap_contributions[col] = -0.60 if val == 1 else 0.20
                elif col == "wall_thickness_min":
                    shap_contributions[col] = 0.20 if 1.5 <= val <= 4.0 else -0.30
                else:
                    shap_contributions[col] = 0.01
            elif pred_process == "sheet_metal_bending":
                if col == "wall_thickness_avg":
                    shap_contributions[col] = 0.30 if 1.0 <= val <= 5.0 else -0.40
                elif col == "aspect_ratio":
                    shap_contributions[col] = 0.20 if val >= 4.0 else -0.10
                else:
                    shap_contributions[col] = 0.01
            else:
                # Default fallback
                if col == "volume":
                    shap_contributions[col] = 0.15
                else:
                    shap_contributions[col] = 0.01

    # Format into List[FeatureContribution]
    contributions = []
    for col in feature_cols:
        val = float(features.get(col, 0.0))
        contrib_val = shap_contributions.get(col, 0.0)
        direction = "positive" if contrib_val >= 0 else "negative"
        contributions.append({
            "feature_name": col,
            "value": val,
            "shap_contribution": round(contrib_val, 4),
            "direction": direction
        })

    # Sort by absolute SHAP contribution descending
    contributions.sort(key=lambda item: abs(item["shap_contribution"]), reverse=True)

    # Build plain-English summary of top 2 features
    top_2 = [c for c in contributions[:2] if abs(c["shap_contribution"]) > 0.01]
    if len(top_2) >= 2:
        f1, f2 = top_2[0], top_2[1]
        sum_str = (
            f"The recommendation is primarily driven by {f1['feature_name']} (value={f1['value']}) "
            f"which had a {f1['direction']} effect, followed by {f2['feature_name']} (value={f2['value']}) "
            f"with a {f2['direction']} effect."
        )
    elif len(top_2) == 1:
        f1 = top_2[0]
        sum_str = f"The recommendation is primarily driven by {f1['feature_name']} (value={f1['value']}) which had a {f1['direction']} effect."
    else:
        sum_str = f"The recommendation is based on a balanced contribution across several geometric metrics."

    return {
        "predicted_process": pred_process,
        "probability": float(prob),
        "shap_values": contributions,
        "top_features": sum_str,
        "explanation_mode": explanation_used
    }
