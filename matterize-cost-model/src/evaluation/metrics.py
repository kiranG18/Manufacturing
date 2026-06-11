# src/evaluation/metrics.py
#
# Cost model evaluation metrics.
# MAPE is the primary metric because part costs span 2+ orders of magnitude —
# an absolute error of $20 means something very different on a $50 part vs a $5,000 casting.

import numpy as np
from typing import Dict, List, Optional


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_true_value: float = 1.0,
) -> float:
    """
    Mean Absolute Percentage Error, excluding near-zero true values.

    The min_true_value threshold prevents near-zero cost components (e.g.
    finishing cost for a bare CNC part) from inflating the MAPE to nonsense values.

    Returns: MAPE as a percentage (e.g. 8.4 means 8.4%)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= min_true_value
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in dollars."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def coverage_within_10pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Fraction of predictions within 10% of actual cost.
    Used in stakeholder reports as a plain-English accuracy metric:
    "X% of estimates are within 10% of the actual invoice."
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    pct_error = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_error <= 0.10))


def per_component_metrics(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> List[Dict]:
    """
    Compute MAPE and MAE per cost component.
    Returns a list of dicts for tabular display.
    """
    results = []
    for component in sorted(y_true_dict.keys()):
        y_t = y_true_dict[component]
        y_p = y_pred_dict[component]
        results.append({
            "component": component,
            "mape": round(mape(y_t, y_p), 1),
            "mae": round(mae(y_t, y_p), 2),
            "within_10pct": round(coverage_within_10pct(y_t, y_p) * 100, 1),
            "n": int(np.sum(np.asarray(y_t) > 0)),
        })
    return results


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Signed mean error (positive = overestimate, negative = underestimate).
    Helps detect systematic bias — e.g. the model consistently underestimates
    tooling for injection molding.
    """
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))
