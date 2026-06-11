# src/evaluation/metrics.py
#
# Custom evaluation metrics for the process classifier.
# Standard sklearn metrics don't include top-k accuracy, which is the primary
# metric we optimise for. This module provides that plus helpers for producing
# the per-class precision/recall tables in the evaluation reports.

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, classification_report


def top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 3,
) -> float:
    """
    Fraction of samples where the true class appears in the top-k predictions.

    This is the primary evaluation metric for the process classifier because
    in production the model returns a ranked list and engineers select from it.
    Being ranked 2nd is far less costly than being absent from the top 3.

    Args:
        y_true:  1-D array of integer class labels
        y_proba: 2-D array of per-class probabilities, shape (n_samples, n_classes)
        k:       Number of top predictions to consider

    Returns:
        Float in [0, 1]
    """
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2-D, got shape {y_proba.shape}")
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same number of samples")

    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]  # top-k indices
    hits = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def per_class_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> List[Dict]:
    """
    Per-class precision, recall, F1, and support.
    Returns a list of dicts (one per class) suitable for tabular output.
    """
    y_pred = np.argmax(y_proba, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    results = []
    for i, name in enumerate(class_names):
        results.append({
            "process": name,
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        })
    # Sort by F1 descending
    results.sort(key=lambda x: -x["f1"])
    return results


def coverage_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 3,
    confidence_threshold: float = 0.15,
) -> Dict[str, float]:
    """
    Reports how often the top-k list contains the correct answer,
    broken down by whether the top-1 probability exceeds the confidence threshold.

    This distinguishes two failure modes:
    1. Model is confident and wrong (high-confidence error, most dangerous)
    2. Model is uncertain and wrong (expected; these get flagged for review)
    """
    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    top1_proba = y_proba[np.arange(len(y_true)), np.argmax(y_proba, axis=1)]
    in_top_k = np.array([y_true[i] in top_k[i] for i in range(len(y_true))])

    high_conf_mask = top1_proba >= confidence_threshold
    low_conf_mask = ~high_conf_mask

    return {
        f"top_{k}_accuracy": float(np.mean(in_top_k)),
        f"top_{k}_accuracy_high_confidence": float(
            np.mean(in_top_k[high_conf_mask]) if high_conf_mask.any() else 0.0
        ),
        f"top_{k}_accuracy_low_confidence": float(
            np.mean(in_top_k[low_conf_mask]) if low_conf_mask.any() else 0.0
        ),
        "high_confidence_fraction": float(np.mean(high_conf_mask)),
        "low_confidence_fraction": float(np.mean(low_conf_mask)),
    }


def macro_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Macro-averaged F1 score across all classes."""
    from sklearn.metrics import f1_score
    y_pred = np.argmax(y_proba, axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Weighted-averaged F1, weighted by class support."""
    from sklearn.metrics import f1_score
    y_pred = np.argmax(y_proba, axis=1)
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
