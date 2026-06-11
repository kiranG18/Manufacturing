# src/evaluation/plots.py
#
# Visualisation utilities for model evaluation.
# All functions save to disk and optionally return the figure object.

import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Consistent style across all report figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/confusion_matrix.png",
    title: str = "Process Classifier — Confusion Matrix (v2 test set)",
    normalize: bool = True,
) -> plt.Figure:
    """
    Renders and saves a confusion matrix heatmap.
    Normalized by true class (rows sum to 1) to account for class imbalance.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
        cbar_label = "Fraction of true class"
    else:
        cm_display = cm
        fmt = "d"
        vmax = None
        cbar_label = "Sample count"

    # Only display classes that appear in the test set
    active = [i for i in range(len(class_names)) if cm[i].sum() > 0 or cm[:, i].sum() > 0]
    cm_display = cm_display[np.ix_(active, active)]
    active_names = [class_names[i] for i in active]

    short_names = [n.replace("Sheet Metal ", "SM ").replace("Additive ", "Add. ").replace("CNC ", "") for n in active_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        linewidths=0.3,
        linecolor="#e0e0e0",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )
    ax.set_xlabel("Predicted process", labelpad=10)
    ax.set_ylabel("True process", labelpad=10)
    ax.set_title(title, pad=14)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved confusion matrix → {output_path}")
    return fig


def plot_class_distribution(
    class_counts: dict,
    output_path: str = "outputs/class_distribution.png",
    title: str = "Training Set — Samples per Process Class",
    threshold_line: int = 100,
) -> plt.Figure:
    """
    Horizontal bar chart of sample counts per class.
    A red dashed line marks the threshold below which per-class metrics
    are considered unreliable.
    """
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_items)

    colors = ["#4a90d9" if c >= threshold_line else "#e05c5c" for c in counts]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), counts, color=colors, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(threshold_line, color="#e05c5c", linestyle="--", linewidth=1.2,
               label=f"Reliability threshold (n={threshold_line})")
    ax.set_xlabel("Sample count")
    ax.set_title(title, pad=12)
    ax.legend(fontsize=9)

    for bar, count in zip(bars, counts):
        ax.text(count + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8, color="#555")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved class distribution → {output_path}")
    return fig


def plot_precision_recall(
    per_class_metrics: list,
    output_path: str = "outputs/precision_recall.png",
) -> plt.Figure:
    """Grouped bar chart: precision vs recall per process class."""
    names = [m["process"].replace("CNC ", "").replace("Sheet Metal ", "SM ")
             for m in per_class_metrics if m["support"] > 0]
    precision = [m["precision"] for m in per_class_metrics if m["support"] > 0]
    recall = [m["recall"] for m in per_class_metrics if m["support"] > 0]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, precision, width, label="Precision", color="#4a90d9", alpha=0.85)
    ax.bar(x + width / 2, recall, width, label="Recall", color="#e07b3f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision and Recall (v2 model, test set)")
    ax.legend()
    ax.axhline(0.6, color="#aaa", linestyle=":", linewidth=1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved precision/recall chart → {output_path}")
    return fig
