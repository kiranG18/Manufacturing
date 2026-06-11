# src/evaluation/plots.py
#
# Visualisation utilities for cost model evaluation.

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    process_code: str,
    component: str,
    output_path: str,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted cost for a single process/component.
    Perfect predictions lie on the diagonal. Points above are underestimates;
    points below are overestimates.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.55, s=28, color="#4a90d9", edgecolors="none")

    lim_max = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

    # ±10% bands
    ax.fill_between(
        [0, lim_max],
        [0, lim_max * 0.9],
        [0, lim_max * 1.1],
        alpha=0.10, color="#4a90d9", label="±10% band"
    )

    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Actual cost (USD)")
    ax.set_ylabel("Predicted cost (USD)")
    ax.set_title(f"{process_code.replace('_', ' ').title()} — {component.replace('_', ' ').title()}")
    ax.legend(fontsize=9)

    # Annotate MAPE
    from .metrics import mape as compute_mape
    m = compute_mape(y_true, y_pred)
    ax.text(0.05, 0.93, f"MAPE: {m:.1f}%", transform=ax.transAxes,
            fontsize=10, color="#333", verticalalignment="top")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    process_code: str,
    output_path: str,
) -> plt.Figure:
    """
    Residual percentage error distribution. Should be centred near 0;
    a shift indicates systematic bias.
    """
    pct_error = (y_pred - y_true) / np.maximum(y_true, 1e-6) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pct_error, bins=40, color="#4a90d9", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(np.median(pct_error), color="#e05c5c", linewidth=1.2,
               linestyle="-", label=f"Median: {np.median(pct_error):.1f}%")
    ax.set_xlabel("Percentage error (predicted − actual) / actual × 100")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual distribution — {process_code.replace('_', ' ').title()}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_mape_by_process(
    process_mapes: Dict[str, float],
    output_path: str,
) -> plt.Figure:
    """Bar chart of MAPE per process, coloured by reliability tier."""
    sorted_items = sorted(process_mapes.items(), key=lambda x: x[1])
    names, mapes = zip(*sorted_items)
    colors = ["#4caf50" if m < 10 else "#ff9800" if m < 15 else "#e05c5c" for m in mapes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), mapes, color=colors, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ").title() for n in names], fontsize=9)
    ax.axvline(10, color="#777", linestyle="--", linewidth=1, label="10% threshold")
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Cost Model MAPE by Process (v2, test set)")
    ax.legend(fontsize=9)
    for bar, m in zip(bars, mapes):
        ax.text(m + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{m:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig
