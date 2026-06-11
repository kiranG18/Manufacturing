# src/validation/drift_detector.py
import logging
import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Compares the incoming dataset distribution with a baseline dataset
    (typically the previous dataset version) to detect statistical drift
    in geometry features, quantities, and cost labels.
    """

    def __init__(self, data_dir: str = "data/processed", drift_threshold_std: float = 0.5):
        self.data_dir = data_dir
        self.drift_threshold_std = drift_threshold_std

    def find_latest_baseline(self) -> Optional[str]:
        """Finds the path of the latest training_data_vN.csv file."""
        if not os.path.isdir(self.data_dir):
            return None

        pattern = re.compile(r"training_data_v(\d+)\.csv")
        files = []
        for fname in os.listdir(self.data_dir):
            match = pattern.match(fname)
            if match:
                v = int(match.group(1))
                files.append((v, os.path.join(self.data_dir, fname)))

        if not files:
            return None

        # Sort by version number and return path of the highest version
        return sorted(files)[-1][1]

    def detect_drift(
        self,
        new_df: pd.DataFrame,
        baseline_path: str = None,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Compare new_df against baseline dataset.
        Returns (drift_detected, drift_metrics, list_of_warnings).
        """
        warnings = []
        metrics = {}
        drift_detected = False

        if baseline_path is None:
            baseline_path = self.find_latest_baseline()

        if not baseline_path or not os.path.exists(baseline_path):
            logger.info("No baseline dataset found. Skipping drift detection.")
            return False, {"note": "No baseline dataset found"}, []

        try:
            base_df = pd.read_csv(baseline_path)
            logger.info(f"Comparing incoming data (n={len(new_df)}) to baseline '{baseline_path}' (n={len(base_df)})")
        except Exception as e:
            logger.error(f"Failed to load baseline dataset: {e}")
            return False, {"error": f"Failed to load baseline: {e}"}, []

        # Compare numeric features
        numeric_cols = [
            "volume", "surface_area", "wall_thickness_min", "wall_thickness_avg",
            "quantity", "material_price_per_kg", "cost_total"
        ]

        # Use Kolmogorov-Smirnov test if scipy is available, otherwise fall back to mean-shift
        use_ks = False
        try:
            from scipy.stats import ks_2samp
            use_ks = True
        except ImportError:
            logger.warning("scipy.stats.ks_2samp is not available. Falling back to mean-shift check.")

        drift_metrics = {}

        for col in numeric_cols:
            if col not in new_df.columns or col not in base_df.columns:
                continue

            new_col = new_df[col].dropna()
            base_col = base_df[col].dropna()

            if len(new_col) < 5 or len(base_col) < 5:
                continue

            # Standard stats comparison
            base_mean = float(base_col.mean())
            base_std = float(base_col.std())
            new_mean = float(new_col.mean())

            if base_std > 0:
                shift = abs(new_mean - base_mean) / base_std
            else:
                shift = 0.0

            drift_metrics[col] = {
                "base_mean": base_mean,
                "new_mean": new_mean,
                "std_shift": shift,
                "drift_flag": False
            }

            if use_ks:
                # KS test: null hypothesis is that both samples are from the same distribution.
                # If p-value is small (e.g. < 0.01), we reject null hypothesis -> drift detected.
                stat, p_val = ks_2samp(base_col, new_col)
                drift_metrics[col]["ks_pval"] = float(p_val)
                if p_val < 0.01:
                    drift_metrics[col]["drift_flag"] = True
                    drift_detected = True
                    warnings.append(
                        f"DRIFT WARNING: Column '{col}' failed distribution equality check (KS p-val = {p_val:.4f})."
                    )
            else:
                # Fallback standard dev shift check
                if shift > self.drift_threshold_std:
                    drift_metrics[col]["drift_flag"] = True
                    drift_detected = True
                    warnings.append(
                        f"DRIFT WARNING: Column '{col}' shifted by {shift:.2f} standard deviations (threshold={self.drift_threshold_std})."
                    )

        # Compare categorical process distribution
        if "process_code" in new_df.columns and "process_code" in base_df.columns:
            base_counts = base_df["process_code"].value_counts(normalize=True).to_dict()
            new_counts = new_df["process_code"].value_counts(normalize=True).to_dict()
            
            proc_drift = {}
            for proc in set(base_counts.keys()) | set(new_counts.keys()):
                b_prop = base_counts.get(proc, 0.0)
                n_prop = new_counts.get(proc, 0.0)
                diff = abs(n_prop - b_prop)
                proc_drift[proc] = {"base_prop": b_prop, "new_prop": n_prop, "diff": diff}
                
                # If a process ratio shifts by more than 15% absolute in the dataset, warn
                if diff > 0.15:
                    drift_detected = True
                    warnings.append(
                        f"DRIFT WARNING: process_code '{proc}' proportion shifted from {b_prop:.1%} to {n_prop:.1%} (diff {diff:.1%})"
                    )
            drift_metrics["process_code_drift"] = proc_drift

        metrics["drift_metrics"] = drift_metrics
        metrics["drift_detected"] = drift_detected

        if drift_detected:
            logger.warning(f"Distribution drift detected across {len(warnings)} columns.")
        else:
            logger.info("No significant distribution drift detected vs. baseline.")

        return drift_detected, metrics, warnings
