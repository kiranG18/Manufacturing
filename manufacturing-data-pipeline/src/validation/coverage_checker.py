# src/validation/coverage_checker.py
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


class CoverageChecker:
    """
    Checks that the dataset has sufficient coverage (sample counts) across
    all process codes and material codes before model training runs.
    """

    def __init__(self, min_samples_per_process: int = 20, warn_threshold: int = 100):
        self.min_samples = min_samples_per_process
        self.warn_threshold = warn_threshold

    def check_coverage(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Aggregates coverage counts and checks against thresholds.
        Returns (is_valid, coverage_stats, list_of_warnings_or_errors).
        """
        warnings = []
        is_valid = True
        stats = {}

        if df.empty:
            return False, {}, ["Dataset is empty. Coverage is 0."]

        # 1. Process Code coverage
        if "process_code" in df.columns:
            proc_counts = df["process_code"].value_counts().to_dict()
            stats["process_coverage"] = proc_counts

            for proc, count in proc_counts.items():
                if count < self.min_samples:
                    warnings.append(
                        f"CRITICAL: process_code '{proc}' has only {count} rows. "
                        f"Below absolute minimum threshold of {self.min_samples}."
                    )
                    is_valid = False
                elif count < self.warn_threshold:
                    warnings.append(
                        f"WARNING: process_code '{proc}' has {count} rows. "
                        f"Below recommended size of {self.warn_threshold}."
                    )
        else:
            warnings.append("ERROR: process_code column missing in dataset.")
            is_valid = False

        # 2. Material Code coverage
        if "material_code" in df.columns:
            mat_counts = df["material_code"].value_counts().to_dict()
            stats["material_coverage"] = mat_counts
            
            for mat, count in mat_counts.items():
                if count < 5:
                    warnings.append(
                        f"WARNING: material_code '{mat}' has very low representation ({count} rows)."
                    )
        else:
            warnings.append("WARNING: material_code column missing in dataset.")

        # 3. Material Price Freshness
        if "price_staleness_days" in df.columns:
            avg_staleness = float(df["price_staleness_days"].mean())
            max_staleness = int(df["price_staleness_days"].max())
            stats["price_staleness"] = {
                "average_days": avg_staleness,
                "max_days": max_staleness
            }
            if max_staleness > 7:
                warnings.append(
                    f"WARNING: Some material prices are stale by up to {max_staleness} days. "
                    f"Outdated spot pricing may introduce prediction bias."
                )

        # Print a nice summary in log
        logger.info("=== DATASET COVERAGE SUMMARY ===")
        if "process_coverage" in stats:
            for proc, count in stats["process_coverage"].items():
                status = "OK" if count >= self.warn_threshold else ("WARN" if count >= self.min_samples else "FAIL")
                logger.info(f"  {proc:<22} | {count:<5} rows | {status}")

        return is_valid, stats, warnings
