# src/validation/schema_validator.py
import logging
import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates a training dataset against rules defined in validation_rules.yaml.
    Fails invalid rows and returns validation statistics.
    """

    def __init__(self, rules_path: str = "configs/validation_rules.yaml"):
        self.rules_path = rules_path
        self._load_rules()

    def _load_rules(self):
        if not os.path.exists(self.rules_path):
            # Try absolute path from workspace root
            root_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs", "validation_rules.yaml"
            )
            if os.path.exists(root_path):
                self.rules_path = root_path

        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r") as f:
                rules = yaml.safe_load(f)
                self.columns_rules = rules.get("columns", {})
        else:
            logger.warning(f"Rules file {self.rules_path} not found. SchemaValidator will pass all rows.")
            self.columns_rules = {}

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate the DataFrame column types, numeric ranges, and nulls.
        Drops rows that fail required/drop_row policies or are outside bounds.
        Returns (validated_df, validation_metrics).
        """
        if df.empty or not self.columns_rules:
            return df, {"total_rows": len(df), "dropped_rows": 0, "dropped_details": {}}

        n_start = len(df)
        df_clean = df.copy()

        dropped_indices = set()
        dropped_reasons = {}

        # Scan each rule
        for col, col_rules in self.columns_rules.items():
            if col not in df_clean.columns:
                if col_rules.get("null_policy") == "required":
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame(), {"error": f"Missing required column {col}"}
                continue

            # 1. Null Policy check
            null_policy = col_rules.get("null_policy")
            null_mask = df_clean[col].isna()
            null_count = null_mask.sum()

            if null_count > 0:
                if null_policy in ["required", "drop_row"]:
                    to_drop = df_clean[null_mask].index
                    dropped_indices.update(to_drop)
                    for idx in to_drop:
                        dropped_reasons[idx] = f"Null value in required column '{col}'"
                elif null_policy == "default_value" and "default" in col_rules:
                    df_clean[col] = df_clean[col].fillna(col_rules["default"])
                elif null_policy == "fallback_to_historical_avg" and col == "material_price_per_kg":
                    # Imputation already done upstream in normalizer/joiner, but double check
                    df_clean[col] = df_clean[col].fillna(5.0)  # generic fallback price

            # 2. Type cast check
            col_type = col_rules.get("type")
            try:
                if col_type == "int":
                    # Fill na to avoid float conversion error
                    df_clean[col] = df_clean[col].fillna(0).astype(int)
                elif col_type == "float":
                    df_clean[col] = df_clean[col].astype(float)
                elif col_type == "string":
                    df_clean[col] = df_clean[col].astype(str)
            except Exception as e:
                logger.warning(f"Failed to cast column '{col}' to type '{col_type}': {e}")

            # 3. Numeric range check
            col_range = col_rules.get("range")
            if col_range and col_type in ["int", "float"]:
                min_val, max_val = col_range
                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                except (ValueError, TypeError):
                    logger.warning(f"Failed to cast range limits {[min_val, max_val]} to float for column '{col}'")
                    continue
                # Find indices outside range (excluding NaNs)
                out_of_bounds = df_clean[
                    (df_clean[col].notna()) &
                    ((df_clean[col] < min_val) | (df_clean[col] > max_val))
                ].index
                if len(out_of_bounds) > 0:
                    dropped_indices.update(out_of_bounds)
                    for idx in out_of_bounds:
                        val = df_clean.at[idx, col]
                        dropped_reasons[idx] = f"Column '{col}' value {val} out of range {col_range}"

            # 4. Allowed values check
            allowed = col_rules.get("allowed_values")
            if allowed:
                # Find indices with unallowed values
                unallowed = df_clean[
                    (df_clean[col].notna()) &
                    (~df_clean[col].isin(allowed))
                ].index
                if len(unallowed) > 0:
                    dropped_indices.update(unallowed)
                    for idx in unallowed:
                        val = df_clean.at[idx, col]
                        dropped_reasons[idx] = f"Column '{col}' value '{val}' not in allowed list"

        # Unique index check (e.g. drop duplicate part_id or order_id)
        if "part_id" in df_clean.columns:
            duplicates = df_clean[df_clean.duplicated(subset=["part_id", "process_code"], keep="first")].index
            if len(duplicates) > 0:
                dropped_indices.update(duplicates)
                for idx in duplicates:
                    dropped_reasons[idx] = "Duplicate combination of part_id and process_code"

        # Drop the flagged rows
        if dropped_indices:
            df_clean = df_clean.drop(index=list(dropped_indices)).reset_index(drop=True)

        n_end = len(df_clean)
        dropped_count = n_start - n_end

        # Group dropped reasons
        reason_summary = {}
        for reason in dropped_reasons.values():
            # truncate specific values to keep summary clean
            base_reason = reason.split(" value ")[0] if " value " in reason else reason
            reason_summary[base_reason] = reason_summary.get(base_reason, 0) + 1

        metrics = {
            "total_rows_pre_validation": n_start,
            "total_rows_post_validation": n_end,
            "dropped_rows_count": dropped_count,
            "dropped_reasons_summary": reason_summary,
            "pass_rate_pct": round(n_end / max(n_start, 1) * 100, 1),
        }

        logger.info(f"Schema validation completed: {dropped_count} rows dropped ({metrics['pass_rate_pct']}% pass rate)")
        return df_clean, metrics
