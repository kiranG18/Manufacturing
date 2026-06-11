# src/transforms/label_builder.py
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Process-specific cost decomposition ratios
DECOMPOSITION_RATIOS = {
    "cnc_milling": {
        "machine_time": 0.45,
        "setup": 0.15,
        "tooling": 0.10,
        "raw_material": 0.25,
        "finishing": 0.05,
    },
    "cnc_turning": {
        "machine_time": 0.45,
        "setup": 0.15,
        "tooling": 0.10,
        "raw_material": 0.25,
        "finishing": 0.05,
    },
    "sheet_metal_bending": {
        "machine_time": 0.30,
        "setup": 0.25,
        "tooling": 0.10,
        "raw_material": 0.25,
        "finishing": 0.10,
    },
    "sheet_metal_stamping": {
        "machine_time": 0.20,
        "setup": 0.20,
        "tooling": 0.30,
        "raw_material": 0.20,
        "finishing": 0.10,
    },
    "injection_molding": {
        "machine_time": 0.20,
        "setup": 0.10,
        "tooling": 0.50,
        "raw_material": 0.15,
        "finishing": 0.05,
    },
    "die_casting": {
        "machine_time": 0.25,
        "setup": 0.10,
        "tooling": 0.45,
        "raw_material": 0.15,
        "finishing": 0.05,
    },
    "sand_casting": {
        "machine_time": 0.35,
        "setup": 0.15,
        "tooling": 0.25,
        "raw_material": 0.15,
        "finishing": 0.10,
    },
    "additive_fdm": {
        "machine_time": 0.55,
        "setup": 0.15,
        "tooling": 0.00,
        "raw_material": 0.25,
        "finishing": 0.05,
    },
    "additive_slm": {
        "machine_time": 0.50,
        "setup": 0.15,
        "tooling": 0.00,
        "raw_material": 0.30,
        "finishing": 0.05,
    },
    "laser_cutting": {
        "machine_time": 0.35,
        "setup": 0.15,
        "tooling": 0.00,
        "raw_material": 0.40,
        "finishing": 0.10,
    },
}

DEFAULT_RATIOS = {
    "machine_time": 0.40,
    "setup": 0.15,
    "tooling": 0.15,
    "raw_material": 0.20,
    "finishing": 0.10,
}


class CostLabelBuilder:
    """
    Ensures each row has a complete set of cost component labels.
    If component values are null, they are built using heuristic ratios
    applied to cost_total and flagged as 'estimated' or 'decomposed'.
    """

    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes invoice cost totals into components where necessary.
        """
        if df.empty:
            return df

        df = df.copy()

        # Check required columns
        if "cost_total" not in df.columns:
            logger.error("cost_total column is missing. Cannot build labels.")
            return df

        for component in ["cost_machine_time", "cost_setup", "cost_tooling", "cost_raw_material", "cost_finishing"]:
            if component not in df.columns:
                df[component] = None

        if "cost_label_source" not in df.columns:
            df["cost_label_source"] = "estimated"

        # Apply decomposition row by row
        for idx, row in df.iterrows():
            total = float(row["cost_total"])
            if pd.isna(total) or total <= 0:
                continue

            # Check if all component costs are already populated
            components_exist = all(
                pd.notna(row[c]) and float(row[c]) >= 0
                for c in ["cost_machine_time", "cost_setup", "cost_tooling", "cost_raw_material", "cost_finishing"]
            )

            if components_exist:
                # If they sum to total, we tag as exact
                df.at[idx, "cost_label_source"] = "exact"
                continue

            # Otherwise, decompose total
            proc = row.get("process_code", "")
            ratios = DECOMPOSITION_RATIOS.get(proc, DEFAULT_RATIOS)

            df.at[idx, "cost_machine_time"] = round(total * ratios["machine_time"], 2)
            df.at[idx, "cost_setup"] = round(total * ratios["setup"], 2)
            df.at[idx, "cost_tooling"] = round(total * ratios["tooling"], 2)
            df.at[idx, "cost_raw_material"] = round(total * ratios["raw_material"], 2)
            df.at[idx, "cost_finishing"] = round(total * ratios["finishing"], 2)
            df.at[idx, "cost_label_source"] = "estimated"

        logger.info("Cost labels built successfully")
        return df
