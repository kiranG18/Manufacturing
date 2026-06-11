# src/transforms/normalizer.py
import logging
import os
import math
import yaml
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class DatasetNormalizer:
    """
    Cleans, imputes, encodes, and normalizes the joined manufacturing data.
    """

    def __init__(self, schema_config_path: str = "configs/schema_v2.yaml"):
        self.schema_config_path = schema_config_path
        self._load_schema()

    def _load_schema(self):
        if not os.path.exists(self.schema_config_path):
            # Try absolute path from workspace root
            root_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs", "schema_v2.yaml"
            )
            if os.path.exists(root_path):
                self.schema_config_path = root_path

        if os.path.exists(self.schema_config_path):
            with open(self.schema_config_path, "r") as f:
                schema = yaml.safe_load(f)
                self.processes = schema.get("processes", {})
                self.materials = schema.get("materials", {})
        else:
            logger.warning(f"Schema config path {self.schema_config_path} not found. Using defaults.")
            self.processes = {}
            self.materials = {}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and clean a dataset DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()

        # 1. Clean strings
        if "process_code" in df.columns:
            df["process_code"] = df["process_code"].astype(str).str.strip().str.lower()
        if "material_code" in df.columns:
            df["material_code"] = df["material_code"].astype(str).str.strip().str.upper()
        if "tolerance_band" in df.columns:
            df["tolerance_band"] = df["tolerance_band"].astype(str).str.strip().str.upper()

        # 2. Impute null values using specified strategies
        # Impute wall thickness with process-class median
        for col in ["wall_thickness_min", "wall_thickness_avg"]:
            if col in df.columns:
                imputed_flag_col = f"{col}_imputed"
                df[imputed_flag_col] = df[col].isna().astype(int)
                
                # Group by process_code to get median
                if "process_code" in df.columns:
                    medians = df.groupby("process_code")[col].transform("median")
                    # If some processes are entirely NaN, fill with overall median
                    overall_median = df[col].median()
                    if pd.isna(overall_median):
                        overall_median = 1.5  # fallback
                    medians = medians.fillna(overall_median)
                    df[col] = df[col].fillna(medians)
                else:
                    df[col] = df[col].fillna(df[col].median().fillna(1.5))

        # Impute hole_count with default 0
        if "hole_count" in df.columns:
            df["hole_count"] = df["hole_count"].fillna(0).astype(int)

        # Impute hole_diameter_min with column mean or default
        if "hole_diameter_min" in df.columns:
            mean_diam = df["hole_diameter_min"].mean()
            if pd.isna(mean_diam):
                mean_diam = 5.0
            df["hole_diameter_min"] = df["hole_diameter_min"].fillna(mean_diam)

        # Impute undercut_flag with default 0
        if "undercut_flag" in df.columns:
            df["undercut_flag"] = df["undercut_flag"].fillna(0).astype(int)

        # Impute symmetry_score with default 0.5
        if "symmetry_score" in df.columns:
            df["symmetry_score"] = df["symmetry_score"].fillna(0.5)

        # Impute curvature_complexity with default 0.0
        if "curvature_complexity" in df.columns:
            df["curvature_complexity"] = df["curvature_complexity"].fillna(0.0)

        # Impute cost_label_source with default 'estimated'
        if "cost_label_source" in df.columns:
            df["cost_label_source"] = df["cost_label_source"].fillna("estimated")

        # 3. Categorical encoding
        if "process_code" in df.columns:
            df["process_code_encoded"] = df["process_code"].map(self.processes).fillna(0).astype(int)
        if "material_code" in df.columns:
            df["material_code_encoded"] = df["material_code"].map(self.materials).fillna(0).astype(int)

        # 4. derived interaction features
        if "quantity" in df.columns:
            df["log_quantity"] = np.log1p(df["quantity"].astype(float))
        
        if "quantity" in df.columns and "batch_size" not in df.columns:
            df["batch_size"] = df["quantity"]
        elif "batch_size" in df.columns:
            df["batch_size"] = df["batch_size"].fillna(df["quantity"] if "quantity" in df.columns else 1.0)

        # Volume x Price
        if "volume" in df.columns and "material_price_per_kg" in df.columns:
            df["volume_x_price"] = df["volume"] * df["material_price_per_kg"] / 1e6
        else:
            df["volume_x_price"] = 0.0

        # surface_finish_ra (convert text specs like 'fine' to microns if needed, otherwise read number)
        SURFACE_FINISH_DEFAULTS = {
            "as_machined": 3.2,
            "fine": 1.6,
            "very_fine": 0.8,
            "mirror": 0.1,
        }
        if "surface_finish_ra" in df.columns:
            # If surface_finish_ra is stored as text in the DB, map it
            # Otherwise keep float
            def convert_ra(val):
                if isinstance(val, str):
                    val_clean = val.strip().lower()
                    if val_clean in SURFACE_FINISH_DEFAULTS:
                        return SURFACE_FINISH_DEFAULTS[val_clean]
                    try:
                        return float(val)
                    except ValueError:
                        return 3.2
                return float(val) if val is not None else 3.2
            df["surface_finish_ra"] = df["surface_finish_ra"].apply(convert_ra)
        else:
            df["surface_finish_ra"] = 3.2

        # tolerance_index (map IT grade to index, e.g. IT8 -> 5)
        TOLERANCE_IT_INDEX = {
            "IT4": 1, "IT5": 2, "IT6": 3, "IT7": 4,
            "IT8": 5, "IT9": 6, "IT10": 7, "IT11": 8, "IT12": 9,
        }
        if "tolerance_band" in df.columns:
            df["tolerance_index"] = df["tolerance_band"].map(TOLERANCE_IT_INDEX).fillna(5).astype(float)
        else:
            df["tolerance_index"] = 5.0

        # thin_wall_flag derivation (walls < 1.5 mm)
        if "wall_thickness_min" in df.columns:
            df["thin_wall_flag"] = (df["wall_thickness_min"] < 1.5).astype(int)

        logger.info(f"Normalized dataset DataFrame. Columns present: {list(df.columns)}")
        return df
