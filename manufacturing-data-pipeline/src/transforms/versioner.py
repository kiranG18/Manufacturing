# src/transforms/versioner.py
import datetime
import json
import logging
import os
import re
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DatasetVersioner:
    """
    Handles saving datasets with incremental versioning (e.g., training_data_v1.csv, v2.csv)
    and outputs a companion metadata JSON file for tracking provenance and run statistics.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir

    def get_next_version(self) -> int:
        """
        Scans the output directory to find the highest existing version number
        and returns the next version integer.
        """
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            return 1

        highest_v = 0
        pattern = re.compile(r"training_data_v(\d+)\.csv")
        for fname in os.listdir(self.output_dir):
            match = pattern.match(fname)
            if match:
                v = int(match.group(1))
                if v > highest_v:
                    highest_v = v

        return highest_v + 1

    def save_version(
        self,
        df: pd.DataFrame,
        pipeline_metrics: Dict[str, Any],
        version: int = None,
    ) -> Tuple[str, str]:
        """
        Saves the DataFrame to CSV and writes the companion JSON metadata.
        Returns the absolute paths of (csv_path, json_path).
        """
        if version is None:
            version = self.get_next_version()

        os.makedirs(self.output_dir, exist_ok=True)
        csv_name = f"training_data_v{version}.csv"
        json_name = f"training_data_v{version}_meta.json"

        csv_path = os.path.join(self.output_dir, csv_name)
        json_path = os.path.join(self.output_dir, json_name)

        # Save CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset version v{version} to {csv_path}")

        # Construct metadata
        metadata = {
            "version": version,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "pipeline_run_stats": pipeline_metrics,
        }

        # Save JSON metadata
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved companion metadata to {json_path}")

        return csv_path, json_path
