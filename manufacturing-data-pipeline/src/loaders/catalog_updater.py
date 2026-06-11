# src/loaders/catalog_updater.py
import json
import logging
import os
import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CatalogUpdater:
    """
    Maintains a central dataset catalog (dataset_catalog.json) that registers
    each new validated training dataset version along with size and creation details.
    """

    def __init__(self, catalog_path: str = "data/processed/dataset_catalog.json"):
        self.catalog_path = catalog_path

    def update_catalog(
        self,
        version: int,
        csv_path: str,
        row_count: int,
        columns_count: int,
        pass_rate_pct: float,
    ) -> Dict[str, Any]:
        """
        Appends or updates a version entry in the dataset catalog JSON file.
        """
        # Resolve path relative to pipeline root if needed
        pipeline_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        actual_path = self.catalog_path
        if not os.path.isabs(actual_path):
            actual_path = os.path.normpath(os.path.join(pipeline_root, actual_path))

        os.makedirs(os.path.dirname(actual_path) or ".", exist_ok=True)

        catalog = {}
        if os.path.exists(actual_path):
            try:
                with open(actual_path, "r") as f:
                    catalog = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read dataset catalog: {e}. Reinitializing.")

        # Update or add entry
        catalog[str(version)] = {
            "version": version,
            "file_path": csv_path,
            "row_count": row_count,
            "column_count": columns_count,
            "validation_pass_rate": pass_rate_pct,
            "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        # Keep track of which is the "latest" version
        catalog["latest_version"] = version

        # Write catalog back
        with open(actual_path, "w") as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Updated dataset catalog at {actual_path} with version v{version}")
        return catalog
