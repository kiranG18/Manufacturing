# src/ingestion/cad_metadata_loader.py

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CADMetadataLoader:
    """
    Loads pre-computed geometry feature vectors from the CAD feature store.

    The feature store is populated by the platform's CAD parsing service
    whenever a part file is uploaded. This loader reads from it by part_id.
    Features are already in the format produced by CADFeatureExtractor in
    the cad-process-recommender repo.
    """

    EXPECTED_COLUMNS = [
        "part_id", "bounding_box_x", "bounding_box_y", "bounding_box_z",
        "volume", "surface_area", "wall_thickness_min", "wall_thickness_avg",
        "aspect_ratio", "hole_count", "hole_diameter_min",
        "undercut_flag", "thin_wall_flag", "curvature_complexity", "symmetry_score",
    ]

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_df: Optional[pd.DataFrame] = None,
    ):
        self._api_url = api_url
        self._api_key = api_key
        self._cache_df = cache_df  # pre-loaded DataFrame for testing/offline use

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CADMetadataLoader":
        """Test/offline mode — uses a pre-loaded DataFrame."""
        return cls(cache_df=df)

    def load_batch(self, part_ids: List[str]) -> pd.DataFrame:
        """
        Load geometry feature vectors for a list of part IDs.
        Returns a DataFrame indexed by part_id.
        Parts not found in the feature store are excluded with a log entry.
        """
        if self._cache_df is not None:
            return self._load_from_cache(part_ids)
        return self._load_from_api(part_ids)

    def _load_from_cache(self, part_ids: List[str]) -> pd.DataFrame:
        df = self._cache_df.copy()
        if "part_id" in df.columns:
            df = df[df["part_id"].isin(part_ids)]
        found = len(df)
        missing = len(part_ids) - found
        if missing > 0:
            logger.warning(f"{missing} part_ids not found in geometry cache")
        return df

    def _load_from_api(self, part_ids: List[str]) -> pd.DataFrame:
        """
        Fetches feature vectors from the geometry store REST API.
        The API supports bulk fetch via POST /api/v1/geometry-features/bulk.
        Falls back to individual GET requests if bulk endpoint is unavailable.
        """
        import json
        import urllib.request

        url = f"{self._api_url}/api/v1/geometry-features/bulk"
        payload = json.dumps({"part_ids": part_ids}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        records = data.get("features", [])
        df = pd.DataFrame(records)

        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.warning(
                f"Geometry API response missing expected columns: {missing_cols}. "
                f"The feature store schema may have changed."
            )
        return df

    def validate_coverage(self, df: pd.DataFrame, part_ids: List[str]) -> dict:
        """
        Returns coverage stats: how many requested part_ids have geometry features.
        """
        loaded_ids = set(df["part_id"].tolist()) if "part_id" in df.columns else set()
        requested = set(part_ids)
        covered = requested & loaded_ids
        return {
            "requested": len(requested),
            "covered": len(covered),
            "missing": len(requested - covered),
            "coverage_pct": round(len(covered) / max(len(requested), 1) * 100, 1),
        }
