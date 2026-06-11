# src/features/extractor.py
#
# CADFeatureExtractor
# -------------------
# Converts a parsed CAD geometry object (or its metadata dict) into the
# 13-feature vector consumed by the ProcessClassifier.
#
# In production, CAD files arrive as STEP or STL uploads. The platform's
# ingestion layer pre-parses them into a geometry metadata dict before
# calling this extractor, so this class does not handle raw file I/O directly.
# For local development and testing, a lightweight mock geometry loader is
# provided in tests/fixtures.py.
#
# Design choice: no deep-learning on raw mesh. We extract engineered scalar
# features that correspond to manufacturing constraints a process engineer
# would check manually. This keeps the model interpretable, fast at inference,
# and trainable on the ~4,000 labeled historical parts we had available.

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from .definitions import FEATURE_COLUMNS, HARD_CONSTRAINT_FEATURES
from .validators import FeatureValidator


@dataclass
class GeometryMetadata:
    """
    Parsed representation of a CAD part's geometry.
    In production this is hydrated by the upstream CAD parsing service.
    Fields marked Optional may be absent for low-fidelity uploads (e.g. STL
    files without feature annotations).
    """
    bounding_box: Dict[str, float]       # {"x": mm, "y": mm, "z": mm}
    volume_mm3: float
    surface_area_mm2: float
    wall_thicknesses: list               # Sampled values from ray-cast analysis
    holes: list                          # [{"diameter": mm, "depth": mm}, ...]
    has_undercuts: bool
    curvature_samples: list             # Per-face curvature values (normalized)
    symmetry_axes: Dict[str, float]     # {"x": 0-1, "y": 0-1, "z": 0-1}
    source_format: str                  # "STEP" | "STL" | "IGES"
    part_id: Optional[str] = None


class CADFeatureExtractor:
    """
    Extracts the 13-feature vector from a GeometryMetadata object.

    Usage:
        extractor = CADFeatureExtractor()
        features = extractor.extract(geometry_metadata)
        # returns: {"bounding_box_x": 45.2, "wall_thickness_min": 2.1, ...}

    Features are validated before return. Invalid geometries raise ValueError
    with a descriptive message so the API can return a meaningful error rather
    than silently producing garbage predictions.
    """

    THIN_WALL_THRESHOLD_MM = 1.5    # walls below this trigger thin_wall_flag
    MIN_HOLE_DIAMETER_FALLBACK = None

    def __init__(self, validate: bool = True):
        self.validate = validate
        self._validator = FeatureValidator()

    def extract(self, geo: GeometryMetadata) -> Dict[str, float]:
        """
        Main extraction entry point.
        Returns a flat dict matching FEATURE_COLUMNS ordering.
        """
        features = {}

        # Dimensional
        features["bounding_box_x"] = geo.bounding_box["x"]
        features["bounding_box_y"] = geo.bounding_box["y"]
        features["bounding_box_z"] = geo.bounding_box["z"]
        features["volume"] = geo.volume_mm3
        features["surface_area"] = geo.surface_area_mm2

        # Wall geometry
        wall_t = geo.wall_thicknesses
        features["wall_thickness_min"] = float(np.min(wall_t)) if wall_t else np.nan
        features["wall_thickness_avg"] = float(np.mean(wall_t)) if wall_t else np.nan

        # Shape ratios
        dims = sorted([geo.bounding_box["x"], geo.bounding_box["y"], geo.bounding_box["z"]])
        features["aspect_ratio"] = dims[-1] / dims[0] if dims[0] > 0 else 1.0

        # Hole features
        diameters = [h["diameter"] for h in geo.holes] if geo.holes else []
        features["hole_count"] = len(diameters)
        features["hole_diameter_min"] = (
            float(min(diameters)) if diameters else self.MIN_HOLE_DIAMETER_FALLBACK
        )

        # Binary constraint flags
        features["undercut_flag"] = int(geo.has_undercuts)
        features["thin_wall_flag"] = int(
            features["wall_thickness_min"] < self.THIN_WALL_THRESHOLD_MM
            if not np.isnan(features["wall_thickness_min"])
            else False
        )

        # Complexity scores
        features["curvature_complexity"] = self._compute_curvature_score(geo.curvature_samples)
        features["symmetry_score"] = self._compute_symmetry_score(geo.symmetry_axes)

        if self.validate:
            self._validator.validate(features)

        return features

    def extract_batch(self, geometries: list) -> list:
        """Extract features from a list of GeometryMetadata objects."""
        return [self.extract(g) for g in geometries]

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dict to ordered numpy array matching FEATURE_COLUMNS.
        NaN values are median-imputed by the classifier pipeline, not here.
        """
        return np.array([features.get(col, np.nan) for col in FEATURE_COLUMNS])

    # --- Private helpers ---

    def _compute_curvature_score(self, curvature_samples: list) -> float:
        """
        Normalizes per-face curvature variance to a 0-1 complexity score.
        0 = flat/prismatic (e.g. a simple bracket), 1 = highly organic.
        """
        if not curvature_samples:
            return 0.0
        variance = float(np.var(curvature_samples))
        # Empirical normalization: 99th percentile variance in training set was ~0.85
        return float(np.clip(variance / 0.85, 0.0, 1.0))

    def _compute_symmetry_score(self, symmetry_axes: Dict[str, float]) -> float:
        """
        Returns the max symmetry score across all axes.
        A perfectly cylindrical part scores ~1.0 on the rotational axis.
        """
        if not symmetry_axes:
            return 0.0
        return float(max(symmetry_axes.values()))
