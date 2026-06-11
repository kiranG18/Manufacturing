# src/features/validators.py

import math
from typing import Dict, Optional

from .definitions import FEATURE_REGISTRY


class ValidationError(ValueError):
    """Raised when a feature vector fails validation."""
    pass


class FeatureValidator:
    """
    Validates a feature dict against the ranges and constraints defined in
    FEATURE_REGISTRY. Called by CADFeatureExtractor before returning features
    to downstream consumers.

    Design intent: surface bad inputs as early as possible so the API returns
    a 422 rather than letting garbage flow into the model and producing a
    wrong-but-confident-looking prediction.
    """

    # Features that must be present and non-null — no imputation allowed upstream
    REQUIRED_FEATURES = [
        "bounding_box_x", "bounding_box_y", "bounding_box_z",
        "volume", "surface_area",
        "undercut_flag", "thin_wall_flag",
        "hole_count", "aspect_ratio",
    ]

    def validate(self, features: Dict[str, Optional[float]]) -> None:
        """
        Validates a feature dict. Raises ValidationError with a descriptive
        message if any check fails. Returns None on success.
        """
        self._check_required_present(features)
        self._check_ranges(features)
        self._check_derived_consistency(features)

    def _check_required_present(self, features: Dict) -> None:
        for name in self.REQUIRED_FEATURES:
            val = features.get(name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                raise ValidationError(
                    f"Required feature '{name}' is missing or NaN. "
                    f"Check the upstream geometry parser output."
                )

    def _check_ranges(self, features: Dict) -> None:
        for name, spec in FEATURE_REGISTRY.items():
            val = features.get(name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue  # optional features may be NaN; required ones caught above

            lo, hi = spec["range"]
            if not (lo <= val <= hi):
                raise ValidationError(
                    f"Feature '{name}' = {val:.4g} is outside expected range "
                    f"[{lo}, {hi}] ({spec['unit']}). "
                    f"This part may have a malformed geometry export."
                )

    def _check_derived_consistency(self, features: Dict) -> None:
        # thin_wall_flag should agree with wall_thickness_min if both present
        wt_min = features.get("wall_thickness_min")
        flag = features.get("thin_wall_flag")
        if wt_min is not None and flag is not None:
            if not math.isnan(wt_min):
                expected_flag = int(wt_min < 1.5)
                if int(flag) != expected_flag:
                    raise ValidationError(
                        f"Inconsistency: thin_wall_flag={int(flag)} but "
                        f"wall_thickness_min={wt_min:.2f} mm "
                        f"(threshold is 1.5 mm). Recheck feature extraction."
                    )

        # aspect_ratio must be >= 1 (longest / shortest)
        ar = features.get("aspect_ratio")
        if ar is not None and ar < 1.0:
            raise ValidationError(
                f"aspect_ratio={ar:.4g} is less than 1.0, which is geometrically "
                f"impossible. Check bounding box dimension ordering."
            )

        # volume should be consistent with bounding box (can't exceed box volume)
        vol = features.get("volume")
        bbx = features.get("bounding_box_x")
        bby = features.get("bounding_box_y")
        bbz = features.get("bounding_box_z")
        if vol and bbx and bby and bbz:
            box_vol = bbx * bby * bbz
            if vol > box_vol * 1.05:  # 5% tolerance for mesh approximation error
                raise ValidationError(
                    f"Part volume ({vol:.0f} mm³) exceeds bounding box volume "
                    f"({box_vol:.0f} mm³). This indicates a geometry parsing error."
                )
