# tests/test_validators.py
#
# Unit tests for FeatureValidator.
# Exercises all validation rules: required field checks, range checks,
# and cross-feature consistency checks.

import math
import pytest
from src.features.validators import FeatureValidator, ValidationError


@pytest.fixture
def validator():
    return FeatureValidator()


@pytest.fixture
def valid_features():
    """A geometrically consistent feature set that should pass all checks."""
    return {
        "bounding_box_x": 85.0,
        "bounding_box_y": 60.0,
        "bounding_box_z": 20.0,
        "volume": 54200.0,
        "surface_area": 22800.0,
        "wall_thickness_min": 3.2,
        "wall_thickness_avg": 3.8,
        "aspect_ratio": 4.25,
        "hole_count": 3.0,
        "hole_diameter_min": 5.0,
        "undercut_flag": 0.0,
        "thin_wall_flag": 0.0,
        "curvature_complexity": 0.08,
        "symmetry_score": 0.32,
    }


class TestRequiredFieldChecks:
    def test_valid_features_passes(self, validator, valid_features):
        validator.validate(valid_features)  # should not raise

    def test_missing_volume_raises(self, validator, valid_features):
        del valid_features["volume"]
        with pytest.raises(ValidationError, match="volume"):
            validator.validate(valid_features)

    def test_nan_volume_raises(self, validator, valid_features):
        valid_features["volume"] = float("nan")
        with pytest.raises(ValidationError, match="volume"):
            validator.validate(valid_features)

    def test_missing_bounding_box_x_raises(self, validator, valid_features):
        del valid_features["bounding_box_x"]
        with pytest.raises(ValidationError, match="bounding_box_x"):
            validator.validate(valid_features)

    def test_missing_undercut_flag_raises(self, validator, valid_features):
        del valid_features["undercut_flag"]
        with pytest.raises(ValidationError, match="undercut_flag"):
            validator.validate(valid_features)

    def test_missing_aspect_ratio_raises(self, validator, valid_features):
        del valid_features["aspect_ratio"]
        with pytest.raises(ValidationError, match="aspect_ratio"):
            validator.validate(valid_features)


class TestRangeChecks:
    def test_negative_wall_thickness_raises(self, validator, valid_features):
        valid_features["wall_thickness_min"] = -0.5
        with pytest.raises(ValidationError, match="wall_thickness_min"):
            validator.validate(valid_features)

    def test_zero_wall_thickness_raises(self, validator, valid_features):
        valid_features["wall_thickness_min"] = 0.0
        with pytest.raises(ValidationError, match="wall_thickness_min"):
            validator.validate(valid_features)

    def test_wall_thickness_at_lower_bound_passes(self, validator, valid_features):
        valid_features["wall_thickness_min"] = 0.1
        valid_features["thin_wall_flag"] = 1
        validator.validate(valid_features)

    def test_curvature_above_one_raises(self, validator, valid_features):
        valid_features["curvature_complexity"] = 1.5
        with pytest.raises(ValidationError, match="curvature_complexity"):
            validator.validate(valid_features)

    def test_curvature_below_zero_raises(self, validator, valid_features):
        valid_features["curvature_complexity"] = -0.1
        with pytest.raises(ValidationError, match="curvature_complexity"):
            validator.validate(valid_features)

    def test_extremely_large_bounding_box_raises(self, validator, valid_features):
        valid_features["bounding_box_x"] = 5000.0
        with pytest.raises(ValidationError, match="bounding_box_x"):
            validator.validate(valid_features)

    def test_negative_hole_count_raises(self, validator, valid_features):
        valid_features["hole_count"] = -1.0
        with pytest.raises(ValidationError, match="hole_count"):
            validator.validate(valid_features)


class TestDerivedConsistency:
    def test_thin_wall_flag_inconsistency_raises(self, validator, valid_features):
        # wall_thickness_min=3.2 but thin_wall_flag=1 (should be 0)
        valid_features["thin_wall_flag"] = 1.0
        with pytest.raises(ValidationError, match="thin_wall_flag"):
            validator.validate(valid_features)

    def test_thin_wall_flag_consistent_thin_wall(self, validator, valid_features):
        # wall_thickness_min=0.9, flag=1 — should pass
        valid_features["wall_thickness_min"] = 0.9
        valid_features["wall_thickness_avg"] = 1.2
        valid_features["thin_wall_flag"] = 1.0
        validator.validate(valid_features)

    def test_aspect_ratio_below_one_raises(self, validator, valid_features):
        valid_features["aspect_ratio"] = 0.8
        with pytest.raises(ValidationError, match="aspect_ratio"):
            validator.validate(valid_features)

    def test_volume_exceeding_bounding_box_raises(self, validator, valid_features):
        # 85*60*20=102000 mm³ — set volume way beyond this
        valid_features["volume"] = 200000.0
        with pytest.raises(ValidationError, match="volume"):
            validator.validate(valid_features)

    def test_volume_slightly_above_box_with_tolerance_passes(self, validator, valid_features):
        # box = 85*60*20 = 102000; volume with 4% tolerance should be fine
        valid_features["volume"] = 102000.0 * 1.04
        valid_features["thin_wall_flag"] = 0
        validator.validate(valid_features)


class TestOptionalFeatures:
    def test_missing_optional_hole_diameter_min_does_not_raise(self, validator, valid_features):
        valid_features["hole_diameter_min"] = None
        valid_features["hole_count"] = 0.0
        validator.validate(valid_features)

    def test_nan_optional_feature_does_not_raise(self, validator, valid_features):
        valid_features["hole_diameter_min"] = float("nan")
        validator.validate(valid_features)
