# tests/test_extractor.py
#
# Unit tests for CADFeatureExtractor.
# Covers extraction correctness, edge cases, and reference part validation.

import math
import pytest
from src.features.extractor import CADFeatureExtractor, GeometryMetadata


@pytest.fixture
def extractor():
    return CADFeatureExtractor(validate=True)


@pytest.fixture
def standard_bracket_geo():
    """A straightforward aluminium bracket: flat, 3 holes, no undercuts."""
    return GeometryMetadata(
        bounding_box={"x": 85.0, "y": 60.0, "z": 20.0},
        volume_mm3=54200.0,
        surface_area_mm2=22800.0,
        wall_thicknesses=[3.2, 3.4, 3.0, 3.6, 3.1],
        holes=[
            {"diameter": 5.0, "depth": 20.0},
            {"diameter": 5.0, "depth": 20.0},
            {"diameter": 8.0, "depth": 12.0},
        ],
        has_undercuts=False,
        curvature_samples=[0.02, 0.01, 0.03, 0.02, 0.015],
        symmetry_axes={"x": 0.28, "y": 0.35, "z": 0.12},
        source_format="STEP",
        part_id="REF-001",
    )


@pytest.fixture
def shaft_geo():
    """Cylindrical shaft — should score high symmetry, low hole count."""
    return GeometryMetadata(
        bounding_box={"x": 180.0, "y": 22.0, "z": 22.0},
        volume_mm3=61070.0,
        surface_area_mm2=14800.0,
        wall_thicknesses=[6.0, 6.2, 5.9, 6.1],
        holes=[],
        has_undercuts=False,
        curvature_samples=[0.04, 0.03, 0.05, 0.03],
        symmetry_axes={"x": 0.91, "y": 0.12, "z": 0.12},
        source_format="STEP",
        part_id="REF-002",
    )


class TestFeatureExtraction:
    def test_returns_all_13_features(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        expected = [
            "bounding_box_x", "bounding_box_y", "bounding_box_z",
            "volume", "surface_area",
            "wall_thickness_min", "wall_thickness_avg",
            "aspect_ratio", "hole_count", "hole_diameter_min",
            "undercut_flag", "thin_wall_flag",
            "curvature_complexity", "symmetry_score",
        ]
        for feat in expected:
            assert feat in features, f"Missing feature: {feat}"

    def test_bounding_box_passthrough(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["bounding_box_x"] == pytest.approx(85.0)
        assert features["bounding_box_y"] == pytest.approx(60.0)
        assert features["bounding_box_z"] == pytest.approx(20.0)

    def test_wall_thickness_min_computed_correctly(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["wall_thickness_min"] == pytest.approx(3.0)

    def test_wall_thickness_avg_computed_correctly(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        expected_avg = (3.2 + 3.4 + 3.0 + 3.6 + 3.1) / 5
        assert features["wall_thickness_avg"] == pytest.approx(expected_avg)

    def test_hole_count(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["hole_count"] == 3

    def test_hole_diameter_min(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["hole_diameter_min"] == pytest.approx(5.0)

    def test_no_holes_sets_count_to_zero(self, extractor, shaft_geo):
        features = extractor.extract(shaft_geo)
        assert features["hole_count"] == 0

    def test_no_holes_sets_diameter_to_none(self, extractor, shaft_geo):
        features = extractor.extract(shaft_geo)
        assert features["hole_diameter_min"] is None

    def test_undercut_flag_false(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["undercut_flag"] == 0

    def test_undercut_flag_true(self, extractor, standard_bracket_geo):
        standard_bracket_geo.has_undercuts = True
        features = extractor.extract(standard_bracket_geo)
        assert features["undercut_flag"] == 1

    def test_thin_wall_flag_not_set_for_thick_walls(self, extractor, standard_bracket_geo):
        # wall_thickness_min = 3.0, threshold is 1.5
        features = extractor.extract(standard_bracket_geo)
        assert features["thin_wall_flag"] == 0

    def test_thin_wall_flag_set_for_thin_walls(self, extractor):
        geo = GeometryMetadata(
            bounding_box={"x": 200.0, "y": 150.0, "z": 1.2},
            volume_mm3=28000.0,
            surface_area_mm2=62400.0,
            wall_thicknesses=[1.2, 1.1, 1.3, 1.2],
            holes=[],
            has_undercuts=False,
            curvature_samples=[0.01, 0.01],
            symmetry_axes={"x": 0.3, "y": 0.4, "z": 0.1},
            source_format="STEP",
        )
        features = extractor.extract(geo)
        assert features["thin_wall_flag"] == 1

    def test_aspect_ratio_is_at_least_one(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert features["aspect_ratio"] >= 1.0

    def test_shaft_has_high_symmetry_score(self, extractor, shaft_geo):
        features = extractor.extract(shaft_geo)
        assert features["symmetry_score"] >= 0.85

    def test_curvature_complexity_in_range(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        assert 0.0 <= features["curvature_complexity"] <= 1.0

    def test_to_array_returns_correct_length(self, extractor, standard_bracket_geo):
        features = extractor.extract(standard_bracket_geo)
        arr = extractor.to_array(features)
        from src.features.definitions import FEATURE_COLUMNS
        assert len(arr) == len(FEATURE_COLUMNS)

    def test_empty_curvature_samples_returns_zero(self, extractor):
        geo = GeometryMetadata(
            bounding_box={"x": 50.0, "y": 40.0, "z": 30.0},
            volume_mm3=30000.0,
            surface_area_mm2=14800.0,
            wall_thicknesses=[5.0],
            holes=[],
            has_undercuts=False,
            curvature_samples=[],
            symmetry_axes={},
            source_format="STL",
        )
        features = extractor.extract(geo)
        assert features["curvature_complexity"] == 0.0
        assert features["symmetry_score"] == 0.0


class TestBatchExtraction:
    def test_batch_returns_list(self, extractor, standard_bracket_geo, shaft_geo):
        results = extractor.extract_batch([standard_bracket_geo, shaft_geo])
        assert len(results) == 2
        assert isinstance(results[0], dict)
        assert isinstance(results[1], dict)
