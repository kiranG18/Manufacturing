# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas.request import RecommendRequest, BoundingBox

def test_valid_bounding_box():
    bb = BoundingBox(x=10.0, y=20.0, z=5.0)
    assert bb.x == 10.0
    assert bb.y == 20.0
    assert bb.z == 5.0

def test_invalid_bounding_box():
    with pytest.raises(ValidationError):
        # negative dimension is invalid
        BoundingBox(x=-1.0, y=20.0, z=5.0)
        
    with pytest.raises(ValidationError):
        # exceeds max limit (le=1000 for z)
        BoundingBox(x=10.0, y=20.0, z=1500.0)

def test_recommend_request_schema_validation():
    payload = {
        "part_id": "PART-123",
        "geometry": {
            "bounding_box": {"x": 50.0, "y": 50.0, "z": 10.0},
            "volume_mm3": 25000.0,
            "surface_area_mm2": 6000.0,
            "wall_thicknesses": [2.5, 3.0],
            "holes": [],
            "has_undercuts": False,
            "curvature_samples": [0.01],
            "symmetry_axes": {"x": 0.5},
            "source_format": "STEP"
        },
        "requirements": {
            "material_code": "AL6061",
            "quantity": 100,
            "surface_finish_preset": "fine",
            "tolerance_band": "IT8"
        },
        "top_k": 3
    }
    
    req = RecommendRequest(**payload)
    assert req.part_id == "PART-123"
    
    # Check flattening helper
    flat = req.to_feature_dict()
    assert flat["bounding_box_x"] == 50.0
    assert flat["volume"] == 25000.0
    assert flat["wall_thickness_min"] == 2.5
    assert flat["quantity"] == 100.0
    assert flat["material_code"] == "AL6061"

def test_invalid_recommend_request_negative_wall_thickness():
    payload = {
        "part_id": "PART-123",
        "geometry": {
            "bounding_box": {"x": 50.0, "y": 50.0, "z": 10.0},
            "volume_mm3": 25000.0,
            "surface_area_mm2": 6000.0,
            "wall_thicknesses": [-1.0, 3.0],  # negative wall thickness is invalid
            "holes": [],
            "has_undercuts": False,
            "source_format": "STEP"
        },
        "requirements": {
            "material_code": "AL6061",
            "quantity": 100
        }
    }
    with pytest.raises(ValidationError):
        RecommendRequest(**payload)
