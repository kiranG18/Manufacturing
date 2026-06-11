# tests/test_schema_validator.py
import pandas as pd
import pytest
from src.validation.schema_validator import SchemaValidator

@pytest.fixture
def sample_test_df():
    return pd.DataFrame([
        {
            # Good Row
            "part_id": "PART-001",
            "bounding_box_x": 100.0,
            "bounding_box_y": 100.0,
            "bounding_box_z": 20.0,
            "volume": 20000.0,
            "surface_area": 4000.0,
            "wall_thickness_min": 2.0,
            "wall_thickness_avg": 3.0,
            "aspect_ratio": 2.0,
            "hole_count": 0,
            "hole_diameter_min": 5.0,
            "undercut_flag": 0,
            "thin_wall_flag": 0,
            "curvature_complexity": 0.0,
            "symmetry_score": 0.8,
            "process_code": "cnc_milling",
            "material_code": "AL6061",
            "quantity": 10,
            "material_price_per_kg": 2.80,
            "cost_machine_time": 100.0,
            "cost_setup": 50.0,
            "cost_tooling": 0.0,
            "cost_raw_material": 30.0,
            "cost_finishing": 10.0,
            "cost_total": 190.0,
            "cost_label_source": "exact"
        },
        {
            # Bad Row: cost_total out of range (negative)
            "part_id": "PART-002",
            "bounding_box_x": 100.0,
            "bounding_box_y": 100.0,
            "bounding_box_z": 20.0,
            "volume": 20000.0,
            "surface_area": 4000.0,
            "wall_thickness_min": 2.0,
            "wall_thickness_avg": 3.0,
            "aspect_ratio": 2.0,
            "hole_count": 0,
            "hole_diameter_min": 5.0,
            "undercut_flag": 0,
            "thin_wall_flag": 0,
            "curvature_complexity": 0.0,
            "symmetry_score": 0.8,
            "process_code": "cnc_milling",
            "material_code": "AL6061",
            "quantity": 10,
            "material_price_per_kg": 2.80,
            "cost_machine_time": 100.0,
            "cost_setup": 50.0,
            "cost_tooling": 0.0,
            "cost_raw_material": 30.0,
            "cost_finishing": 10.0,
            "cost_total": -5.0,  # negative total cost is invalid
            "cost_label_source": "exact"
        },
        {
            # Bad Row: process_code is unallowed value
            "part_id": "PART-003",
            "bounding_box_x": 100.0,
            "bounding_box_y": 100.0,
            "bounding_box_z": 20.0,
            "volume": 20000.0,
            "surface_area": 4000.0,
            "wall_thickness_min": 2.0,
            "wall_thickness_avg": 3.0,
            "aspect_ratio": 2.0,
            "hole_count": 0,
            "hole_diameter_min": 5.0,
            "undercut_flag": 0,
            "thin_wall_flag": 0,
            "curvature_complexity": 0.0,
            "symmetry_score": 0.8,
            "process_code": "weaving_socks",  # invalid process
            "material_code": "AL6061",
            "quantity": 10,
            "material_price_per_kg": 2.80,
            "cost_machine_time": 100.0,
            "cost_setup": 50.0,
            "cost_tooling": 0.0,
            "cost_raw_material": 30.0,
            "cost_finishing": 10.0,
            "cost_total": 190.0,
            "cost_label_source": "exact"
        }
    ])

def test_schema_validator_filtering(sample_test_df):
    validator = SchemaValidator()
    
    clean_df, metrics = validator.validate(sample_test_df)
    
    # Pre-validation size should be 3
    assert metrics["total_rows_pre_validation"] == 3
    # Post-validation size should be 1 (only PART-001 is clean)
    assert metrics["total_rows_post_validation"] == 1
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["part_id"] == "PART-001"
    
    # Dropped details should explain reasons
    assert metrics["dropped_rows_count"] == 2
    reasons = metrics["dropped_reasons_summary"]
    assert any("cost_total" in r for r in reasons)
    assert any("process_code" in r for r in reasons)
