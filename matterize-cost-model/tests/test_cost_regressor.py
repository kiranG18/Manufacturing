# tests/test_cost_regressor.py
import os
import pytest
from src.models.cost_regressor import CostDecompositionModel

# Use the actual trained models from outputs/models/ for integration testing
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "models"
)

def test_cost_regressor_load_not_found():
    with pytest.raises(FileNotFoundError):
        CostDecompositionModel.load("nonexistent_directory_for_sure")

def test_cost_regressor_predict_cnc_milling():
    # Verify we can load the models we trained
    if not os.path.isdir(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
        pytest.skip("No trained models found to test. Skipping integration test.")
        
    model = CostDecompositionModel.load(MODELS_DIR)
    
    # Check that cnc_milling is supported
    assert "cnc_milling" in model.supported_processes()
    
    # Mock features vector
    features = {
        "bounding_box_x": 100.0,
        "bounding_box_y": 100.0,
        "bounding_box_z": 20.0,
        "volume": 200000.0,
        "surface_area": 4000.0,
        "wall_thickness_min": 5.0,
        "wall_thickness_avg": 8.0,
        "aspect_ratio": 5.0,
        "hole_count": 4,
        "hole_diameter_min": 6.0,
        "undercut_flag": 0.0,
        "thin_wall_flag": 0.0,
        "curvature_complexity": 0.1,
        "symmetry_score": 0.8,
        "quantity": 10.0,
        "log_quantity": 2.39,
        "batch_size": 10.0,
        "material_code_encoded": 1,
        "material_price_per_kg": 2.80,
        "surface_finish_ra": 3.2,
        "tolerance_index": 5.0,
        "volume_x_price": 0.56,
        "price_date": "2026-06-11"
    }
    
    breakdown = model.predict("cnc_milling", features)
    
    # Verify keys are present
    assert "machine_time_cost" in breakdown
    assert "setup_cost" in breakdown
    assert "tooling_cost" in breakdown
    assert "raw_material_cost" in breakdown
    assert "finishing_cost" in breakdown
    assert "total_cost" in breakdown
    assert breakdown["confidence"] in ["low", "medium", "high"]
    assert breakdown["price_date"] == "2026-06-11"

def test_cost_regressor_fallback_estimate():
    if not os.path.isdir(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
        pytest.skip("No trained models found to test. Skipping integration test.")
        
    model = CostDecompositionModel.load(MODELS_DIR)
    
    features = {
        "volume": 100000.0,
        "material_price_per_kg": 10.0,
        "quantity": 5,
        "price_date": "2026-06-11"
    }
    
    # Call predict on unsupported process
    breakdown = model.predict("unsupported_extrusion", features)
    
    assert breakdown["confidence"] == "low"
    assert "note" in breakdown
    assert "No trained model for process" in breakdown["note"]
    assert breakdown["total_cost"] > 0
    assert breakdown["price_date"] == "2026-06-11"
