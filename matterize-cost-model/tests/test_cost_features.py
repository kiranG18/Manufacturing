# tests/test_cost_features.py
import pytest
from src.features.cost_features import CostFeatureBuilder, CostInput, SURFACE_FINISH_DEFAULTS

def test_cost_input_creation():
    cost_in = CostInput(
        process_code="cnc_milling",
        material_code="AL6061",
        quantity=50,
        surface_finish_spec=1.6,
        tolerance_band="IT8"
    )
    assert cost_in.process_code == "cnc_milling"
    assert cost_in.material_code == "AL6061"
    assert cost_in.quantity == 50
    assert cost_in.surface_finish_spec == 1.6
    assert cost_in.tolerance_band == "IT8"
    assert cost_in.batch_size is None

def test_cost_feature_builder_fallback_pricing():
    builder = CostFeatureBuilder(live_prices=False)
    geom_features = {
        "volume": 10000.0,
        "surface_area": 500.0,
        "wall_thickness_min": 2.5
    }
    cost_in = CostInput(
        process_code="cnc_milling",
        material_code="AL6061",
        quantity=100,
        surface_finish_spec=3.2,
        tolerance_band="IT8"
    )
    
    features = builder.build(geom_features, cost_in)
    
    assert features["process_code_encoded"] == 1
    assert features["quantity"] == 100.0
    assert features["batch_size"] == 100.0
    assert features["surface_finish_ra"] == 3.2
    assert features["tolerance_index"] == 5
    assert features["material_code_encoded"] == 1
    assert features["material_price_per_kg"] == 2.80  # historical fallback for AL6061
    assert features["price_staleness_days"] == 999
    assert features["volume_x_price"] == (10000.0 * 2.80 / 1e6)

def test_safe_log():
    builder = CostFeatureBuilder(live_prices=False)
    assert builder._safe_log(0) == 0.0
    assert builder._safe_log(-5.0) == 0.0
    assert builder._safe_log(9.0) > 0.0

def test_encode_process():
    builder = CostFeatureBuilder(live_prices=False)
    assert builder._encode_process("cnc_milling") == 1
    assert builder._encode_process("cnc_turning") == 2
    assert builder._encode_process("nonexistent_process") == 0

def test_encode_material():
    builder = CostFeatureBuilder(live_prices=False)
    assert builder._encode_material("AL6061") == 1
    assert builder._encode_material("SS316L") == 2
    assert builder._encode_material("nonexistent_material") == 0
