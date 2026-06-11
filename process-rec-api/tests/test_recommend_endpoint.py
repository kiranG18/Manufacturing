# tests/test_recommend_endpoint.py
import os
import sys

# Resolve paths to allow loading models from neighbor repos
os.environ["CLASSIFIER_MODEL_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "cad-process-recommender",
    "outputs",
    "model_v2.pkl"
)
os.environ["COST_MODELS_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "matterize-cost-model",
    "outputs",
    "models"
)

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    # lifespan executes here to load the models
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["classifier_ready"] is True
    assert data["cost_model_ready"] is True

def test_recommend_endpoint(client):
    payload = {
        "part_id": "PART-E2E-TEST",
        "geometry": {
            "bounding_box": {"x": 100.0, "y": 80.0, "z": 20.0},
            "volume_mm3": 120000.0,
            "surface_area_mm2": 25000.0,
            "wall_thicknesses": [2.5, 3.0, 2.8],
            "holes": [{"diameter": 6.0, "depth": 15.0}],
            "has_undercuts": False,
            "curvature_samples": [0.01, 0.02],
            "symmetry_axes": {"x": 0.5, "y": 0.5, "z": 0.5},
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
    
    response = client.post("/v1/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["part_id"] == "PART-E2E-TEST"
    assert len(data["recommendations"]) > 0
    
    # Check that recommendations are ordered and have cost estimates
    rec1 = data["recommendations"][0]
    assert rec1["rank"] == 1
    assert "process_code" in rec1
    assert "probability" in rec1
    
    # Cost breakdown should be present since cost models are loaded
    cost = rec1["cost_estimate"]
    assert cost is not None
    assert "total_cost" in cost
    assert "machine_time_cost" in cost

def test_explain_endpoint(client):
    payload = {
        "part_id": "PART-E2E-TEST",
        "geometry": {
            "bounding_box": {"x": 100.0, "y": 80.0, "z": 20.0},
            "volume_mm3": 120000.0,
            "surface_area_mm2": 25000.0,
            "wall_thicknesses": [2.5, 3.0, 2.8],
            "holes": [{"diameter": 6.0, "depth": 15.0}],
            "has_undercuts": False,
            "curvature_samples": [0.01, 0.02],
            "symmetry_axes": {"x": 0.5, "y": 0.5, "z": 0.5},
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
    response = client.post("/v1/debug/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["part_id"] == "PART-E2E-TEST"
    assert "predicted_process" in data
    assert "feature_contributions" in data
    assert len(data["feature_contributions"]) > 0
