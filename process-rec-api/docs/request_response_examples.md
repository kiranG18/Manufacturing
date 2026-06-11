# Request and Response Examples

This document provides complete, copy-pasteable JSON payloads showing typical interaction flows with the Process Recommendation API.

## 1. Recommend Process and Cost (CNC Milling)

### Request: `POST /v1/recommend`
```json
{
  "part_id": "PART-2026-MILL-01",
  "geometry": {
    "bounding_box": {"x": 100.0, "y": 80.0, "z": 20.0},
    "volume_mm3": 120000.0,
    "surface_area_mm2": 25000.0,
    "wall_thicknesses": [2.5, 3.0, 2.8],
    "holes": [{"diameter": 6.0, "depth": 15.0}],
    "has_undercuts": false,
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
```

### Response
```json
{
  "part_id": "PART-2026-MILL-01",
  "recommendations": [
    {
      "rank": 1,
      "process_code": "cnc_milling",
      "process_name": "CNC Milling",
      "probability": 0.72,
      "confidence": "high",
      "cost_estimate": {
        "machine_time_cost": 42.10,
        "setup_cost": 18.50,
        "tooling_cost": 9.20,
        "raw_material_cost": 24.80,
        "finishing_cost": 6.40,
        "total_cost": 101.00,
        "confidence": "high",
        "price_date": "2026-06-11"
      }
    },
    {
      "rank": 2,
      "process_code": "sheet_metal_bending",
      "process_name": "Sheet Metal Bending",
      "probability": 0.18,
      "confidence": "medium",
      "cost_estimate": {
        "machine_time_cost": 28.30,
        "setup_cost": 22.00,
        "tooling_cost": 14.50,
        "raw_material_cost": 19.20,
        "finishing_cost": 5.10,
        "total_cost": 89.10,
        "confidence": "high",
        "price_date": "2026-06-11"
      }
    }
  ],
  "model_version": "classifier_v2_cost_v2",
  "latency_ms": 15.2,
  "constraint_filtered_count": 1
}
```

---

## 2. Explain Predictions (SHAP Debug)

### Request: `POST /v1/debug/explain`
(Same payload as above)

### Response
```json
{
  "part_id": "PART-2026-MILL-01",
  "predicted_process": "cnc_milling",
  "probability": 0.72,
  "feature_contributions": [
    {
      "feature_name": "undercut_flag",
      "value": 0.0,
      "shap_contribution": 0.35,
      "direction": "positive"
    },
    {
      "feature_name": "wall_thickness_min",
      "value": 2.5,
      "shap_contribution": 0.22,
      "direction": "positive"
    },
    {
      "feature_name": "hole_count",
      "value": 1.0,
      "shap_contribution": 0.12,
      "direction": "positive"
    }
  ],
  "top_features_summary": "The recommendation is primarily driven by undercut_flag (value=0.0) which had a positive effect, followed by wall_thickness_min (value=2.5) with a positive effect.",
  "model_version": "classifier_v2"
}
```
