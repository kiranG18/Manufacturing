# API Reference Manual

This document details the REST endpoints exposed by the Matterize Process Recommendation API.

## Core Endpoints

### 1. GET `/health`
Returns the status and health metrics of the service.

- **Request Headers:** None
- **Response Format:** JSON

#### Response Example
```json
{
  "status": "ok",
  "models_loaded": true,
  "price_feed_fresh": true,
  "price_age_days": 1,
  "classifier_version": "v2",
  "cost_model_version": "v2.1",
  "uptime_seconds": 84312
}
```

---

### 2. POST `/v1/recommend`
Generates process recommendations and itemised cost estimations for a CAD part.

- **Content-Type:** `application/json`
- **Request Parameters:**
  - `part_id` (String): Unique identifier.
  - `geometry` (Object): Bounding box, volume, surface area, thickness, holes.
  - `requirements` (Object): Material, quantity, surface preset, tolerance band.
  - `top_k` (Integer, default 3): Number of recommendations.

#### Response Example
```json
{
  "part_id": "PART-20241114-001",
  "recommendations": [
    {
      "rank": 1,
      "process_code": "cnc_milling",
      "process_name": "CNC Milling",
      "confidence": "high",
      "probability": 0.72,
      "cost_estimate": {
        "machine_time_cost": 42.10,
        "setup_cost": 18.50,
        "tooling_cost": 9.20,
        "raw_material_cost": 24.80,
        "finishing_cost": 6.40,
        "total_cost": 101.00,
        "confidence": "high",
        "price_date": "2024-11-13"
      }
    }
  ],
  "model_version": "classifier_v2_cost_v2",
  "latency_ms": 18.4,
  "constraint_filtered_count": 0
}
```

---

### 3. POST `/v1/debug/explain`
Returns SHAP values explaining the recommendation for debugging and audit.

- **Content-Type:** `application/json`
- **Request Payload:** Same as `/v1/recommend`

#### Response Example
```json
{
  "part_id": "PART-SAMPLE-001",
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
      "value": 2.1,
      "shap_contribution": 0.22,
      "direction": "positive"
    }
  ],
  "top_features_summary": "The recommendation is primarily driven by undercut_flag (value=0.0) which had a positive effect...",
  "model_version": "classifier_v2"
}
```
