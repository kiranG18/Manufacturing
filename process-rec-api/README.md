# process-rec-api

FastAPI microservice that serves manufacturing process recommendations and cost decomposition estimates to the Matterize Quanta and Optima product interfaces.

Built during an AI/ML internship at Matterize. Wraps the trained models from `cad-process-recommender` and `matterize-cost-model` behind a clean REST interface with input validation, confidence scoring, and latency monitoring.

---

## What this does

When a user uploads a CAD part to Quanta or Optima, the frontend sends a POST request to this API with the part's geometry metadata and any specified requirements. The API:

1. Validates and preprocesses the input
2. Runs the process classifier → returns top-3 recommended processes with confidence scores
3. For each recommended process, runs the cost decomposition model → returns per-component cost breakdown
4. Applies hard constraint filters (e.g. removes injection molding if undercuts present and no side-action specified)
5. Returns a ranked list of process-cost options

---

## Folder structure

```
process-rec-api/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── app/
│   ├── main.py                      # FastAPI app setup, middleware registration, routers
│   ├── config.py                    # Settings from env vars (model paths, log level, etc.)
│   ├── routes/
│   │   ├── recommend.py             # POST /v1/recommend — main inference endpoint
│   │   ├── health.py                # GET /health — liveness + readiness checks
│   │   └── debug.py                 # GET /v1/debug/{part_id} — explain recommendation
│   ├── schemas/
│   │   ├── request.py               # Pydantic models: RecommendRequest, PartGeometry
│   │   └── response.py              # Pydantic models: RecommendResponse, ProcessOption
│   ├── services/
│   │   ├── classifier_service.py    # Loads process classifier, runs inference
│   │   ├── cost_service.py          # Loads cost models, runs per-process decomposition
│   │   ├── constraint_filter.py     # Applies hard manufacturing constraints post-inference
│   │   └── explainer_service.py     # Returns SHAP-based feature importance for a prediction
│   └── middleware/
│       ├── logging_middleware.py    # Request/response logging with latency
│       └── error_handler.py        # Structured error responses
│
├── tests/
│   ├── test_recommend_endpoint.py   # Integration tests: known parts → expected top process
│   ├── test_constraint_filter.py    # Unit tests: undercut parts, large parts, thin walls
│   ├── test_schemas.py              # Pydantic validation: missing fields, bad types
│   └── fixtures/
│       ├── sample_requests.json     # 12 sample request payloads for testing
│       └── expected_responses.json  # Expected top-1 process for each sample
│
├── docs/
│   ├── api_reference.md             # Full endpoint documentation
│   ├── request_response_examples.md # Sample input/output for each endpoint
│   └── latency_benchmarks.md        # P50/P95/P99 inference times
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/                         # Kubernetes manifests (if used)
│       ├── deployment.yaml
│       └── service.yaml
│
└── monitoring/
    ├── latency_dashboard.json        # Grafana dashboard config (or equivalent)
    └── drift_alert_config.yaml       # Alert rules for prediction distribution drift
```

---

## API endpoints

### POST /v1/recommend

Main inference endpoint. Accepts part geometry and requirements, returns ranked process-cost options.

**Request schema:**

```json
{
  "part_id": "PART-20241114-001",
  "geometry": {
    "bounding_box": {"x": 45.2, "y": 30.1, "z": 12.8},
    "volume_mm3": 8420.0,
    "surface_area_mm2": 5210.0,
    "wall_thicknesses": [2.1, 2.4, 1.9, 3.2, 2.0],
    "holes": [
      {"diameter": 3.0, "depth": 12.0},
      {"diameter": 3.0, "depth": 12.0},
      {"diameter": 5.0, "depth": 8.0}
    ],
    "has_undercuts": false,
    "curvature_samples": [0.02, 0.01, 0.04, 0.03, 0.02],
    "symmetry_axes": {"x": 0.2, "y": 0.7, "z": 0.1},
    "source_format": "STEP"
  },
  "requirements": {
    "material_code": "AL6061",
    "quantity": 200,
    "surface_finish_spec": 1.6,
    "tolerance_band": "IT8",
    "max_lead_time_days": 14
  }
}
```

**Response schema:**

```json
{
  "part_id": "PART-20241114-001",
  "recommendations": [
    {
      "rank": 1,
      "process_code": "cnc_milling",
      "process_name": "CNC Milling",
      "confidence": "high",
      "classifier_probability": 0.72,
      "cost_breakdown": {
        "machine_time_cost": 42.10,
        "setup_cost": 18.50,
        "tooling_cost": 9.20,
        "raw_material_cost": 24.80,
        "finishing_cost": 6.40,
        "total_cost": 101.00
      },
      "cost_confidence": "high",
      "material_price_date": "2024-11-13",
      "top_features": [
        {"feature": "wall_thickness_min", "direction": "supports_cnc", "value": 2.1},
        {"feature": "hole_count", "direction": "supports_cnc", "value": 3},
        {"feature": "undercut_flag", "direction": "no_constraint", "value": 0}
      ]
    },
    {
      "rank": 2,
      "process_code": "sheet_metal_bending",
      "process_name": "Sheet Metal Bending",
      "confidence": "medium",
      "classifier_probability": 0.18,
      "cost_breakdown": {
        "machine_time_cost": 28.30,
        "setup_cost": 22.00,
        "tooling_cost": 14.50,
        "raw_material_cost": 19.20,
        "finishing_cost": 5.10,
        "total_cost": 89.10
      },
      "cost_confidence": "high",
      "material_price_date": "2024-11-13",
      "top_features": [
        {"feature": "wall_thickness_avg", "direction": "supports_sheet_metal", "value": 2.3},
        {"feature": "aspect_ratio", "direction": "neutral", "value": 3.5}
      ]
    },
    {
      "rank": 3,
      "process_code": "cnc_turning",
      "process_name": "CNC Turning",
      "confidence": "low",
      "classifier_probability": 0.07,
      "cost_breakdown": null,
      "cost_confidence": "low",
      "note": "Low classifier confidence. Cost estimate omitted. Recommend manual review."
    }
  ],
  "constraints_applied": [],
  "request_id": "req-4f8a2b1c",
  "inference_time_ms": 18,
  "model_version": "classifier_v2_cost_v2"
}
```

---

### GET /v1/debug/{part_id}

Returns the last prediction for a part with full SHAP feature importance breakdown. Used by the internal engineering tool to audit and validate model outputs.

**Response includes:**
- Full feature vector sent to the classifier
- SHAP values for top-3 recommended processes
- Which constraint rules fired (if any)
- Cost model inputs and outputs per process

---

### GET /health

Returns service liveness and readiness. Readiness checks that model artifacts are loaded and material price feed is reachable.

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

## Constraint filter rules

After model inference, hard manufacturing constraints are applied to remove physically impossible recommendations:

| Rule | Condition | Action |
|---|---|---|
| Undercut → no standard molding | `undercut_flag == 1` | Remove injection_molding, die_casting from results |
| Part too large for molding | `bounding_box_x > 600mm` | Remove injection_molding |
| Thin wall → no sand casting | `wall_thickness_min < 2.0mm` | Remove sand_casting |
| Very thin wall → additive preferred | `wall_thickness_min < 0.8mm` | Boost additive_slm in ranking |
| Very high curvature → add 5-axis note | `curvature_complexity > 0.7` | Add `requires_5axis_note` to cnc_milling |

These rules are applied *after* model scoring, not before. The model still scores constrained processes (which is useful for explainability — you can tell the user "this would have been injection molding if not for the undercuts"), but they are filtered out of the final ranked list before the API response is built.

---

## Latency targets and results

| Metric | Target | Actual (production load) |
|---|---|---|
| P50 inference time | < 50 ms | 12 ms |
| P95 inference time | < 100 ms | 31 ms |
| P99 inference time | < 200 ms | 67 ms |

Inference is fast because both models are gradient boosting (not neural networks). The bottleneck is not model scoring — it's material price feed lookup, which adds ~8 ms on average.

---

## Running locally

```bash
# Build and run
docker-compose up --build

# Health check
curl http://localhost:8000/health

# Sample request
curl -X POST http://localhost:8000/v1/recommend \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_requests.json
```

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `CLASSIFIER_MODEL_PATH` | Path to classifier artifact | `models/classifier_v2.pkl` |
| `COST_MODELS_DIR` | Directory with per-process cost model artifacts | `models/cost_v2/` |
| `PRICE_FEED_URL` | Material price API endpoint | internal |
| `PRICE_FEED_API_KEY` | Auth key for price feed | from secrets |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `MODEL_VERSION_TAG` | Version string returned in API responses | `classifier_v2_cost_v2` |

---

## Version history

| Version | Date | Changes |
|---|---|---|
| v1.0 | Week 6 | Initial Flask API. Single model endpoint. No cost breakdown. |
| v1.5 | Week 7 | Migrated to FastAPI. Added Pydantic validation. |
| v2.0 | Week 9 | Added cost decomposition endpoint. Constraint filter. Debug endpoint. |
| v2.1 | Week 10 | Added confidence field. Latency logging. Health endpoint freshness check. |

---

*Internal service. Not for public distribution.*
