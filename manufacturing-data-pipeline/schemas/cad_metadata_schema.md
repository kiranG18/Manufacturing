# CAD Geometry Store API Schema

This document specifies the schema of the JSON payloads returned by the CAD Geometry Store API (`cad_geometry_store` connection in `pipeline_config.yaml`).

## Bulk Fetch Endpoint

- **Endpoint:** `POST /api/v1/geometry-features/bulk`
- **Headers:**
  - `Content-Type: application/json`
  - `Authorization: Bearer <API_KEY>`

### Request Payload

```json
{
  "part_ids": [
    "PART-SAMPLE-001",
    "PART-SAMPLE-002"
  ]
}
```

### Response Payload

```json
{
  "status": "success",
  "features": [
    {
      "part_id": "PART-SAMPLE-001",
      "bounding_box_x": 120.5,
      "bounding_box_y": 80.2,
      "bounding_box_z": 15.0,
      "volume": 144600.0,
      "surface_area": 24000.0,
      "wall_thickness_min": 1.8,
      "wall_thickness_avg": 2.2,
      "aspect_ratio": 8.0,
      "hole_count": 4,
      "hole_diameter_min": 3.0,
      "undercut_flag": 0,
      "thin_wall_flag": 0,
      "curvature_complexity": 0.05,
      "symmetry_score": 0.95
    }
  ]
}
```

## Field Definitions

| Field Name | Type | Unit | Description |
|---|---|---|---|
| `part_id` | String | — | Unique identifier of the uploaded CAD model. |
| `bounding_box_x` | Float | mm | Bounding box length. |
| `bounding_box_y` | Float | mm | Bounding box width. |
| `bounding_box_z` | Float | mm | Bounding box height. |
| `volume` | Float | mm³ | Part physical volume. |
| `surface_area` | Float | mm² | Part total surface area. |
| `wall_thickness_min` | Float | mm | Thinest section of the geometry. |
| `wall_thickness_avg` | Float | mm | Average thickness. |
| `aspect_ratio` | Float | Ratio | BB Max / BB Min. |
| `hole_count` | Integer | Count | Number of holes detected. |
| `hole_diameter_min` | Float | mm | Minimum hole diameter. |
| `undercut_flag` | Integer | Binary | `1` if undercut features are found, otherwise `0`. |
| `thin_wall_flag` | Integer | Binary | `1` if walls < 1.5mm exist, otherwise `0`. |
| `curvature_complexity`| Float | 0.0 - 1.0 | Score rating the ratio of freeform curve faces. |
| `symmetry_score` | Float | 0.0 - 1.0 | Rating for rotational/reflective symmetry. |
