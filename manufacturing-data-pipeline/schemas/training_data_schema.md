# Training Data Schema — v2
**File:** `training_data_v{N}.csv`  
**Last updated:** Week 8  
**Author:** Aris  

This document is the authoritative reference for the output schema of the manufacturing data pipeline. Anyone training or evaluating a model should read this first.

---

## Column reference

### Geometry features (from CAD feature extractor)

| Column | Type | Units | Range | Null policy | Notes |
|---|---|---|---|---|---|
| `part_id` | string | — | — | required | Unique part identifier |
| `bounding_box_x` | float | mm | 0.5–3000 | required | |
| `bounding_box_y` | float | mm | 0.5–3000 | required | |
| `bounding_box_z` | float | mm | 0.5–1000 | required | |
| `volume` | float | mm³ | 1–1e9 | required | |
| `surface_area` | float | mm² | 1–5e7 | required | |
| `wall_thickness_min` | float | mm | 0.1–500 | impute with process median | See note 1 |
| `wall_thickness_avg` | float | mm | 0.5–500 | impute with process median | |
| `aspect_ratio` | float | — | 1–50 | derived; never null | |
| `hole_count` | int | count | 0–200 | 0 if absent | |
| `hole_diameter_min` | float | mm | 0.5–100 | null if no holes | Model handles NaN |
| `undercut_flag` | int | bool | 0 or 1 | 0 if unknown | |
| `thin_wall_flag` | int | bool | 0 or 1 | derived from wall_thickness_min | |
| `curvature_complexity` | float | score | 0.0–1.0 | 0.0 if STL (no annotation) | |
| `symmetry_score` | float | score | 0.0–1.0 | 0.5 default | |

### Process and order context

| Column | Type | Units | Range | Null policy | Notes |
|---|---|---|---|---|---|
| `process_code` | string | — | catalog values | required; drop row if missing | |
| `process_code_encoded` | int | — | 1–20 | derived | |
| `material_code` | string | — | catalog values | required | |
| `material_code_encoded` | int | — | 0–6 | derived | Family encoding |
| `quantity` | int | count | 1–100000 | required | Order quantity |
| `log_quantity` | float | — | 0–12 | derived | log(quantity + 1) |
| `batch_size` | int | count | 1–100000 | defaults to quantity | |
| `surface_finish_ra` | float | µm Ra | 0.05–25 | 3.2 default (as-machined) | |
| `tolerance_index` | int | IT grade | 1–9 | 5 default (IT8) | |

### Material pricing

| Column | Type | Units | Range | Null policy | Notes |
|---|---|---|---|---|---|
| `material_price_per_kg` | float | USD/kg | 0.50–500 | fallback to historical avg | See note 2 |
| `price_date` | date | YYYY-MM-DD | — | required when live | |
| `price_staleness_days` | int | days | 0–999 | 999 if historical fallback | Flag >7 in training |
| `volume_x_price` | float | USD proxy | — | derived | Raw material cost proxy |

### Cost labels (training targets)

| Column | Type | Units | Notes |
|---|---|---|---|
| `cost_machine_time` | float | USD | See note 3 |
| `cost_setup` | float | USD | Includes programming and fixturing |
| `cost_tooling` | float | USD | Per-part amortized tooling cost |
| `cost_raw_material` | float | USD | Actual material cost from invoice |
| `cost_finishing` | float | USD | Post-processing, coating, deburring |
| `cost_total` | float | USD | Sum of above (used for validation cross-check) |
| `cost_label_source` | string | — | "exact", "decomposed", "estimated" — see note 3 |

### Metadata

| Column | Type | Notes |
|---|---|---|
| `order_date` | date | Date order was placed |
| `completion_date` | date | Date order was fulfilled |
| `facility_id` | string | Manufacturing facility identifier |
| `split` | string | "train", "val", or "test" — pre-assigned at dataset creation |
| `pipeline_version` | int | Which pipeline run produced this row |

---

## Notes

**Note 1 — wall_thickness_min imputation:**  
When `wall_thickness_min` is null (occurs for ~5% of STL uploads where ray-casting fails), it is imputed with the median wall thickness for that process class in the training split. This imputation is done by the pipeline, not the model. Models should not see raw nulls in this column.

**Note 2 — material_price_per_kg:**  
In v1, this column was a static lookup (historical average). In v2, it is the actual spot price on the order date for historical rows, and the current day's price for inference. Models trained on v2 data are therefore price-aware: they have seen cost labels across a range of price environments for each material. This is why v2 models generalize better during price spikes.

**Note 3 — cost_label_source:**  
This is the weakest part of the data.

- `"exact"`: Supplier invoice had itemized breakdown matching our 5 components. ~38% of rows.
- `"decomposed"`: Invoice had a lump total; we applied `label_builder.py` heuristics to decompose it. ~51% of rows.
- `"estimated"`: No invoice available; cost was estimated from internal quotes. ~11% of rows. These rows are included in training but excluded from evaluation metrics to avoid artificially good-looking numbers on estimated labels.

When reporting evaluation metrics, only rows with `cost_label_source != "estimated"` are used.

---

## Schema changelog

| Version | Date | Changes |
|---|---|---|
| v1 | Week 3 | Initial schema. Static prices. No `curvature_complexity` or `symmetry_score`. |
| v2 | Week 7 | Added `material_price_per_kg` as dynamic feature. Added `curvature_complexity`, `symmetry_score`, `thin_wall_flag`. Added `cost_label_source` column. |
| v2.1 | Week 9 | Added `price_staleness_days` and `volume_x_price` derived columns. Fixed `tolerance_index` encoding for IT4/IT5. |

---

*This schema is used by both the process classifier and cost decomposition models. Any schema change requires updating `configs/schema_v2.yaml`, rerunning the pipeline, and retraining all dependent models.*
