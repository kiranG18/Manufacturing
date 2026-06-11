# Schema Version Changelog

This document tracks updates and modifications to the consolidated training dataset schema produced by the ETL pipeline.

## [v2.0.0] - Week 7

Major overhaul to support process-specific cost regression and live pricing feeds.

### Added
- `material_price_per_kg` column: live commodity spot price feed at order date.
- `price_date` column: timestamp of the spot price.
- `price_staleness_days` column: tracking days elapsed since feed fetch.
- `cost_machine_time`, `cost_setup`, `cost_tooling`, `cost_raw_material`, `cost_finishing` columns: decomposed cost component labels.
- `cost_label_source` column: indicating if cost breakdown was parsed as `exact` or reconstructed as `estimated`.
- `curvature_complexity` column: geometry curvature rating.
- `undercut_flag` column: binary indicator for undercuts.
- `process_code_encoded` and `material_code_encoded` ordinal mappings.
- `log_quantity` and `volume_x_price` interaction fields.

### Changed
- `cost_total` range updated to match sum of components.
- Null policy on geometry features updated to allow `flag_and_impute` instead of strictly dropping rows for min wall thickness.

---

## [v1.0.0] - Week 4

Initial baseline schema supporting the baseline Random Forest recommender and unified cost regressor.

### Added
- Baseline columns: `part_id`, `process_code`, `material_code`, `quantity`, `bounding_box_x`, `bounding_box_y`, `bounding_box_z`, `volume`, `surface_area`, `wall_thickness_min`, `wall_thickness_avg`, `aspect_ratio`, `hole_count`, `hole_diameter_min`.
- Baseline output label: `cost_total`.
