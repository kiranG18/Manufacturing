# Process Cost Schema

This document defines the columns, data types, and manufacturing domain context for the training datasets (e.g., `training_data_v2.csv`) used to train the cost decomposition regressors.

## Columns List

| Column Name | Data Type | Units / Range | Description / Domain Context |
|---|---|---|---|
| `part_id` | String | Format: `PART-YYYYMMDD-XXXX` | Unique identifier for the manufactured part. |
| `process_code` | String | e.g., `cnc_milling`, `sheet_metal_bending`, `injection_molding` | The manufacturing process routing. |
| `material_code` | String | e.g., `AL6061`, `SS316L`, `TI6AL4V`, `PEEK` | Raw material specification. |
| `bounding_box_x` | Float | mm | Maximum part dimension along X axis. |
| `bounding_box_y` | Float | mm | Maximum part dimension along Y axis. |
| `bounding_box_z` | Float | mm | Maximum part dimension along Z axis. |
| `volume` | Float | mm³ | Volumetric displacement of the 3D model. |
| `surface_area` | Float | mm² | Total external surface area of the part. |
| `wall_thickness_min` | Float | mm | Minimum measured wall thickness on the part geometry. |
| `wall_thickness_avg` | Float | mm | Average wall thickness across the part geometry. |
| `aspect_ratio` | Float | Ratio (dimensionless) | Ratio of maximum dimensions (e.g., max / min bounding box dimensions). |
| `hole_count` | Integer | Count | Number of holes detected in the part geometry. |
| `hole_diameter_min` | Float | mm | Smallest hole diameter found on the part. |
| `undercut_flag` | Integer | `0` or `1` (binary) | `1` if the part contains undercuts, `0` otherwise. |
| `thin_wall_flag` | Integer | `0` or `1` (binary) | `1` if walls < 1.5mm exist, `0` otherwise. |
| `curvature_complexity` | Float | `[0.0, 1.0]` | Curvature complexity score. Higher means more freeform curves. |
| `symmetry_score` | Float | `[0.0, 1.0]` | Rotational or reflective symmetry score. Higher means more symmetric. |
| `quantity` | Integer | Count (≥ 1) | The order batch size. |
| `log_quantity` | Float | Log scale (dimensionless) | Natural logarithm of the quantity (`log(quantity)`). |
| `batch_size` | Integer | Count (≥ 1) | The setup run batch size. Often matches quantity. |
| `material_code_encoded` | Integer | Integer index | Encoded index of the material code. |
| `material_price_per_kg`| Float | USD / kg | Live spot price of raw material at the order date. |
| `surface_finish_ra` | Float | µm (Ra scale) | Specified surface roughness. Lower means tighter finish requirement. |
| `tolerance_index` | Integer | Heuristic index (e.g., IT grade) | Heuristic representing design tolerances. Higher means tighter tolerances. |
| `volume_x_price` | Float | Interaction term | `volume * material_price_per_kg` (useful cost indicator). |
| `cost_machine_time` | Float | USD | Actual machines-running cost component. |
| `cost_setup` | Float | USD | Programming, fixturing, first-article inspect cost component. |
| `cost_tooling` | Float | USD | Amortized custom molds/tooling/cutting inserts cost component. |
| `cost_raw_material` | Float | USD | Calculated weight of raw stock × stock price cost component. |
| `cost_finishing` | Float | USD | Post-processing, deburring, coating cost component. |
| `cost_total` | Float | USD | Sum of the five cost components (machine time, setup, tooling, raw material, finishing). |
| `cost_label_source` | String | `actual` or `estimated` | Source of the labels. `estimated` indicates a fallback rule was used; ignored in final model evaluations. |
