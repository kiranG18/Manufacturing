# manufacturing-data-pipeline

ETL pipelines that ingest, validate, and normalize manufacturing order data, CAD geometry metadata, live material prices, and equipment availability into unified training datasets for Matterize's AI estimation models.

Built during an AI/ML internship at Matterize. The ML models in `cad-process-recommender` and `matterize-cost-model` depend entirely on the quality and freshness of data produced by this pipeline.

---

## Why this exists

Before this pipeline, training data was assembled manually: someone would export a CSV from the order database, another CSV from the material pricing system, and join them in a notebook. This meant:

- No reproducibility: two runs with the same "data" could produce different training sets
- No validation: a schema change in the order database would silently corrupt the feature matrix
- No versioning: it was impossible to reproduce a model trained 3 months ago
- No freshness: material prices in the training data could be months out of date

This pipeline fixes all four problems.

---

## Folder structure

```
manufacturing-data-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── order_extractor.py        # Pulls completed orders from internal DB
│   │   ├── cad_metadata_loader.py    # Loads geometry feature vectors from CAD store
│   │   ├── price_feed.py             # Fetches daily material spot prices
│   │   └── equipment_log_loader.py   # Loads machine availability and hourly rate logs
│   ├── transforms/
│   │   ├── joiner.py                 # Joins orders, geometry, prices, equipment into one row
│   │   ├── normalizer.py             # Scales features, handles units, cleans strings
│   │   ├── label_builder.py          # Builds cost component labels from raw invoice data
│   │   └── versioner.py             # Tags and archives each dataset version
│   ├── validation/
│   │   ├── schema_validator.py       # Checks column types, ranges, presence
│   │   ├── coverage_checker.py       # Reports coverage across process types and materials
│   │   └── drift_detector.py        # Flags if incoming data distribution shifts vs. last run
│   └── loaders/
│       ├── dataset_writer.py         # Writes final CSV + metadata to versioned output folder
│       └── catalog_updater.py        # Updates the dataset catalog used by training scripts
│
├── configs/
│   ├── pipeline_config.yaml          # Source connections, output paths, schedule
│   ├── validation_rules.yaml         # Per-column validation: type, range, null policy
│   └── schema_v2.yaml               # Current canonical schema for training data
│
├── schemas/
│   ├── order_schema.md               # Source schema: order database
│   ├── cad_metadata_schema.md        # Source schema: geometry feature store
│   ├── training_data_schema.md       # Output schema: final training dataset
│   └── changelog.md                  # Schema version history
│
├── docs/
│   ├── architecture.md               # Data flow diagram and component descriptions
│   ├── adding_a_new_source.md        # How to wire in a new data source
│   └── validation_runbook.md         # What to do when validation fails
│
├── logs/
│   └── pipeline_runs/               # Per-run logs (gitignored for size)
│
└── tests/
    ├── test_order_extractor.py
    ├── test_joiner.py
    ├── test_schema_validator.py
    └── test_coverage_checker.py
```

---

## Data sources

| Source | Description | Refresh frequency | Owner |
|---|---|---|---|
| Order database | Historical completed orders with actual cost invoices | On demand (new completions) | Engineering |
| CAD geometry store | Pre-computed feature vectors from uploaded part files | On part upload | Platform |
| Material price feed | Spot prices for metals, polymers, composites | Daily | External API |
| Equipment log | Machine availability, downtime, hourly rates | Weekly | Operations |

---

## Pipeline stages

```
[Order DB] ──┐
[CAD Store] ──┤──► joiner.py ──► normalizer.py ──► label_builder.py ──► schema_validator.py ──► versioner.py ──► [Training Dataset vN]
[Price Feed] ─┤
[Equip Log] ──┘
                                                                           ▲
                                                               coverage_checker.py
                                                               drift_detector.py
```

### Stage 1: Ingestion
Each source has its own extractor class. Sources are pulled independently and written to a staging area. If a source pull fails, the pipeline halts with a descriptive error rather than proceeding with stale data.

### Stage 2: Join
The joiner links orders to geometry features by `part_id`, to price data by `material_code + date`, and to equipment data by `process_code + facility_id`. Left join on orders: if a part has no geometry feature vector, it is excluded with a log entry (not silently dropped).

### Stage 3: Normalize
- Numeric: StandardScaler fit on training split only (not full dataset — avoid leakage)
- Categorical: Label encoding using catalog defined in `configs/schema_v2.yaml`
- Units: All dimensions converted to mm, all costs to USD, all weights to kg
- Null policy: per-column rules in `configs/validation_rules.yaml`

### Stage 4: Label building
Raw invoice data has line items, not the per-component breakdown the model needs. `label_builder.py` applies heuristic rules to decompose invoice totals into the 5 cost components (machine time, setup, tooling, material, finishing). This decomposition is the weakest link in the data chain — see limitations.

### Stage 5: Validation
Before writing, the schema validator checks:
- All required columns present with correct types
- Numeric values within defined ranges (e.g. no negative costs, no wall thickness > 500mm)
- No duplicate `part_id + process_code` rows in the training set
- Coverage: each process class has at least `min_samples` rows (default: 20)

### Stage 6: Version + write
Each run produces a versioned output in `data/processed/training_data_vN.csv` with a companion metadata file `training_data_vN_meta.json` containing: run timestamp, source row counts, row count after joins, coverage report, and schema version.

---

## Validation rules (excerpt from validation_rules.yaml)

```yaml
columns:
  wall_thickness_min:
    type: float
    range: [0.1, 500.0]
    null_policy: flag_and_impute   # impute with process-specific median
  cost_total:
    type: float
    range: [0.01, 100000.0]
    null_policy: drop_row
  process_code:
    type: string
    allowed_values: [cnc_milling, cnc_turning, sheet_metal_bending, ...]
    null_policy: drop_row
  material_price_per_kg:
    type: float
    range: [0.50, 500.0]
    null_policy: fallback_to_historical_avg
    staleness_threshold_days: 7   # prices older than 7 days → warn
```

---

## Coverage report (example output)

```
Pipeline run: 2024-11-14 06:00 UTC
Total rows: 4,312 (pre-validation) → 3,847 (post-validation)
Dropped: 465 rows (10.8%)
  - Missing geometry features: 201
  - Cost label out of range: 89
  - Unknown process code: 47
  - Duplicate part_id: 128

Coverage by process class:
  cnc_milling          | 2,693 rows  ████████████████████████ OK
  sheet_metal_bending  | 1,771 rows  ████████████████████     OK
  injection_molding    | 1,436 rows  ████████████████         OK
  cnc_turning          |   1,006 rows ███████████             OK
  additive_fdm         |   879 rows  ██████████               OK
  additive_slm         |   557 rows  ██████                   OK
  die_casting          |   460 rows  █████                    OK
  sand_casting          |   330 rows  ████                     OK
  investment_casting    |   198 rows  ██                       WARN: below 200 threshold
  forging               |   209 rows  ██                       OK
  extrusion             |   263 rows  ███                      OK
  laser_cutting         |   422 rows  █████                    OK
  ...

Material price freshness:
  AL6061   | 2024-11-13 | 1 day old  | OK
  SS316L   | 2024-11-13 | 1 day old  | OK
  TI6AL4V  | 2024-11-10 | 4 days old | OK
  PEEK     | 2024-11-07 | 7 days old | WARN

Dataset version: v18 written to data/processed/training_data_v18.csv
```

---

## Limitations

1. **Label decomposition is heuristic.** The invoice-to-cost-component decomposition in `label_builder.py` uses rules derived from discussions with the manufacturing team, not exact accounting records. Some component labels are estimates, particularly setup and tooling, which suppliers quote as a single line item.

2. **Historical orders only.** The pipeline captures completed orders. Parts that were quoted but never ordered are not included, which may bias the training distribution toward geometries that are "easy" to manufacture.

3. **Single facility.** Equipment log data currently comes from one facility's machines. Hourly rates and availability patterns may differ for other facilities or vendors.

---

## Running the pipeline

```bash
# Full run (all sources, write new version)
python -m src.run_pipeline --config configs/pipeline_config.yaml

# Validation only (no write)
python -m src.run_pipeline --validate-only

# With specific date range
python -m src.run_pipeline --start-date 2024-01-01 --end-date 2024-11-01
```

---

*Internal tooling. Not for public distribution.*
