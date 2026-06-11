# Operational Runbook: Resolving Pipeline Validation Failures

This runbook outlines procedures for diagnosing and resolving failures in the ETL data pipeline validation checks.

## Diagnosing a Failure

When a pipeline run fails, check the latest log file in `logs/pipeline_runs/` or examine the generated companion metadata JSON (`training_data_v{version}_meta.json`).

Look for:
- `dropped_rows_count` (high numbers indicate upstream data quality issues).
- `drift_detected: true` (indicates feature distribution shifts).
- `coverage_passed: false` (indicates insufficient training samples).

---

## Common Failure Modes and Solutions

### 1. High Row Drop Rate in SchemaValidator
- **Symptom:** More than 10% of ingested rows are dropped during validation.
- **Cause:** Upstream database schema changes, null values in required columns, or invalid numeric ranges (e.g., negative cost values).
- **Resolution:**
  1. Inspect the `dropped_reasons_summary` in the companion JSON metadata to identify which columns caused the drops.
  2. If the drop is due to a null value in a required column, verify if a `null_policy` should be updated to `default_value` or `flag_and_impute` in `configs/validation_rules.yaml`.
  3. If a range is too restrictive (e.g., maximum bounding box is exceeded by a new large part), update the bounds in `configs/validation_rules.yaml`.

### 2. Coverage Check Failure (`coverage_passed: false`)
- **Symptom:** A process code has fewer than `min_samples` (default: 20) rows in the validated set.
- **Cause:** Rare manufacturing processes (e.g., investment casting) do not have enough historical orders.
- **Resolution:**
  1. Check if the orders were incorrectly routed or if some were dropped in schema validation.
  2. If the process is simply rare, you may need to reduce the `min_samples_per_class` threshold in `configs/pipeline_config.yaml` or wait for more orders to accumulate before retraining.

### 3. Distribution Drift Detected
- **Symptom:** `drift_detected: true` in pipeline run metrics.
- **Cause:** A sudden change in part types, batch sizes, or material pricing (e.g., a massive commodity price hike).
- **Resolution:**
  1. Review the `drift_metrics` in the run log to see which column drifted.
  2. If the drift is expected (e.g., titanium prices doubled globally), no action is required; proceed with retraining as this represents real-world shifts the model must learn.
  3. If the drift is unexpected, inspect raw order logs for data injection corruption.
