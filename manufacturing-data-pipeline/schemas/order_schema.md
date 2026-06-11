# Order Database Schema

This document specifies the schema of the source relational tables (located in the internal Postgres `order_db` instance) consumed by the `OrderExtractor` component of the ETL pipeline.

## 1. Table: `orders`

Tracks Completed and active manufacturing orders placed by clients.

| Column | Type | Nullable | Description / Domain Context |
|---|---|---|---|
| `order_id` | VARCHAR(50) | NO (PK) | Unique identifier for the order. |
| `part_id` | VARCHAR(50) | NO | Reference to the CAD part ID. |
| `process_code` | VARCHAR(50) | NO | Selected manufacturing process (e.g. `cnc_milling`). |
| `material_code` | VARCHAR(50) | NO | Chosen stock material grade (e.g. `AL6061`). |
| `quantity` | INTEGER | NO | Ordered quantity (must be ≥ 1). |
| `batch_size` | INTEGER | NO | Production batch run size (default same as quantity). |
| `surface_finish_ra` | NUMERIC(5,2) | YES | Target surface roughness in microns. |
| `tolerance_band` | VARCHAR(10) | YES | ISO IT Tolerance band (e.g. `IT8`). |
| `order_date` | DATE | NO | Timestamp when order was placed. |
| `completion_date` | DATE | YES | Timestamp when order was fulfilled (status='completed'). |
| `facility_id` | VARCHAR(50) | NO | Manufacturing shop floor site code. |
| `status` | VARCHAR(20) | NO | Lifecycle state (e.g., `completed`, `pending`, `cancelled`). |

---

## 2. Table: `invoices`

Holds financial billing details per completed order, from which the target labels are extracted.

| Column | Type | Nullable | Description / Domain Context |
|---|---|---|---|
| `invoice_id` | VARCHAR(50) | NO (PK) | Unique billing record ID. |
| `order_id` | VARCHAR(50) | NO (FK) | Reference back to parent order. |
| `cost_total` | NUMERIC(10,2) | NO | Total billed amount for the order (USD). |
| `cost_machine_time` | NUMERIC(10,2) | YES | Billed cost for machine run hours. |
| `cost_setup` | NUMERIC(10,2) | YES | Billed cost for manual setup, fixturing. |
| `cost_tooling` | NUMERIC(10,2) | YES | Billed cost for tooling/dies/inserts. |
| `cost_raw_material` | NUMERIC(10,2) | YES | Billed cost for feedstock stock. |
| `cost_finishing` | NUMERIC(10,2) | YES | Billed cost for deburring/anodizing/coating. |
| `label_source` | VARCHAR(20) | NO | Source of decomposition metrics: `exact` or `decomposed`. |
