# Developer Guide: Adding a New Data Source

This document outlines the step-by-step process for adding a new data source to the Manufacturing Data Pipeline.

## Step 1: Add a Loader / Client in Ingestion

Create a new file in `src/ingestion/` (e.g., `src/ingestion/shipping_log_loader.py`) and implement a client class to fetch the raw data:

```python
# src/ingestion/shipping_log_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ShippingLogLoader:
    def __init__(self, log_path: str):
        self.log_path = log_path
        
    def load(self) -> pd.DataFrame:
        # Load logic (e.g. read CSV, fetch API, query DB)
        logger.info(f"Loading shipping log from {self.log_path}")
        return pd.read_csv(self.log_path)
```

Expose the class in `src/ingestion/__init__.py`.

## Step 2: Configure the Source Connection

Update `configs/pipeline_config.yaml` to include parameters for your new source:

```yaml
sources:
  ...
  shipping_log:
    type: csv_dump
    path: "data/raw/shipping_log_{date}.csv"
```

## Step 3: Integrate with the Joiner

Modify `src/transforms/joiner.py` to accept the new loader in its constructor and join the new data in `join()`:

```python
# In src/transforms/joiner.py
class OrderGeometryJoiner:
    def __init__(self, cad_loader, price_feed, equipment_loader, shipping_loader):
        ...
        self.shipping_loader = shipping_loader

    def join(self, orders: List[OrderRecord]) -> pd.DataFrame:
        # Perform existing joins...
        joined_df = ...
        
        # Load shipping logs
        shipping_df = self.shipping_loader.load()
        
        # Left join shipping on order_id
        joined_df = pd.merge(joined_df, shipping_df, on="order_id", how="left")
        return joined_df
```

Update `src/run_pipeline.py` to instantiate and pass the new loader to the joiner.

## Step 4: Define Validation Rules

Update `configs/validation_rules.yaml` to define expected types, ranges, allowed values, and null policies for the newly joined columns:

```yaml
columns:
  ...
  shipping_lead_time_days:
    type: int
    range: [1, 90]
    null_policy: default_value
    default: 5
```
