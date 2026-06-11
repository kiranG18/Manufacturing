# matterize-cost-model

Regression models for per-process-step cost decomposition of custom manufactured parts.

Built during an AI/ML internship at Matterize to improve the accuracy of the cost estimation layer behind the Quanta and Optima products. Given a part's geometry features and a selected manufacturing process, the model outputs a cost breakdown: machine time, setup, tooling, raw material, and finishing.

---

## Problem

A total cost estimate is not enough for procurement teams. If a part is quoted at $180, a buyer needs to know: is that expensive because of machine time (negotiate on run rate), tooling (amortize across larger batch), or raw material (switch alloy grade)? The breakdown drives the negotiation strategy.

Traditional cost estimation at Matterize used per-process lookup tables: average cost-per-hour rates, average material cost-per-kg. These were accurate for typical parts but drifted badly for unusual geometries or when raw material prices moved. This project trains regression models on historical order data — using geometry features plus real-time market inputs — to produce more accurate, current decompositions.

---

## Folder structure

```
matterize-cost-model/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                              # Raw order export (gitignored — PII/proprietary)
│   ├── processed/
│   │   ├── training_data_v1.csv          # Joined features + cost labels, all processes
│   │   ├── training_data_v2.csv          # + live material price column added
│   │   └── material_price_snapshots.csv  # Daily price feed archive (Al, SS, Ti, PEEK, etc.)
│   └── reference/
│       ├── process_cost_schema.md        # Column definitions for training data
│       └── material_index.csv            # Material name → commodity code → price feed key
│
├── notebooks/
│   ├── 01_cost_data_eda.ipynb            # Cost distributions, outlier analysis, label quality
│   ├── 02_feature_analysis.ipynb         # Feature-to-cost correlations per process
│   ├── 03_model_per_process.ipynb        # Training one model per process type
│   ├── 04_model_unified.ipynb            # Single model with process as a feature — comparison
│   └── 05_material_price_sensitivity.ipynb  # How cost predictions shift with material price
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── cost_features.py              # CostFeatureBuilder: geometry + process + live price
│   │   └── price_fetcher.py              # Pulls current raw material prices from feed
│   ├── models/
│   │   ├── cost_regressor.py             # CostDecompositionModel: wraps per-process regressors
│   │   ├── train_per_process.py          # Training entry point: fits one model per process
│   │   └── predict.py                    # Inference: accepts features, returns cost dict
│   └── evaluation/
│       ├── metrics.py                    # mape, mae, coverage_within_10pct
│       └── plots.py                      # Actual vs predicted scatter, residual plots
│
├── reports/
│   ├── cost_model_evaluation_v1.md       # Baseline vs model comparison
│   ├── cost_model_evaluation_v2.md       # After adding live material prices
│   ├── per_process_accuracy.md           # Per-process MAPE breakdown
│   └── material_price_sensitivity.md     # How much predictions drift with price changes
│
├── configs/
│   ├── process_routing.yaml              # Which model handles which process codes
│   └── hyperparams.yaml                  # Hyperparameter config per process model
│
├── outputs/
│   ├── predictions_holdout.csv           # Model predictions vs actuals on test set
│   └── models/                           # Serialized per-process model artifacts
│       ├── cnc_milling_v2.pkl
│       ├── injection_molding_v2.pkl
│       ├── sheet_metal_bending_v2.pkl
│       └── ...
│
└── tests/
    ├── test_cost_features.py
    ├── test_cost_regressor.py
    └── test_price_fetcher.py
```

---

## Cost decomposition schema

Each prediction returns a breakdown across these cost components:

| Component | Description | Typical driver |
|---|---|---|
| `machine_time_cost` | Hourly machine rate × estimated run time | Geometry complexity, operation count |
| `setup_cost` | Fixturing, programming, first-article inspection | Part complexity, batch size |
| `tooling_cost` | Tool wear amortization per part | Material hardness, surface finish requirement |
| `raw_material_cost` | Material weight × live market price | Volume, material grade |
| `finishing_cost` | Post-processing: deburring, coating, anodizing | Surface area, finish spec |
| `total_cost` | Sum of above | — |

The model predicts each component separately. Components are trained independently per process to allow different feature importance profiles (e.g. raw material dominates in additive; tooling dominates in injection molding).

---

## Model architecture

**Design choice:** Per-process models rather than one unified model.

Each manufacturing process has a fundamentally different cost structure. Injection molding is dominated by tooling amortization (which scales with batch size, not geometry). CNC Milling is dominated by machine time (which scales with feature complexity). Training a single model across all processes risks the model learning average behavior rather than the specific physics of each process.

**Per-process model type:** Gradient Boosting Regressor (XGBoost) with log-transformed targets  
**Input features:** Geometry features (from `cad-process-recommender`) + process code + live material price  
**Output:** One value per cost component, then summed to total  
**Target transform:** log(cost + 1) to handle skewed cost distributions; inverse-transformed at prediction time  

---

## Feature set

Inherits geometry features from `cad-process-recommender/src/features/definitions.py`, plus:

| Feature | Source | Rationale |
|---|---|---|
| `process_code` | Classifier output / user selection | Routes to correct per-process model |
| `quantity` | User input | Amortizes setup and tooling cost |
| `material_code` | User input | Drives raw material cost via price feed |
| `material_price_per_kg` | Live price feed (daily) | Current spot price for selected alloy |
| `surface_finish_spec` | User input (Ra value) | Drives finishing cost |
| `tolerance_band` | User input (IT grade) | Tighter tolerances increase machine time |

---

## Results (v2, held-out test set)

### Per-process MAPE (mean absolute percentage error)

| Process | Samples | Total MAPE | Machine Time MAPE | Material MAPE | Tooling MAPE |
|---|---|---|---|---|---|
| CNC Milling | 138 | 8.4% | 11.2% | 5.1% | 9.3% |
| Sheet Metal Bending | 91 | 7.1% | 8.4% | 4.8% | 6.2% |
| Injection Molding | 74 | 12.3% | 9.1% | 3.9% | 19.4% |
| CNC Turning | 52 | 9.7% | 12.1% | 5.5% | 8.8% |
| Additive SLM | 29 | 10.2% | 7.3% | 6.1% | N/A |
| Sand Casting | 17 | 16.8% | 14.2% | 8.4% | 21.1% |

**Overall weighted MAPE: 9.6%** (v1 baseline: 18.3%)

Key insight: Tooling cost is consistently the hardest component to predict. It is most sensitive to batch size and operator-specific amortization decisions that aren't fully captured in the training data.

---

## Usage

```python
from src.features.cost_features import CostFeatureBuilder
from src.models.predict import CostDecompositionModel

builder = CostFeatureBuilder(live_prices=True)
features = builder.build(
    geometry_features={"wall_thickness_min": 2.1, "volume": 45000, ...},
    process_code="cnc_milling",
    material_code="AL6061",
    quantity=50,
    surface_finish_spec=1.6,
    tolerance_band="IT8"
)

model = CostDecompositionModel.load("outputs/models/")
breakdown = model.predict(features)

# returns:
# {
#   "machine_time_cost": 42.10,
#   "setup_cost": 18.50,
#   "tooling_cost": 9.20,
#   "raw_material_cost": 24.80,
#   "finishing_cost": 6.40,
#   "total_cost": 101.00,
#   "confidence": "high",        # high / medium / low based on training data coverage
#   "price_date": "2024-11-12"   # date of material price used
# }
```

---

## Why live material prices matter

Without live prices, a cost model trained on historical data during a period of low aluminum prices will underestimate costs when aluminum prices spike. In v1, we used historical average prices baked into training labels. This caused systematic underestimation on orders placed during supply chain disruptions.

In v2, we feed the current spot price as a feature at inference time. The model was retrained with price as a feature across a 24-month window of price variation, so it has learned how cost scales with price for each material-process combination.

---

## Version history

| Version | Date | Key changes |
|---|---|---|
| v1 | Week 5 | Baseline: single model across all processes, static prices |
| v2 | Week 8 | Per-process models, live material price feed as input feature |
| v2.1 | Week 10 | Log-transform of targets, Platt-calibrated confidence output |

---

*Internal project. Training data contains proprietary order history.*
