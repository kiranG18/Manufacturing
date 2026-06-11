# cad-process-recommender

Multi-class classifier that maps 3D CAD part geometry to a ranked list of feasible manufacturing processes.

Built during an AI/ML internship at [Matterize](https://matterize.com), where the core product ingests uploaded CAD files and recommends optimal manufacturing routes and cost estimates for R&D and procurement teams.

---

## Problem

A manufacturing engineer looking at a new part has to manually decide which processes are viable — CNC machining, injection molding, sheet metal forming, casting, additive, etc. That judgment requires years of domain experience and takes time. This project trains a model to do that initial triage automatically, returning a ranked list of feasible processes so engineers can focus on comparing options rather than generating them.

---

## What this repository does

1. Parses 3D CAD geometry into structured numerical features
2. Trains a gradient-boosted classifier on those features
3. Evaluates recommendations using top-3 accuracy and per-class precision/recall
4. Exports a model artifact and prediction CSV for downstream cost modeling

---

## Folder structure

```
cad-process-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        # Original CAD metadata exports (not committed — see .gitignore)
│   ├── processed/
│   │   ├── features_v1.csv         # Feature matrix, one row per part
│   │   ├── features_v2.csv         # Version after adding curvature + undercut flags
│   │   └── label_map.json          # Process name → integer class index
│   └── reference/
│       ├── process_catalog.csv     # 50+ processes with feasibility constraints
│       └── sample_parts.csv        # 12 manually validated reference parts for sanity checks
│
├── notebooks/
│   ├── 01_eda.ipynb                # Distribution of processes in training data, class imbalance analysis
│   ├── 02_feature_engineering.ipynb # Feature computation, correlation matrix, importance ranking
│   ├── 03_model_training.ipynb     # Training run, hyperparameter search, CV results
│   └── 04_error_analysis.ipynb     # Confusion matrix, failure case review, boundary cases
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── extractor.py            # CADFeatureExtractor class: parses geometry → feature dict
│   │   ├── definitions.py          # Feature names, units, expected ranges, domain notes
│   │   └── validators.py           # Sanity checks: wall thickness > 0, aspect ratio bounds, etc.
│   ├── models/
│   │   ├── classifier.py           # ProcessClassifier wrapper around XGBoostClassifier
│   │   ├── train.py                # Training entry point: loads data, fits, saves artifact
│   │   └── predict.py              # Inference: loads artifact, returns ranked process list
│   └── evaluation/
│       ├── metrics.py              # top_k_accuracy, per_class_report, coverage_at_k
│       └── plots.py                # Confusion matrix heatmap, class distribution bar chart
│
├── reports/
│   ├── model_evaluation_v1.md      # First model run: baseline heuristic vs. classifier
│   ├── model_evaluation_v2.md      # After feature additions: wall thickness, undercut flag
│   ├── error_analysis.md           # Deep dive on misclassified parts, boundary process pairs
│   └── feature_importance.md       # Top 10 features by Shapley value with domain explanation
│
├── outputs/
│   ├── predictions_holdout.csv     # Model predictions on held-out test set
│   ├── confusion_matrix.png        # Rendered confusion matrix
│   └── model_v2.pkl                # Final serialized model artifact (gitignored in prod)
│
└── tests/
    ├── test_extractor.py           # Unit tests: feature extraction on reference parts
    ├── test_classifier.py          # Smoke test: model loads and returns ranked list
    └── test_validators.py          # Edge cases: zero-thickness walls, missing fields
```

---

## Feature set (v2)

Features are computed from CAD geometry metadata, not raw mesh data. Each feature maps to a manufacturing constraint a process engineer would check manually.

| Feature | Units | Rationale |
|---|---|---|
| `bounding_box_x/y/z` | mm | Machine bed size constraints |
| `volume` | mm³ | Material cost and process feasibility |
| `surface_area` | mm² | Finishing cost, coating feasibility |
| `wall_thickness_min` | mm | Critical for injection molding and sheet metal |
| `wall_thickness_avg` | mm | Overall manufacturability signal |
| `aspect_ratio` | — | Long thin parts: CNC/turning; compact: molding |
| `hole_count` | count | Drilling operations; affects CNC vs. casting routing |
| `hole_diameter_min` | mm | Tight holes: CNC; larger: casting viable |
| `undercut_flag` | bool | Undercuts eliminate molding from candidates |
| `thin_wall_flag` | bool | Walls < 1.5 mm: additive or precision sheet metal |
| `curvature_complexity` | 0–1 score | Complex curves favor additive or 5-axis CNC |
| `symmetry_score` | 0–1 score | Symmetric parts viable for turning/forging |
| `estimated_setup_complexity` | low/med/high | Heuristic from feature combination |

---

## Model

**Algorithm:** XGBoost multi-class classifier with softmax objective  
**Input:** Feature vector (13 features, v2)  
**Output:** Probability distribution over 18 process classes; top-3 returned to API  
**Evaluation metric:** Top-3 accuracy (primary), per-class F1 (secondary)  
**Baseline:** Lookup-table heuristic based on wall thickness and volume thresholds  

Training used stratified k-fold (k=5) to handle class imbalance across 18 process types.

---

## Results summary (v2 model, held-out test set)

| Metric | Baseline (heuristic) | Model v1 | Model v2 |
|---|---|---|---|
| Top-1 accuracy | 41% | 58% | 63% |
| Top-3 accuracy | 62% | 79% | 84% |
| Macro F1 | 0.38 | 0.54 | 0.61 |

Hardest process pairs to distinguish: CNC Milling ↔ CNC Turning (geometry overlap), Sheet Metal Stamping ↔ Sheet Metal Bending (feature similarity).

---

## Usage

```python
from src.features.extractor import CADFeatureExtractor
from src.models.predict import ProcessClassifier

extractor = CADFeatureExtractor()
features = extractor.extract("path/to/part.step")

model = ProcessClassifier.load("outputs/model_v2.pkl")
recommendations = model.predict_top_k(features, k=3)

# returns: [("CNC Milling", 0.72), ("Sheet Metal Bending", 0.18), ("Turning", 0.07)]
```

---

## Limitations and known gaps

- Training data skewed toward CNC and sheet metal (most common in historical orders). Low-volume processes (investment casting, forging) have higher misclassification rates.
- Feature extraction depends on CAD files being parseable STEP or STL. Malformed uploads fall back to a default feature vector and are flagged for human review.
- Model does not yet account for batch quantity, which affects process economics (injection molding only makes sense above a certain volume threshold).

---

## Version history

| Version | Date | Change |
|---|---|---|
| v1 | Week 4 | Baseline: 10 features, random forest |
| v2 | Week 7 | Added curvature, undercut flag, undercut-aware constraint filtering; switched to XGBoost |
| v2.1 | Week 9 | Calibrated probabilities using Platt scaling for downstream confidence display |

---

## Dependencies

```
xgboost>=1.7
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
shap>=0.41
matplotlib>=3.6
pytest>=7.2
```

---

*Internal project. Not for public distribution. Training data contains proprietary order history.*
