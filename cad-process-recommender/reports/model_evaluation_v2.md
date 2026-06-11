# Model Evaluation Report — v2
**Date:** Week 9 of internship  
**Author:** Aris (AI/ML Intern)  
**Model version:** `model_v2.pkl`  
**Dataset:** `features_v2.csv` — 3,847 parts, 18 process classes  
**Train/val/test split:** 70/15/15 stratified by process class  

---

## Summary

v2 improves top-3 accuracy by 5 points over v1 (84% vs 79%) and by 22 points over the baseline heuristic (84% vs 62%). The largest gains are on additive manufacturing classes (SLM, FDM) where the new `curvature_complexity` and `thin_wall_flag` features give the model enough signal to distinguish these from CNC-viable parts.

The hardest remaining confusion pairs are CNC Milling vs CNC Turning and Sheet Metal Stamping vs Sheet Metal Bending. These are discussed in `error_analysis.md`.

---

## Dataset characteristics

| Process class | Training samples | Test samples | Notes |
|---|---|---|---|
| CNC Milling | 912 | 138 | Dominant class |
| Sheet Metal Bending | 601 | 91 | Second most common |
| Injection Molding | 487 | 74 | Moderate frequency |
| CNC Turning | 341 | 52 | Overlaps with milling |
| Additive FDM | 298 | 45 | Increased with v2 features |
| Additive SLM | 189 | 29 | Low-count class |
| Sheet Metal Stamping | 201 | 31 | Confusable with bending |
| Die Casting | 156 | 24 | Low count |
| Sand Casting | 112 | 17 | Low count |
| Investment Casting | 67 | 10 | Very low count — see limitations |
| Laser Cutting | 143 | 22 | Good signal from hole features |
| Turning + Milling | 98 | 15 | Compound process |
| Forging | 71 | 11 | |
| Extrusion | 89 | 14 | High aspect ratio signal helps |
| Powder Coating | 43 | 7 | Finishing process; rarely standalone |
| EDM | 38 | 6 | Very low count |
| Vacuum Forming | 61 | 9 | |
| 5-Axis CNC | 82 | 13 | Curvature complexity a strong signal |

Total: 3,847 train+val, 588 test

---

## Results

### Aggregate metrics (test set)

| Metric | Heuristic baseline | Model v1 | Model v2 | Delta (v1→v2) |
|---|---|---|---|---|
| Top-1 accuracy | 41.0% | 58.3% | 63.1% | +4.8 pp |
| Top-3 accuracy | 62.4% | 79.2% | 84.0% | +4.8 pp |
| Top-5 accuracy | 74.1% | 87.6% | 90.3% | +2.7 pp |
| Macro F1 | 0.38 | 0.54 | 0.61 | +0.07 |
| Weighted F1 | 0.44 | 0.61 | 0.68 | +0.07 |

The gap between macro and weighted F1 reflects class imbalance. Macro F1 is the more conservative metric and the one we report to stakeholders.

### Per-class precision and recall (v2 model, test set)

| Process | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CNC Milling | 0.79 | 0.82 | 0.80 | 138 |
| Sheet Metal Bending | 0.74 | 0.76 | 0.75 | 91 |
| Injection Molding | 0.81 | 0.78 | 0.79 | 74 |
| CNC Turning | 0.61 | 0.58 | 0.59 | 52 |
| Additive FDM | 0.77 | 0.80 | 0.78 | 45 |
| Additive SLM | 0.71 | 0.69 | 0.70 | 29 |
| Sheet Metal Stamping | 0.58 | 0.55 | 0.56 | 31 |
| Die Casting | 0.69 | 0.67 | 0.68 | 24 |
| Sand Casting | 0.63 | 0.59 | 0.61 | 17 |
| Investment Casting | 0.50 | 0.40 | 0.44 | 10 |
| Laser Cutting | 0.82 | 0.86 | 0.84 | 22 |
| Forging | 0.64 | 0.55 | 0.59 | 11 |
| Extrusion | 0.78 | 0.79 | 0.78 | 14 |
| 5-Axis CNC | 0.73 | 0.69 | 0.71 | 13 |

---

## Feature importance (SHAP values, test set)

Top 10 features by mean absolute SHAP value:

| Rank | Feature | Mean |SHAP| | Key process it informs |
|---|---|---|---|
| 1 | `wall_thickness_min` | 0.41 | Injection molding, sheet metal |
| 2 | `undercut_flag` | 0.38 | Injection molding exclusion |
| 3 | `curvature_complexity` | 0.35 | Additive, 5-axis CNC |
| 4 | `volume` | 0.29 | Casting vs additive split |
| 5 | `aspect_ratio` | 0.27 | Turning, extrusion |
| 6 | `hole_count` | 0.24 | CNC, laser cutting |
| 7 | `thin_wall_flag` | 0.22 | Additive, precision sheet metal |
| 8 | `bounding_box_x` | 0.19 | Large-part process exclusions |
| 9 | `symmetry_score` | 0.17 | Turning, forging |
| 10 | `hole_diameter_min` | 0.14 | EDM, precision CNC |

The two most important features (`wall_thickness_min`, `undercut_flag`) are manufacturing hard constraints. This is reassuring from a domain perspective: the model has learned the same primary decision criteria a process engineer would apply first.

---

## Calibration

Raw XGBoost probabilities were uncalibrated on validation set (Brier score = 0.14). After Platt scaling calibration, Brier score improved to 0.09. Calibrated probabilities are used in the API response to support downstream confidence display.

---

## Limitations

1. **Investment casting and EDM** have too few samples (<15 test) for reliable per-class metrics. These classes should be treated as "flag for review" rather than high-confidence predictions.

2. **Batch quantity not modeled.** A part viable for injection molding at 10,000 units may not be economical at 50 units. This is a known gap; quantity as a feature is targeted for v3.

3. **Material not modeled.** Some process constraints depend on the target material (e.g. titanium eliminates most casting options). Material input would be a meaningful v3 addition.

4. **Feature extraction accuracy.** `curvature_complexity` and `symmetry_score` depend on mesh quality. Low-resolution STL uploads produce noisy values for these features.

---

## Recommended next steps

- Add `quantity` and `target_material` as input features for v3
- Consider a two-stage model: first predict process family (subtractive / additive / forming / casting), then predict specific process within family
- Investigate active learning loop: low-confidence predictions flagged for engineer review could be relabeled and folded into training data

---

*Evaluation run on AWS ml.t3.medium. Training time: ~4 min for 5-fold CV. Inference: ~12 ms per part at P95.*
