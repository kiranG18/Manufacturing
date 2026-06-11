# Model Evaluation Report — v1
**Date:** Week 7  
**Author:** Aris  
**Model version:** `model_v1.pkl` (baseline)  
**Dataset:** `features_v1.csv` — 3,847 parts, 18 process classes, 10 features  
**Train/val/test split:** 70/15/15 stratified  

---

## Context

This is the evaluation of the first trained classifier, before the v2 feature additions (curvature_complexity, symmetry_score, thin_wall_flag). It establishes the baseline against which v2 is measured and documents what the model gets wrong that motivates the v2 feature work.

---

## Aggregate metrics

| Metric | Heuristic baseline | Model v1 |
|---|---|---|
| Top-1 accuracy | 41.0% | 58.3% |
| Top-3 accuracy | 62.4% | 79.2% |
| Macro F1 | 0.38 | 0.54 |

The v1 model beats the heuristic baseline by 17 percentage points on top-3 accuracy. That said, the absolute number (79.2%) means roughly 1 in 5 parts has the correct process not appearing anywhere in the top-3 recommendation list. That's a meaningful failure rate for a product that engineers are supposed to trust.

---

## What the heuristic baseline is

The baseline is a lookup table with three rules:
1. If `wall_thickness_min < 1.5 mm` → return `[additive_fdm, sheet_metal_bending, laser_cutting]`
2. If `volume > 5e6 mm³` → return `[sand_casting, die_casting, forging]`
3. Otherwise → return `[cnc_milling, sheet_metal_bending, injection_molding]`

This is roughly what a less-experienced engineer would do without looking at a part carefully. The model does substantially better, but v1 still leaves a lot of room.

---

## Worst-performing classes (v1)

| Process | F1 | Problem |
|---|---|---|
| CNC Turning | 0.41 | Confused with CNC Milling — geometry overlap |
| Additive SLM | 0.38 | No curvature signal in v1 features |
| Additive FDM | 0.44 | Same problem as SLM |
| Investment Casting | 0.29 | Only 67 training samples |
| Sheet Metal Stamping | 0.46 | Confused with Sheet Metal Bending |

---

## Failure modes that drove v2 feature additions

**Additive misclassifications:** In v1, FDM and SLM parts were frequently predicted as CNC Milling or Sheet Metal, because the model had no shape complexity signal. The only differentiating features were wall thickness and volume, which are insufficient. Adding `curvature_complexity` gave the model the signal it was missing.

**Turning/Milling confusion:** The model had no way to distinguish a cylindrical shaft (Turning) from a prismatic block with holes (Milling) beyond aspect ratio alone. Adding `symmetry_score` directly targeted this gap.

**Thin-wall additive:** `thin_wall_flag` was added after reviewing 23 cases where FDM parts with walls < 1.5 mm were routed to CNC. The model needed an explicit threshold signal because the continuous `wall_thickness_min` alone didn't create a clean split in the tree.

---

*v1 was trained on a random forest (100 estimators, max_depth=8). Switched to XGBoost in v2 after a hyperparameter sweep showed XGBoost consistently better on this dataset for top-3 accuracy (+2.1 pp on CV) with comparable training time.*
