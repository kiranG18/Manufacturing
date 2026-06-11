# Feature Importance Report — v2 Model
**Date:** Week 9  
**Author:** Aris  
**Method:** SHAP (TreeExplainer) on test set (n=588)  
**Model:** `model_v2.pkl` — XGBoost multi-class, 18 process classes  

---

## Why SHAP

Standard feature importance (gain or split count) in tree models has a known flaw: it overstates the importance of high-cardinality continuous features and understates binary flags. SHAP values are a principled alternative — they assign each feature a contribution to each individual prediction, grounded in game theory (Shapley values). For a classification model, we look at mean absolute SHAP across all test samples and all classes to get a global importance ranking.

---

## Top 10 features by mean |SHAP|

| Rank | Feature | Mean \|SHAP\| | Primary process(es) it informs | Domain explanation |
|---|---|---|---|---|
| 1 | `wall_thickness_min` | 0.41 | Injection Molding, Sheet Metal, Additive SLM | Minimum wall thickness is the most fundamental manufacturability constraint. Molding requires ≥ 1.0–1.5 mm; values < 0.8 mm point to additive. |
| 2 | `undercut_flag` | 0.38 | Injection Molding (exclusion), Die Casting (exclusion) | A binary hard constraint. When present, it eliminates standard tooling processes. The model has learned this almost perfectly — undercut parts are never predicted as injection molding. |
| 3 | `curvature_complexity` | 0.35 | Additive SLM/FDM, 5-Axis CNC | Added in v2 specifically to fix additive misclassification. High curvature (> 0.6) strongly predicts additive or 5-axis. |
| 4 | `volume` | 0.29 | Sand Casting, Forging, Additive FDM | Separates micro-parts (additive viable) from large structural parts (casting/forging). Has high kurtosis — a few very large parts drive much of its importance. |
| 5 | `aspect_ratio` | 0.27 | CNC Turning, Extrusion | The cleanest separator between rotational (high aspect) and prismatic (low aspect) processes. Turning parts cluster at aspect > 8; casting/molding at < 2.5. |
| 6 | `hole_count` | 0.24 | CNC Milling, Laser Cutting | Many holes signal CNC. Many small holes in thin flat parts signal laser cutting. Zero holes with high aspect ratio is a strong turning/extrusion signal. |
| 7 | `thin_wall_flag` | 0.22 | Additive FDM, Sheet Metal Bending | Derived from wall_thickness_min but adds a step-function nonlinearity the tree exploits independently. Improved additive recall by 8 pp in ablation test. |
| 8 | `bounding_box_x` | 0.19 | Injection Molding (constraint), Sand Casting | Machine bed size limits. Parts > 500 mm are infeasible for most molding machines; the model has learned to exclude injection molding above this threshold. |
| 9 | `symmetry_score` | 0.17 | CNC Turning, Forging | Added in v2 to fix the Milling/Turning confusion. Turning parts score 0.7–0.95; milling parts 0.1–0.4. Not useful for casting or additive classes. |
| 10 | `hole_diameter_min` | 0.14 | EDM, Precision CNC | Low global importance, but critical for the rare EDM class — very small holes (< 1 mm) are the primary signal. Without this feature, EDM would essentially never appear in the top-3. |

---

## Features below the top 10

| Feature | Mean \|SHAP\| | Notes |
|---|---|---|
| `bounding_box_y` | 0.09 | Largely redundant with bounding_box_x for this dataset |
| `bounding_box_z` | 0.07 | Somewhat informative for extrusion (uniform tall cross-section) |
| `surface_area` | 0.11 | Overlaps with volume; useful for thin-walled parts where SA/V ratio is distinctive |
| `wall_thickness_avg` | 0.08 | Less important than min; captured as secondary signal |

---

## Key findings

**1. The model is learning the right manufacturing physics.**  
The top two features — `wall_thickness_min` and `undercut_flag` — are literally the first two questions a manufacturing engineer asks when routing a part. This is reassuring: the model isn't pattern-matching on superficial geometry correlation. It has learned the hard constraint structure.

**2. v2 feature additions changed the importance ordering.**  
`curvature_complexity` (added in v2) jumped to rank 3, above `volume` and `aspect_ratio`. This reflects how badly the model was underperforming on additive classes in v1 — those classes had no reliable signal, and curvature provided one.

**3. `symmetry_score` is task-specific.**  
On its own, symmetry_score has modest global importance (rank 9). But for the Milling-vs-Turning confusion pair specifically, it's the most important feature. This is a good argument against dropping low-ranked features globally: importance is task-relative.

**4. Binary flags punch above their information-theoretic weight.**  
`undercut_flag` and `thin_wall_flag` are simple 0/1 values but rank 2nd and 7th globally. This is because XGBoost trees use them as high-confidence branching points — a single split on `undercut_flag = 1` eliminates injection molding and die casting from the entire subtree.

---

## What these numbers imply for v3

- **`pocket_depth`** (not yet implemented): estimated importance ~0.18 based on residual error analysis in CNC milling deep-pocket cases. Would likely rank 7th or 8th.
- **`bend_feature_count`**: estimated ~0.15 for the sheet metal classes. Would help with the Stamping/Bending confusion.
- **`quantity`** (user input, not geometry): expected importance ~0.25 for injection molding vs. CNC routing — molding only makes economic sense above ~5,000–10,000 units.

---

*Generated using `shap.TreeExplainer` with XGBoost format model. SHAP values computed on test set (n=588) and averaged across all output classes.*
