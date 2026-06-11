# Notebook: 02_feature_engineering.ipynb
# cad-process-recommender
# Author: Aris | Week 5–6

"""
NOTEBOOK OBJECTIVE
------------------
This notebook documents the feature selection and engineering process for
the CAD-to-process classifier. It answers three questions:

  1. Which geometric features have meaningful correlation with process labels?
  2. Do any features need transformation (log scale, binning, flag derivation)?
  3. Which features add the most predictive value when added incrementally?

This is the working document for the transition from feature set v1 (10 features)
to feature set v2 (13 features, adding curvature_complexity, symmetry_score,
and thin_wall_flag).

SECTIONS
--------
0. Setup and data loading
1. Class distribution analysis
2. Per-feature distributions
3. Feature-to-label correlation analysis
4. Pairwise feature correlation (collinearity check)
5. Feature importance from v1 model (permutation importance)
6. New feature exploration: curvature_complexity
7. New feature exploration: symmetry_score
8. New feature exploration: thin_wall_flag
9. Feature set v2 validation
10. Conclusions and v2 feature list

---

SECTION 0: SETUP
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import shap
import xgboost as xgb

# Load processed data
df = pd.read_csv("../data/processed/features_v1.csv")
print(f"Dataset: {len(df)} parts, {df['process_code'].nunique()} process classes")

# Dataset: 3,847 parts, 18 process classes

"""
---

SECTION 1: CLASS DISTRIBUTION ANALYSIS

Key finding: CNC Milling is 23% of the dataset. Investment Casting is 1.7%.
This is the class imbalance problem documented in the model evaluation report.

Plot: Bar chart of sample count per process class.
Annotation: Red dashed line at n=100 (below this, per-class metrics are unreliable).

Result:
  - 14 of 18 classes are above the 100-sample threshold
  - Investment Casting (67), EDM (38), Vacuum Forming (61) are below
  - Recommendation: use class_weight='balanced' in training OR oversample these
  - Decision taken: class_weight='balanced' (simpler; oversampling adds noise risk)

---

SECTION 2: PER-FEATURE DISTRIBUTIONS

For each of the 10 v1 features, plot:
  - Histogram (all parts)
  - Box plot colored by process class (top 6 classes only for readability)

Key findings:

  wall_thickness_min:
    - Bimodal distribution: peak at ~2mm (sheet metal/CNC) and ~5mm (casting)
    - Very useful for separating thin-walled from thick-walled processes
    - Log-transform not needed; distribution is already interpretable

  volume:
    - Highly right-skewed. Range: 1 mm³ (tiny additive parts) to 2.4e8 mm³ (large castings)
    - Log-transform applied: log_volume = log(volume + 1)
    - After log transform: approximately normal within each process class

  aspect_ratio:
    - Turning/Extrusion parts cluster at aspect_ratio > 6
    - Molding/Casting parts cluster at aspect_ratio 1–3
    - Very clean separation on this feature

  hole_count:
    - CNC Milling parts have much higher hole counts than other processes
    - Turning parts: almost always 0 holes (turned on center axis)
    - This feature helps separate Milling from Turning

  undercut_flag:
    - 18% of all parts have undercuts
    - Of undercut parts: 0% are injection_molding or die_casting (correct — these
      cannot have undercuts in standard tooling)
    - Strong discriminative signal for molding exclusion

---

SECTION 3: FEATURE-TO-LABEL CORRELATION

Method: ANOVA F-statistic between each continuous feature and process class.
Higher F-statistic = more variance explained by class membership.

Top features by F-statistic:
  1. wall_thickness_min     F=284.2   *** very strong
  2. aspect_ratio           F=241.7   *** very strong
  3. volume                 F=198.4   *** strong
  4. hole_count             F=156.8   *** strong
  5. undercut_flag          F=134.1   *** strong (binary → biserial correlation)
  6. surface_area           F=112.3   ** strong
  7. bounding_box_x         F=89.4    ** moderate-strong
  8. hole_diameter_min      F=67.2    ** moderate
  9. wall_thickness_avg     F=54.8    * moderate
  10. symmetry_score [NEW]  F=61.4    ** moderate-strong  ← unexpectedly strong
  11. curvature_complexity [NEW] F=78.9 ** strong

Note: symmetry_score was added in v2 exploration and turned out to have stronger
correlation with process label than wall_thickness_avg (a v1 feature).

---

SECTION 4: PAIRWISE FEATURE CORRELATION (COLLINEARITY CHECK)

Pearson correlation matrix on all features.

High correlations found:
  - volume ↔ surface_area: r=0.87  → keep both; surface_area adds signal for
    thin-walled parts that volume alone misses
  - bounding_box_x ↔ bounding_box_y: r=0.71  → acceptable; both kept
  - wall_thickness_min ↔ wall_thickness_avg: r=0.68  → keep; min captures
    the critical constraint, avg captures overall material distribution
  - wall_thickness_min ↔ thin_wall_flag: r=0.81  → near-duplicate
    Decision: keep thin_wall_flag as a complementary feature because it
    introduces a nonlinearity (step function) that the tree model uses
    differently from the continuous value.

No features dropped for collinearity because tree-based models handle
correlated features natively. Would reconsider for logistic regression.

---

SECTION 5: PERMUTATION IMPORTANCE (v1 model)

Method: Permute each feature 50 times, measure drop in top-3 accuracy.
This gives a model-agnostic importance estimate for the v1 feature set.

Results:
  Feature                 | Mean accuracy drop when permuted
  wall_thickness_min      | -8.3%   *** most important
  undercut_flag           | -7.1%   *** second most important
  aspect_ratio            | -5.4%   ** important
  volume                  | -4.8%   ** important
  hole_count              | -4.1%   ** important
  bounding_box_x          | -3.2%   * moderate
  surface_area            | -2.9%   * moderate
  wall_thickness_avg      | -2.1%   * moderate
  hole_diameter_min       | -1.8%   low
  symmetry_score [ADDED]  | -3.8%   ** important (v2 candidate)

Conclusion: the feature set is healthy. No feature can be dropped without
meaningful accuracy loss. Adding symmetry_score looks worthwhile.

---

SECTION 6: CURVATURE COMPLEXITY — NEW FEATURE EXPLORATION

Background: curvature_complexity was not in v1. It was added in v2 after
a cluster of additive manufacturing (SLM, FDM) misclassifications were
attributed to the model not seeing shape complexity signals.

Exploration:
  - Plot: curvature_complexity distribution split by process class
  - Observation: additive parts cluster at curvature > 0.5, CNC milling at
    curvature < 0.3, sheet metal at curvature < 0.1 (nearly flat)
  - 5-axis CNC also clusters at curvature > 0.45 — distinguishable from
    additive by volume and wall thickness

Ablation test: add curvature_complexity to v1 model, measure accuracy change
  Result: Top-3 accuracy on additive classes: 61% → 76% (+15 pp)
  Overall top-3 accuracy: 79% → 82% (+3 pp)
  → Include in v2.

---

SECTION 7: SYMMETRY SCORE — NEW FEATURE EXPLORATION

Background: CNC Turning vs CNC Milling confusion was the #1 error category in v1.
Hypothesis: turning parts are rotationally symmetric; milling parts are not.

Exploration:
  - Compute symmetry_score for all parts in dataset
  - Plot: symmetry_score distribution for turning vs milling parts
  - Observation: turning parts: mean symmetry = 0.76 (std=0.14)
                 milling parts: mean symmetry = 0.31 (std=0.22)
  - Very clean separation. This is exactly the signal we needed.

Ablation test: add symmetry_score to v1 model
  CNC Turning F1: 0.48 → 0.59 (+0.11)
  CNC Milling F1: 0.74 → 0.79 (+0.05, fewer mislabeled as turning)
  → Include in v2.

---

SECTION 8: THIN_WALL_FLAG — NEW FEATURE EXPLORATION

This is derived from wall_thickness_min (thin_wall = wall_thickness_min < 1.5mm).
Rationale: the model may not learn the threshold at 1.5mm automatically from
continuous data; encoding it explicitly helps.

Test: compare v1 model with and without thin_wall_flag added
  Additive FDM recall: 0.68 → 0.75
  Additive SLM recall: 0.61 → 0.69
  Sheet metal (precision sheet) recall: small improvement
  → Include in v2.

---

SECTION 9: FEATURE SET v2 VALIDATION

Final v2 feature list: 13 features
  Original 10 (v1): bounding_box_x/y/z, volume, surface_area, wall_thickness_min,
                    wall_thickness_avg, aspect_ratio, hole_count, hole_diameter_min,
                    undercut_flag
  New in v2: curvature_complexity, symmetry_score, thin_wall_flag

Validation:
  - No feature has >90% correlation with another feature
  - All features have F-statistic > 50 (all informative)
  - Permutation importance: all features have measurable effect when dropped
  - VIF (variance inflation factor) check: no feature > 5 (acceptable multicollinearity)

---

SECTION 10: CONCLUSIONS

1. wall_thickness_min and undercut_flag are the most important features and
   correspond to manufacturing hard constraints — the model is learning the
   right things for the right reasons.

2. curvature_complexity and symmetry_score were the most impactful v2 additions.
   curvature fixed the additive misclassification problem; symmetry fixed the
   turning/milling confusion.

3. thin_wall_flag is a redundant feature (correlated with wall_thickness_min)
   but adds value because it introduces a step-function nonlinearity that the
   tree handles differently from the continuous input.

4. hole_diameter_min has the lowest importance score but should not be dropped:
   it is critical for the rare EDM class (very small holes) which would be
   invisible without it.

5. Volume needs log transformation in linear models but not in XGBoost.
   Using raw volume is fine for the tree model.

Next steps:
  - Explore pocket_depth as a v3 feature (would help CNC Milling deep-pocket
    misestimation identified in error_analysis.md)
  - Explore bend_feature_count (would help Stamping/Bending confusion)
"""
