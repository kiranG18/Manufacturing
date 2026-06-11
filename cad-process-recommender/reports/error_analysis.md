# Error Analysis — Process Classifier v2
**Date:** Week 9  
**Author:** Aris  
**Scope:** Misclassified parts in test set (n=94 top-1 errors out of 588 total)  

---

## Objective

Understand *why* the model gets things wrong, not just *that* it does. Each failure category below has a proposed fix for v3.

---

## Failure category 1: CNC Milling vs CNC Turning (28 cases)

**What goes wrong:** Parts that are slightly asymmetric get routed to Milling when Turning would be correct and cheaper. The model is not seeing the "rotational with features" pattern clearly.

**Root cause:** `symmetry_score` helps but doesn't fully distinguish a round part with keyways (Turning + Milling) from a prismatic part with holes (pure Milling). We're missing a feature for "dominant geometry type": rotational vs prismatic.

**Representative case:**
- Part: cylindrical shaft with 4 bolt holes on flange, 80mm diameter, 200mm long
- Predicted: CNC Milling (0.54), CNC Turning (0.31)
- Actual: CNC Turning (primary) + Milling (secondary)
- Why wrong: high hole count pushed the model toward Milling

**Proposed fix:** Add a `primary_geometry_type` feature derived from the ratio of rotational to prismatic surface area. This would separate shaft-class parts from bracket-class parts.

---

## Failure category 2: Sheet Metal Stamping vs Sheet Metal Bending (19 cases)

**What goes wrong:** Both processes work on thin flat stock. The model has trouble distinguishing them because most shared features are similar. Stamping tends to be higher-volume, more uniform geometry. Bending tends to be lower-volume, more complex fold patterns.

**Root cause:** We have no `fold_count` or `bend_line_count` feature. Stamping parts also tend to have more punched holes, but `hole_count` alone is not discriminative enough.

**Representative case:**
- Part: L-bracket, 2mm thick mild steel, 3 bend lines, 6 holes
- Predicted: Sheet Metal Stamping (0.48), Sheet Metal Bending (0.42)
- Actual: Sheet Metal Bending
- Why wrong: 6 holes slightly pushed toward Stamping probability

**Proposed fix:** Extract `bend_feature_count` from STEP geometry annotations. This is available in the CAD parsing layer but was not surfaced as a feature in v1 or v2.

---

## Failure category 3: Investment Casting missed entirely (6/10 cases)

**What goes wrong:** Investment casting is systematically underweighted. The model rarely predicts it in the top-3 even for parts that genuinely require it.

**Root cause:** Only 67 training samples. The model has not seen enough investment casting examples to generalize. Additionally, investment casting parts often look similar to sand casting or 5-axis CNC parts geometrically — the difference is driven by tolerance and finish requirements, not geometry alone.

**Proposed fix:** This class needs more training data. Short-term: implement a rule-based override that forces investment casting into the top-3 when `curvature_complexity > 0.7` AND `wall_thickness_min < 2.0` AND volume > 1000 mm³. This is a heuristic patch, not a model improvement.

---

## Failure category 4: Large parts excluded from molding but predicted as molding (11 cases)

**What goes wrong:** Occasionally a large part (bounding_box_x > 500 mm) gets Injection Molding predicted in top-3. This is not physically possible for most injection molding machines.

**Root cause:** The hard constraint filter in `predict.py` was not consistently applied in v1. Fixed in v2 for bounding box > 600mm. Still some leakage at 400–600mm range.

**Status:** Partially fixed. The constraint threshold is now a config parameter in `predict.py`. Needs tighter calibration on the 400–600mm range.

---

## Failure category 5: Compound processes (Turning + Milling) misclassified (14 cases)

**What goes wrong:** Parts that need both Turning and Milling are often classified as one or the other, not the compound class.

**Root cause:** The compound class "Turning + Milling" has only 98 training samples and is genuinely ambiguous — it is, by definition, similar to both parent classes.

**Proposed fix:** Consider removing the compound class from the classifier and instead running two separate binary predictions: "does this part need turning?" and "does this part need milling?" Then surface both if both are positive. This is a product decision as much as a model decision.

---

## Summary table

| Failure category | Count | % of errors | Fixable in v3? |
|---|---|---|---|
| CNC Milling vs Turning | 28 | 30% | Yes — new feature |
| Sheet Metal Stamping vs Bending | 19 | 20% | Yes — bend count feature |
| Investment casting missed | 6 | 6% | Partially — needs more data |
| Large part constraint leak | 11 | 12% | Yes — config fix |
| Compound process confusion | 14 | 15% | Architectural change needed |
| Other / ambiguous | 16 | 17% | Under investigation |

---

## What these errors cost in production

Top-3 errors (the model doesn't have the right answer anywhere in the top 3) make up 16% of test cases. In production terms, these are the cases where an engineer gets a recommendation list that doesn't include the correct process at all. The fallback for these is manual review, so there is no wrong answer delivered to the user — but there is a missed automation opportunity.

Top-1 errors (right answer is in top 3, but ranked second or third) are less costly. The engineer still sees the correct process and can select it.

---

*This analysis was shared with the product team to prioritize v3 feature additions.*
