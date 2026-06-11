# Cost Model Evaluation Report — v2
**Author:** Aris  
**Date:** Week 8  
**Dataset:** v2 validation dataset (2,462 training rows, 588 held-out test parts)

---

## Executive Summary

The v2 cost model architecture was designed to address the key failure modes of the v1 baseline: specifically, the inability to capture process-specific physical constraints and susceptibility to raw material price fluctuations.

By introducing **per-process isolated models** and feeding **live material prices** as dynamic features, the overall weighted Mean Absolute Percentage Error (MAPE) was reduced from **18.3%** in v1 to **9.6%** in v2.

---

## Architectural Improvements in v2

### 1. Process Isolation (Per-Process Routing)
Instead of a single model, we now route incoming parts to specialized XGBoost regressors depending on the selected `process_code`. This allows each model to fit feature importances to the physics of that specific process:
- **CNC Milling:** Feature importance is driven by geometry complexity, wall thickness, and hole counts.
- **Injection Molding:** Feature importance is dominated by setup, batch size, and material code (to capture mold costs and polymer run rates).

### 2. Log-Transformed Targets
Cost data is highly right-skewed, spanning multiple orders of magnitude (from a $15 turned pin to a $10,000 custom injection mold). In v2, we train all models using `log(cost + 1)` target transformation, preventing high-cost outliers from dominating the gradient updates and improving predictions across low-to-medium cost parts.

### 3. Dynamic Material Price Feed
We added a live commodity spot price feed (`material_price_per_kg`). The model is trained on a 24-month window of price history, learning how overall cost scales with material commodity trends.

---

## Performance Comparison (v1 vs v2)

| Process | v1 MAPE | v2 MAPE | Improvement |
|---|---|---|---|
| CNC Milling | 16.2% | 8.4% | -7.8 pp |
| Sheet Metal Bending | 14.8% | 7.1% | -7.7 pp |
| Injection Molding | 22.1% | 12.3% | -9.8 pp |
| CNC Turning | 18.3% | 9.7% | -8.6 pp |
| Additive SLM | 19.4% | 10.2% | -9.2 pp |
| **Overall Weighted** | **18.3%** | **9.6%** | **-8.7 pp** |

---

## Cost Decomposition Breakdown (v2)

Rather than predicting a single total, v2 outputs 5 independent cost components. Component-level error analysis shows:
- **Raw Material Cost:** Lowest error (overall MAPE 3-5%), since material volume × price feed is highly deterministic.
- **Machine Time Cost:** Highly accurate on standard geometries (MAPE ~8%), but increases on high-curvature complex parts.
- **Tooling Cost:** The hardest component to predict (MAPE ~19.4% in injection molding), as custom tool quotes reflect individual supplier capacity rather than geometry alone.

---

## Summary of Action Items

1. **Deploy to opt-in beta:** The 9.6% overall MAPE is well within the acceptable threshold for R&D estimation.
2. **Develop a confidence scoring heuristic:** Use training sample size and feature ranges to label predictions as `high`, `medium`, or `low` confidence in the API.
3. **Capture pocket depth:** For CNC Milling v3, include a depth-to-width feature to improve deep cavity estimation.
