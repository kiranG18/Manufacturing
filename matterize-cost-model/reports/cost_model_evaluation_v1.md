# Cost Model Evaluation Report — v1 (Baseline)
**Author:** Aris  
**Date:** Week 5  
**Dataset:** v1 baseline dataset (static average prices, historical quotes)  

---

## Executive Summary

The v1 cost model was built as a baseline proof-of-concept for the Matterize cost estimation engine. It used a single unified model across all manufacturing processes (e.g., CNC milling, injection molding, sheet metal bending) and relied on static average material prices. 

While v1 demonstrated that machine learning could outperform traditional manual lookup tables, it suffered from high overall prediction errors (weighted MAPE of **18.3%**) and failed to capture process-specific cost drivers or market price fluctuations.

---

## Baseline Architecture (v1)

- **Model Type:** Single unified Random Forest Regressor
- **Input Features:** Geometry features (volume, surface area, bounding box), process code (encoded as a categorical feature), and quantity.
- **Material Prices:** Hardcoded average historical material prices baked directly into the training data labels.
- **Target:** Combined total part cost (no decomposition).

---

## Results and Performance

The model was evaluated on a held-out test set representing typical order distributions.

### Process-Level Performance (v1)

| Process | Samples | Total MAPE | Key Failure Mode |
|---|---|---|---|
| CNC Milling | 138 | 16.2% | Underestimated setup cost for low quantities; failed to scale with tolerance. |
| Sheet Metal Bending | 91 | 14.8% | Overestimated cost of simple single-bend parts. |
| Injection Molding | 74 | 22.1% | Massive error on low-volume orders due to inability to isolate mold tooling cost. |
| CNC Turning | 52 | 18.3% | Inability to distinguish turning from milling geometry characteristics. |
| Additive SLM | 29 | 19.4% | Failed to scale build-time costs properly. |
| **Overall Weighted** | **384** | **18.3%** | — |

---

## Key Limitations of the v1 Architecture

### 1. The "Average Physics" Problem
By training a single model across all processes, the regressor attempted to fit a single set of feature weights to completely different manufacturing methods. For example, part volume dominates the raw material cost in additive manufacturing, whereas setup and custom tooling dominate the cost of injection molding. The model ended up averaging these effects, leading to high errors for both.

### 2. Static Material Prices
Because raw material prices fluctuate daily, the model's hardcoded prices drifted quickly. When aluminum spot prices rose, the model systematically underestimated the cost of aluminum orders, causing margin erosion.

### 3. Lack of Decomposition
Providing only a single "total cost" value failed to give procurement teams the transparency they needed. A total price of $150 was useless without knowing how much of it was machine time, tooling, or material.

---

## Next Steps for v2

1. **Decompose the Targets:** Retrain the models on 5 distinct cost components (machine time, setup, tooling, raw material, and finishing).
2. **Per-Process Model Isolation:** Split the single unified model into independent, process-specific regressors (e.g., one model for `cnc_milling`, one for `injection_molding`).
3. **Live Price Feed Integration:** Pass live spot material prices from the external commodity feed as a dynamic feature at inference time.
