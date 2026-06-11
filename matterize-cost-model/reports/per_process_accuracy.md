# Per-Process Cost Model Accuracy — v2
**Author:** Aris  
**Date:** Week 10  
**Dataset:** 588 held-out test parts (same split as process classifier)  

---

## Overview

This report breaks down cost model accuracy by process type and by cost component. The purpose is twofold: (1) identify where the model is reliable enough to present to procurement teams without caveats, and (2) identify which process/component combinations still need improvement before going into Optima.

---

## Why per-process matters

A single aggregate MAPE number hides a lot. A 10% average error across all processes could mean the model is uniformly mediocre, or it could mean it's excellent on high-frequency processes (CNC, sheet metal) and poor on low-frequency ones (casting, forging). The latter is actually fine from a product perspective — the common cases cover most of the volume.

---

## CNC Milling (n=138, MAPE=8.4%)

**Reliability: High. Approved for Optima production use.**

CNC Milling has the largest test set and the most stable cost structure. Machine time is the dominant driver (~45% of total cost) and is well-predicted by geometry complexity features.

| Component | MAPE | Notes |
|---|---|---|
| Machine time | 11.2% | Slightly high for complex 5-axis-equivalent geometries |
| Setup | 8.7% | Consistent across part sizes |
| Tooling | 9.3% | Driven by material hardness; AL6061 best-predicted |
| Raw material | 5.1% | Strong — live price feed working well |
| Finishing | 7.8% | Good when finish spec is provided; defaults to Ra 3.2 otherwise |

**Failure pattern:** Parts with deep pockets or narrow internal radii are underestimated by ~15% because the model doesn't yet see a `pocket_depth` feature. This is a known v3 addition.

---

## Sheet Metal Bending (n=91, MAPE=7.1%)

**Reliability: High. Approved for Optima production use.**

Sheet metal cost is dominated by machine time and setup. Material cost is predictable because sheet metal uses standard stock thickness and grade, which have well-indexed pricing.

| Component | MAPE | Notes |
|---|---|---|
| Machine time | 8.4% | Bend count is the main driver; model captures this well |
| Setup | 6.2% | Low error — setup is fairly standardized for sheet metal |
| Tooling | 6.2% | Dies are amortized consistently |
| Raw material | 4.8% | Standard stock pricing: very accurate |
| Finishing | 9.1% | Higher variance — powder coat vs. no finish varies by customer |

---

## Injection Molding (n=74, MAPE=12.3%)

**Reliability: Medium. Use with caveat in Optima. Flag for engineer review on high-value quotes.**

Injection molding has the highest MAPE of the high-frequency processes. The issue is tooling cost, which is the single largest component (often 30–50% of total for low-quantity orders) and the hardest to predict.

| Component | MAPE | Notes |
|---|---|---|
| Machine time | 9.1% | Reasonable — cycle time well-predicted from part volume |
| Setup | 8.4% | Consistent |
| Tooling | 19.4% | High error. Tooling is quoted differently per supplier. See below. |
| Raw material | 3.9% | Polymer pricing is very stable. Best-predicted component. |
| Finishing | 7.2% | Moderate |

**Tooling cost problem:** Tooling (mold fabrication) is quoted per-job by individual toolmakers. The same mold design can cost $8,000 or $20,000 depending on the toolmaker's capacity, location, and material. Historical labels reflect what was actually paid, not a standardized rate. The model learns an average but the variance is fundamentally driven by vendor selection, not part geometry.

**Recommended caveat text for Optima:** "Tooling cost estimate has higher uncertainty for injection molding. Recommend requesting 2–3 toolmaker quotes before finalizing procurement decision."

---

## CNC Turning (n=52, MAPE=9.7%)

**Reliability: High for simple turned parts. Medium for turning+milling compounds.**

| Component | MAPE | Notes |
|---|---|---|
| Machine time | 12.1% | Higher for compound turning+milling operations |
| Setup | 7.4% | Low — turning setup is standardized |
| Tooling | 8.8% | Acceptable |
| Raw material | 5.5% | Bar stock pricing: stable |
| Finishing | 8.2% | — |

---

## Additive SLM (n=29, MAPE=10.2%)

**Reliability: Medium. Small test set; confidence interval wide.**

For additive processes, machine time is essentially build time (which scales with volume and support structure). Tooling is not applicable. Raw material is powder cost per gram, which is well-priced.

| Component | MAPE | Notes |
|---|---|---|
| Machine time | 7.3% | Build time from volume estimate: good |
| Setup | 11.4% | Support removal and post-processing varies by geometry |
| Tooling | N/A | Not applicable for additive |
| Raw material | 6.1% | Powder pricing: stable |
| Finishing | 16.2% | High variance — HIP, heat treat, surface finish vary widely |

---

## Sand Casting (n=17, MAPE=16.8%)

**Reliability: Low. Do not use for Optima procurement negotiations without engineer review.**

Sand casting test set is too small to draw strong conclusions. The high MAPE is driven by pattern cost variability (similar to injection molding's tooling problem) and by the wide variance in casting shop rates.

---

## Aggregate comparison: v1 vs v2

| Process | v1 MAPE | v2 MAPE | Improvement |
|---|---|---|---|
| CNC Milling | 16.2% | 8.4% | -7.8 pp |
| Sheet Metal Bending | 14.8% | 7.1% | -7.7 pp |
| Injection Molding | 22.1% | 12.3% | -9.8 pp |
| CNC Turning | 18.3% | 9.7% | -8.6 pp |
| Additive SLM | 19.4% | 10.2% | -9.2 pp |
| **Overall weighted** | **18.3%** | **9.6%** | **-8.7 pp** |

The consistent ~8-9 percentage point improvement across all processes in v2 is primarily attributable to two changes: (1) switching to per-process models, and (2) adding live material price as an input feature. These two changes together are responsible for most of the gain.

---

## What "confidence" level means in the API response

The `CostDecompositionModel.predict()` returns a `confidence` field:

- `high`: Process has ≥50 training samples, part falls within training distribution, material price data current
- `medium`: Process has 20–50 samples, or part is near the edge of training distribution
- `low`: Process has <20 samples, or part is out-of-distribution on ≥2 features

Low-confidence predictions are shown in the Optima interface with a warning indicator and a recommendation to request a manual supplier quote.

---

*Report generated as part of internship deliverable review, Week 10.*
