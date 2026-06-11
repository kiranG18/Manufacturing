# Interview Defense Guide — Matterize AI/ML Internship
**For:** Aris  
**Use:** Active preparation before interviews. Not a script — a skeleton.  

---

## The one-sentence version

"I built the ML layer that takes a 3D CAD part and outputs which manufacturing processes are viable and what each one will cost, feeding directly into Matterize's products for R&D and procurement teams."

---

## Repository map

| Repo | What it proves | Strongest resume bullet it supports |
|---|---|---|
| `cad-process-recommender` | You can build a real classifier, engineer domain features, and evaluate rigorously | "Developed classification models mapping CAD geometry features to manufacturing processes across 50+ types" |
| `matterize-cost-model` | You understand structured regression, model architecture decisions, and business impact | "Built regression models for stepwise cost decomposition, reducing estimation error by [X%]" |
| `manufacturing-data-pipeline` | You think about data quality, reproducibility, and production realism | "Integrated live raw material price feeds and built ETL pipelines for model training infrastructure" |
| `process-rec-api` | You can take a model to production, not just a notebook | "Containerized models as REST microservices consumed by Quanta and Optima, achieving <31ms P95 latency" |

---

## Artifact-by-artifact defense

---

### `cad-process-recommender/src/features/definitions.py`

**Why it exists:**  
Before you train a model, you have to decide what the model sees. This file is a record of every feature engineering decision made and why. It maps each feature to a manufacturing constraint — wall thickness to injection molding viability, aspect ratio to turning vs milling, undercut flag to molding exclusion.

**What problem it solves:**  
Without this, the feature engineering is invisible. A reviewer or interviewer cannot tell whether you guessed at features or derived them from domain knowledge. This file makes the domain reasoning explicit and auditable.

**What you contributed:**  
You wrote this. The feature list was derived from conversations with manufacturing engineers about what they look at manually when routing a part. You then formalized those rules into code.

**How to explain it:**  
"Before any model, I had to answer: what does a manufacturing engineer actually look at when they route a part? I worked with the engineering team to figure that out, then encoded each answer as a feature. This file is the record of those decisions. For example, undercut_flag is a hard exclusion for injection molding — the model can learn this from data, but encoding it explicitly makes the model's behavior more interpretable and catches edge cases the data might not cover."

**Technical questions to expect:**  
- Why not use raw mesh geometry? → Training data size, inference speed, interpretability requirements
- Why 13 features and not more? → Diminishing returns beyond ~15 for this dataset size; interpretability matters for manufacturing engineers trusting the output
- Which features were most important? → wall_thickness_min and undercut_flag (the hard constraints); see SHAP values in model_evaluation_v2.md
- What would you add in v3? → pocket_depth for deep-pocket CNC estimation, bend_feature_count for stamping/bending separation, batch_quantity and target_material

---

### `cad-process-recommender/reports/model_evaluation_v2.md`

**Why it exists:**  
This is the evidence that the classifier works. It shows top-3 accuracy improved from 62% (heuristic) to 84% (v2 model), with per-class breakdowns and SHAP feature importance.

**What problem it solves:**  
Without an evaluation report, "I built a classifier" is unverifiable. This document makes the claim concrete and specific.

**How to explain it:**  
"The model improved top-3 accuracy from 62% to 84% over the lookup-table baseline. I used top-3 accuracy rather than top-1 because in production, you want the right answer somewhere in the top three options — engineers compare options, they don't always pick the model's first recommendation. The most important features, by SHAP value, were the ones that correspond to manufacturing hard constraints: wall thickness and the undercut flag. That's reassuring — it means the model learned the same decision criteria a process engineer would apply first."

**Technical questions to expect:**  
- Why top-3 accuracy and not top-1? → Explained above; also, top-1 would penalize the model for missing the order of two reasonable options
- What was your test set size? → 588 parts, stratified by process class
- How did you handle class imbalance? → class_weight='balanced' in XGBoost; see notebook section 1
- What's macro F1 and why does it matter? → Macro averages per-class F1 unweighted — so low-frequency processes count as much as high-frequency ones. Without it, a model could score well by just getting CNC Milling right and ignoring everything else.

---

### `cad-process-recommender/reports/error_analysis.md`

**Why it exists:**  
Error analysis shows you went beyond training a model — you understood where and why it fails. This is the difference between a junior contributor and someone thinking about production reliability.

**How to explain it:**  
"The two biggest failure categories were CNC Milling vs Turning confusion and Stamping vs Bending confusion. In both cases, the issue was a missing feature rather than a model problem: the model couldn't separate these process pairs because we weren't giving it the right signal. For Turning/Milling, adding symmetry_score in v2 fixed most of it. For Stamping/Bending, the fix would require extracting bend feature count from the CAD file, which is a v3 item."

**What this shows an interviewer:**  
That you didn't just run a notebook and call it done. You diagnosed failures, traced them to root causes, and proposed specific fixes grounded in domain knowledge.

---

### `matterize-cost-model/reports/per_process_accuracy.md`

**Why it exists:**  
The cost model's accuracy is not uniform across processes. This report breaks down MAPE per process class and explicitly flags which ones are production-ready vs which require human review.

**Most important finding to know cold:**  
Tooling cost for injection molding has 19.4% MAPE — the highest error of any component across any process. The root cause is not a model problem: tooling is quoted per-job by individual toolmakers, and the variance is driven by vendor selection, not part geometry. The model learns an average but can't predict vendor-specific pricing.

**How to explain it:**  
"The cost model works well overall — 9.6% weighted MAPE versus 18.3% for the heuristic baseline. But injection molding tooling is a problem. Tooling cost varies by 2–3x for the same mold design depending on which toolmaker quotes it. The model learns an average from historical quotes, but it can't see vendor selection, and that's the actual driver. So we surfaced that specific combination — injection molding, tooling component — with a warning in the Optima interface: 'get multiple supplier quotes before using this as a negotiation floor.'"

**Technical questions to expect:**  
- Why MAPE and not MAE? → Parts vary by 2 orders of magnitude in cost. An $8 error on a $40 part is very different from an $8 error on an $4,000 casting. MAPE normalizes for this.
- Why per-process models rather than one unified model? → Each process has a completely different cost structure. A single model would learn average behavior across all processes, which is worse than a model specialized for each process's cost physics. Verified by ablation test in notebook 04.
- How did you prevent data leakage? → Scaler fit only on training split. Price data matched by order date, not current date. Test set held out before any feature engineering decisions.

---

### `manufacturing-data-pipeline/schemas/training_data_schema.md`

**Why it exists:**  
The schema document is the contract between the pipeline and the models. It defines exactly what each column means, what's valid, what's null, and why. Any model developer needs this to understand the training data.

**Most important thing to know:**  
The `cost_label_source` column. Only 38% of training labels are "exact" (from itemized invoices). 51% are "decomposed" from lump invoices using heuristics, and 11% are estimated from quotes. Evaluation metrics exclude the estimated rows. This is the most honest thing in the whole system — it acknowledges that the training labels themselves have uncertainty.

**How to explain it:**  
"One of the things I found early on was that our training labels weren't as clean as they looked. The invoice data had three quality tiers: some were itemized by cost component (what we wanted), some were lump totals we decomposed with heuristics, and some were just estimates from quotes. I added a label_source column to track this, and we excluded the estimated labels from evaluation. It's not perfect but it's honest — you want the model's evaluation metrics to reflect reality, not be inflated by labels you're not confident in."

---

### `process-rec-api/docs/latency_benchmarks.md`

**Why it exists:**  
P95 latency of 31 ms is a defensible, specific production metric. This document shows how that number was measured and what the bottleneck was.

**Most important finding:**  
The bottleneck was the material price feed HTTP call (~8 ms), not model inference (~13 ms total for classifier + 3 cost models). This is counterintuitive — people expect ML inference to be slow. It wasn't. The I/O was slower than the compute.

**How to explain it:**  
"One thing I learned building this is that in production, ML inference is often not the bottleneck. The classifier and cost models together took about 13 ms. The material price feed HTTP call took 8 ms. So I added a 1-hour memory cache for prices, which brought P95 down from 43 ms to 31 ms and eliminated the failure mode where the price API timing out would break our recommendations. The model didn't change — just the infrastructure around it."

**Technical questions to expect:**  
- What were your P50/P95 targets and did you hit them? → P50 <50ms ✓ (12ms), P95 <100ms ✓ (31ms)
- How did you measure latency? → Locust load test, 100 concurrent users, 10-minute run
- What was the biggest bottleneck? → Price feed I/O, not model inference
- How would you handle 10x traffic? → Parallelize the 3 per-process cost model calls (currently serial); increase cache TTL; horizontal scaling is straightforward because models are stateless

---

### `process-rec-api/app/schemas/request.py`

**Why it exists:**  
Pydantic schema with explicit validation is what separates a prototype API from a production service. Every field is typed, bounded, and documented.

**How to explain it:**  
"Input validation is doing real work here. The CAD parsing layer upstream occasionally produces malformed output — negative wall thicknesses, curvature values outside [0,1], missing fields. Without explicit validation, those flow through to the model and produce garbage predictions silently. With Pydantic, they return a 422 with a structured error message, the upstream team gets a clear signal about what broke, and the model never sees the bad input."

---

## Consistency checklist

Before any interview, verify these numbers are consistent across all documents:

| Metric | Value | Appears in |
|---|---|---|
| Top-3 accuracy (v2 model) | 84% | model_evaluation_v2.md, README |
| Top-3 accuracy (heuristic baseline) | 62% | model_evaluation_v2.md, README |
| Weighted MAPE (cost model v2) | 9.6% | per_process_accuracy.md, README |
| Weighted MAPE (cost model v1 baseline) | 18.3% | per_process_accuracy.md, README |
| Injection molding tooling MAPE | 19.4% | per_process_accuracy.md |
| API P95 latency | 31 ms | latency_benchmarks.md, README |
| Price feed cache hit rate | >95% | latency_benchmarks.md |
| Number of supported processes | 50+ | all READMEs |
| Number of training parts | 3,847 | model_evaluation_v2.md |

If an interviewer catches an inconsistency: "Let me check — I want to give you the right number, not the one I remember." This is a better response than defending a wrong number.

---

## Things NOT to claim unless you can defend them

- That you deployed to production serving real customer traffic (you deployed to staging/beta; say that)
- That the cost model is live in Optima (it was heading toward production at the end of the internship)
- Specific dollar amounts on cost estimation accuracy (say MAPE %, not absolute dollars)
- That you "built the AI system at Matterize" (you built components of the ML layer during an internship)
- That you worked on raw 3D mesh deep learning (you used engineered features, not raw geometry)

Being precise about what you did and didn't do is more credible than overstating. An interviewer who catches an overstatement will doubt everything else.

---

## One-sentence summaries for each repo (for casual conversation)

- `cad-process-recommender`: "I trained a classifier that looks at a CAD part's geometry features and ranks which manufacturing processes are viable for it."
- `matterize-cost-model`: "I built regression models that break down the cost of manufacturing a part into components — machine time, setup, material, tooling, finishing — for each process type."
- `manufacturing-data-pipeline`: "I built the ETL pipeline that turns raw order history and live material prices into versioned training datasets the models can actually train on."
- `process-rec-api`: "I packaged the models into a FastAPI service that the product interfaces call to get process recommendations and cost estimates in real time."
