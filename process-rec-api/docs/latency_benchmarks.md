# API Latency Benchmarks
**Date:** Week 10  
**Author:** Aris  
**Environment:** Docker container, 2 vCPU, 4GB RAM (matches staging environment)  
**Test tool:** `locust` load test, 100 concurrent users, 10-minute run  

---

## Summary

The API meets its latency targets at the load levels expected for the Quanta/Optima beta launch. The dominant latency source is the material price feed lookup, not model inference itself. A caching layer was added in v2.1 to reduce this.

---

## Results table

| Endpoint | P50 | P95 | P99 | Max | Requests/sec |
|---|---|---|---|---|---|
| POST /v1/recommend (with live prices) | 12 ms | 31 ms | 67 ms | 142 ms | 48 |
| POST /v1/recommend (cached prices) | 8 ms | 19 ms | 41 ms | 88 ms | 71 |
| GET /v1/debug/{part_id} | 28 ms | 54 ms | 94 ms | 201 ms | 22 |
| GET /health | 1 ms | 2 ms | 3 ms | 8 ms | 800 |

---

## Latency breakdown (P95, /v1/recommend with live prices)

| Component | Time | Notes |
|---|---|---|
| Request parsing + Pydantic validation | ~1 ms | Negligible |
| Feature extraction (from geometry dict) | ~2 ms | Pure Python, no I/O |
| Price feed lookup | ~8 ms | External HTTP call (daily cache in v2.1) |
| Process classifier inference | ~4 ms | XGBoost on 13 features |
| Cost model inference (×3 processes) | ~9 ms | 3 × ~3ms per process model |
| Constraint filter application | <1 ms | In-memory rule evaluation |
| SHAP (only when include_debug=true) | ~25 ms | Not on critical path |
| Response serialization | ~1 ms | Pydantic |
| **Total** | **~26 ms** | P50 observed; P95 adds network variance |

---

## Price feed caching (v2.1 change)

In v2.0, every request made a fresh HTTP call to the material price feed. This added 6–12 ms of latency per request and introduced a hard dependency: if the price feed was down, the API returned 503.

In v2.1, prices are cached in memory with a 1-hour TTL. On cache hit (>95% of requests), price lookup takes <0.1 ms. On cache miss, the external call is made and the cache is refreshed. If the external call fails, the cache serves stale prices with a warning in the response.

This change reduced P95 latency by ~12 ms and improved availability.

---

## Error rates

| Error type | Rate | Notes |
|---|---|---|
| 422 Validation errors | 0.3% | Bad input from CAD parser (known edge cases) |
| 503 Price feed timeout | 0.0% | Eliminated by caching in v2.1 |
| 500 Internal errors | 0.02% | Under investigation — suspected NaN in curvature_samples |

---

## Load test configuration

```python
# locustfile.py (summary)

class RecommendUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def recommend(self):
        # Randomly selected from 12 sample parts in tests/fixtures/
        payload = random.choice(SAMPLE_REQUESTS)
        self.client.post("/v1/recommend", json=payload)
```

Test ran for 10 minutes with 100 concurrent users simulating the expected beta load (50–100 active R&D engineers using Quanta simultaneously during a product launch window).

---

## Notes for production scaling

At 100 concurrent users, the API is comfortably within limits on 2 vCPU. For 500+ concurrent users, the cost model inference (3 × serial calls per request) would become the bottleneck. Recommendation: parallelize the 3 per-process cost model calls using `asyncio.gather` or a thread pool, which would cut the cost inference time from ~9 ms to ~3 ms.

This was flagged as a v3 optimization item; not needed for beta.
