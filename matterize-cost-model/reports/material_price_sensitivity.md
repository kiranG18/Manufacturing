# Material Price Sensitivity Analysis
**Author:** Aris  
**Date:** Week 10  
**Focus:** Analyzing how fluctuations in raw material spot prices affect total part cost estimates across processes.

---

## Background

In manufacturing, raw material costs can make up anywhere from 5% to 70% of a part's total cost. For example:
- **CNC Milling / Turning:** Moderate sensitivity. Heavy raw stock is machined away, meaning buy-to-fly ratios are high (lots of wasted material), but metal commodity prices are moderately stable.
- **Injection Molding:** Low sensitivity. Polymers (ABS, PLA) are cheap, and tooling dominates the invoice on small batches.
- **Additive SLM (Metal 3D printing):** Very high sensitivity. Powdered metals (e.g., Titanium TI6AL4V) are extremely expensive ($35 - $80/kg), making material cost a primary driver.

This report documents how our v2 regression model responds to shifts in material prices and quantifies prediction drift when spot price updates are missed.

---

## Sensitivity Profiles by Process

To measure sensitivity, we held part geometry constant and shifted the `material_price_per_kg` feature between -50% and +50% of the historical baseline.

### 1. CNC Milling (AL6061 Aluminum, 2.7g/cm³, Qty = 50)
- **Baseline Price:** $2.80/kg
- **Material Cost Contribution:** ~25% of total
- **Sensitivity Coefficient:** **0.24** (a 10% increase in aluminum price results in a 2.4% increase in predicted total cost).
- **Behavior:** The model scales `raw_material_cost` linearly, while keeping `setup_cost` and `machine_time_cost` constant. This matches physical expectations.

### 2. Additive SLM (TI6AL4V Titanium, 4.43g/cm³, Qty = 10)
- **Baseline Price:** $35.00/kg
- **Material Cost Contribution:** ~55% of total
- **Sensitivity Coefficient:** **0.56** (a 10% increase in titanium powder price leads to a 5.6% increase in predicted total cost).
- **Behavior:** High sensitivity makes live price feeds critical here. A two-week lag in price updates during a titanium shortage could cause significant underquoting.

### 3. Injection Molding (ABS Polymer, 1.04g/cm³, Qty = 1000)
- **Baseline Price:** $2.10/kg
- **Material Cost Contribution:** ~8% of total (tooling amortized)
- **Sensitivity Coefficient:** **0.08** (a 10% increase in ABS price leads to a 0.8% increase in predicted total cost).
- **Behavior:** For injection molding, changes in raw polymer costs are negligible unless quantities reach the tens of thousands.

---

## Impact of Price Feed Lags (Staleness Analysis)

If the daily spot price feed fails, the system falls back to historical average pricing. We simulated feed outages of varying lengths to measure prediction drift against actual market rates during the 2024 aluminum price surge.

| Outage Duration | Pricing Mode | Average Error Drift (CNC Al) | Status |
|---|---|---|---|
| 1 day | Cache | 0.0% | Active Cache TTL (1 hr) handles this |
| 7 days | Fallback | +2.1% | Acceptable |
| 30 days | Fallback | +8.4% | Out of bounds — triggers system alert |

### Recommendation for System Operations
- **Alerting Threshold:** Trigger an operational slack alert if the `price_staleness_days` for any high-volume material (Aluminum, Stainless Steel) exceeds **7 days**.
- **User Warning:** In Optima, display a warning banner next to the cost estimate if `price_staleness_days > 7` to alert the sales representative that estimates are relying on fallback rates.
