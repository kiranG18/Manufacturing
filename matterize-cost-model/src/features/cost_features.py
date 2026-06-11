# src/features/cost_features.py
#
# CostFeatureBuilder
# ------------------
# Assembles the full feature vector for cost decomposition inference.
# Combines:
#   1. Geometry features (from cad-process-recommender feature extractor)
#   2. Process context (selected process, quantity, batch size)
#   3. Material specification and live market price
#   4. Quality/finish requirements (surface finish, tolerance band)
#
# The geometry features are expected to already be computed upstream.
# This module handles the additional cost-specific inputs and joins
# everything into a flat dict ready for the per-process regressor.

import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

from .price_fetcher import MaterialPriceFetcher


# Tolerance band → numeric index (ISO 286 IT grades)
# Tighter tolerance = more machine time = higher cost
TOLERANCE_IT_INDEX = {
    "IT4": 1, "IT5": 2, "IT6": 3, "IT7": 4,
    "IT8": 5, "IT9": 6, "IT10": 7, "IT11": 8, "IT12": 9,
}

# Surface finish spec → numeric: Ra value in microns
# Lower Ra = finer finish = higher finishing cost
SURFACE_FINISH_DEFAULTS = {
    "as_machined": 3.2,
    "fine": 1.6,
    "very_fine": 0.8,
    "mirror": 0.1,
}


@dataclass
class CostInput:
    """
    All non-geometry inputs needed to build a cost feature vector.
    Collected from the user-facing form in Quanta/Optima.
    """
    process_code: str               # e.g. "cnc_milling", "injection_molding"
    material_code: str              # e.g. "AL6061", "SS316L", "PEEK"
    quantity: int                   # Number of parts in the order
    surface_finish_spec: float      # Ra value in microns (or use SURFACE_FINISH_DEFAULTS)
    tolerance_band: str             # IT grade string, e.g. "IT8"
    batch_size: Optional[int] = None  # If different from quantity (e.g. production run size)


class CostFeatureBuilder:
    """
    Builds the complete feature dict for cost decomposition.

    Usage:
        builder = CostFeatureBuilder(live_prices=True)
        features = builder.build(geometry_features, cost_input)
    """

    def __init__(self, live_prices: bool = True):
        self.live_prices = live_prices
        self._price_fetcher = MaterialPriceFetcher() if live_prices else None

    def build(
        self,
        geometry_features: Dict[str, float],
        cost_input: CostInput,
    ) -> Dict[str, float]:
        """
        Joins geometry features with cost-specific inputs.
        Returns a flat dict for the regressor.
        """
        features = dict(geometry_features)  # copy — don't mutate the input

        # Process context
        features["process_code_encoded"] = self._encode_process(cost_input.process_code)
        features["quantity"] = float(cost_input.quantity)
        features["log_quantity"] = self._safe_log(cost_input.quantity)
        features["batch_size"] = float(cost_input.batch_size or cost_input.quantity)

        # Quality requirements
        features["surface_finish_ra"] = cost_input.surface_finish_spec
        features["tolerance_index"] = TOLERANCE_IT_INDEX.get(cost_input.tolerance_band, 5)

        # Material + live price
        features["material_code_encoded"] = self._encode_material(cost_input.material_code)

        if self.live_prices and self._price_fetcher:
            price, price_date = self._price_fetcher.get_price(cost_input.material_code)
            features["material_price_per_kg"] = price
            features["price_date"] = price_date.isoformat()
            features["price_staleness_days"] = (
                datetime.date.today() - price_date
            ).days
        else:
            # Fallback: use historical average from training data distribution
            features["material_price_per_kg"] = self._historical_avg_price(cost_input.material_code)
            features["price_staleness_days"] = 999  # signals stale price in downstream logic

        # Derived cost signals
        features["volume_x_price"] = (
            features.get("volume", 0) * features["material_price_per_kg"] / 1e6
        )  # rough material cost proxy in $ before model correction

        return features

    # --- Encoding helpers ---
    # These use ordinal encoding rather than one-hot to keep the feature vector
    # consistent size regardless of how many processes/materials are in the catalog.
    # Label encoders are fit on the full catalog and stored in configs/.

    def _encode_process(self, process_code: str) -> int:
        # In production: loaded from configs/process_routing.yaml
        # Placeholder mapping shown here for documentation
        PROCESS_ENCODING = {
            "cnc_milling": 1, "cnc_turning": 2, "sheet_metal_bending": 3,
            "sheet_metal_stamping": 4, "injection_molding": 5, "die_casting": 6,
            "sand_casting": 7, "investment_casting": 8, "additive_fdm": 9,
            "additive_slm": 10, "laser_cutting": 11, "forging": 12, "extrusion": 13,
        }
        return PROCESS_ENCODING.get(process_code, 0)  # 0 = unknown

    def _encode_material(self, material_code: str) -> int:
        # Materials grouped by family for ordinal encoding
        MATERIAL_FAMILY = {
            "AL6061": 1, "AL7075": 1, "AL2024": 1,   # Aluminum
            "SS304": 2, "SS316L": 2, "SS17-4": 2,    # Stainless steel
            "MS_A36": 3, "MS_1018": 3,               # Mild steel
            "TI6AL4V": 4,                             # Titanium
            "PEEK": 5, "POM": 5, "PA66": 5,          # Engineering polymers
            "ABS": 6, "PLA": 6, "PETG": 6,           # Commodity polymers
        }
        return MATERIAL_FAMILY.get(material_code, 0)

    def _safe_log(self, value: float) -> float:
        import math
        return math.log(value + 1) if value >= 0 else 0.0

    def _historical_avg_price(self, material_code: str) -> float:
        # Historical average prices (USD/kg) from training data period
        # Used as fallback when live feed is unavailable
        HISTORICAL_PRICES = {
            "AL6061": 2.80, "AL7075": 4.10, "SS304": 3.50,
            "SS316L": 5.20, "TI6AL4V": 35.00, "PEEK": 80.00,
            "ABS": 2.10, "PLA": 2.00,
        }
        return HISTORICAL_PRICES.get(material_code, 5.00)  # default: $5/kg
