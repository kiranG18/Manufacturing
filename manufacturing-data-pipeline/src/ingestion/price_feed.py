# src/ingestion/price_feed.py
#
# Fetches daily spot prices for raw materials from the external price API.
# Called once per pipeline run; results are archived in material_price_snapshots.csv.

import datetime
import json
import logging
import os
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Materials we track — superset of what the models currently use
TRACKED_MATERIALS = [
    "AL6061", "AL7075", "AL2024",
    "SS304", "SS316L", "SS17-4",
    "MS_A36", "MS_1018",
    "TI6AL4V",
    "PEEK", "POM", "PA66",
    "ABS", "PLA", "PETG",
    "INCONEL625",
]

# Historical averages used as fallback (USD/kg)
FALLBACK_PRICES = {
    "AL6061": 2.80, "AL7075": 4.10, "AL2024": 3.90,
    "SS304": 3.50, "SS316L": 5.20, "SS17-4": 8.40,
    "MS_A36": 0.95, "MS_1018": 1.10,
    "TI6AL4V": 35.00,
    "PEEK": 82.00, "POM": 4.20, "PA66": 3.80,
    "ABS": 2.10, "PLA": 2.00, "PETG": 2.40,
    "INCONEL625": 55.00,
}


class PriceFeedClient:
    """
    Retrieves current spot prices for tracked materials.
    Appends fetched prices to the price snapshot archive for reproducibility.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        snapshot_path: str = "data/processed/material_price_snapshots.csv",
    ):
        self._api_url = api_url or os.environ.get("PRICE_FEED_URL", "")
        self._api_key = api_key or os.environ.get("PRICE_FEED_API_KEY", "")
        self._snapshot_path = snapshot_path

    def fetch_today(self) -> Dict[str, float]:
        """
        Fetch current prices for all tracked materials.
        Falls back to historical averages if the API is unreachable.
        Returns {material_code: price_usd_per_kg}.
        """
        if self._api_url and self._api_key:
            try:
                prices = self._call_api()
                logger.info(f"Fetched live prices for {len(prices)} materials")
                self._append_to_snapshot(prices, source="live")
                return prices
            except Exception as e:
                logger.warning(f"Price feed unavailable ({e}). Falling back to historical averages.")

        prices = dict(FALLBACK_PRICES)
        self._append_to_snapshot(prices, source="fallback")
        logger.info("Using historical average prices (fallback)")
        return prices

    def get_price_for_date(self, material_code: str, target_date: datetime.date) -> Optional[float]:
        """
        Look up the archived price for a specific material on a specific date.
        Used when joining historical orders to their contemporaneous material prices.
        """
        if not os.path.exists(self._snapshot_path):
            return FALLBACK_PRICES.get(material_code)

        df = pd.read_csv(self._snapshot_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        match = df[
            (df["material_code"] == material_code) &
            (df["date"] == target_date)
        ]
        if match.empty:
            # Use closest available date within 7 days
            nearby = df[
                (df["material_code"] == material_code) &
                (abs((df["date"] - target_date).apply(lambda d: d.days)) <= 7)
            ]
            if not nearby.empty:
                return float(nearby.sort_values("date").iloc[-1]["price_usd_per_kg"])
            return FALLBACK_PRICES.get(material_code)

        return float(match.iloc[0]["price_usd_per_kg"])

    def _call_api(self) -> Dict[str, float]:
        import urllib.request
        url = f"{self._api_url}/v2/prices?materials={','.join(TRACKED_MATERIALS)}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return {item["code"]: float(item["price_usd_per_kg"]) for item in data["prices"]}

    def _append_to_snapshot(self, prices: Dict[str, float], source: str) -> None:
        today = datetime.date.today().isoformat()
        rows = [
            {"date": today, "material_code": code, "price_usd_per_kg": price, "source": source}
            for code, price in prices.items()
        ]
        new_df = pd.DataFrame(rows)
        if os.path.exists(self._snapshot_path):
            existing = pd.read_csv(self._snapshot_path)
            # Avoid duplicates for the same date
            existing = existing[~(
                (existing["date"] == today) & (existing["material_code"].isin(prices.keys()))
            )]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)
            combined = new_df
        combined.to_csv(self._snapshot_path, index=False)
        logger.debug(f"Price snapshot updated: {self._snapshot_path}")
