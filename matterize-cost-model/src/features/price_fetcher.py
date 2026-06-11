# src/features/price_fetcher.py
#
# MaterialPriceFetcher
# --------------------
# Fetches current spot prices for raw materials used in manufacturing.
# In production this hits an internal price aggregation API that itself
# pulls from LME (London Metal Exchange), Platts, and polymer index feeds.
#
# For this codebase, the fetcher has a local cache and a static fallback
# so tests and offline development work without network access.

import datetime
import json
import os
from typing import Optional, Tuple


# Historical average prices (USD/kg) used as fallback when the live feed
# is unavailable. Updated quarterly — stale prices are flagged in the
# API response so consumers know to treat the estimate with caution.
HISTORICAL_AVG_PRICES = {
    "AL6061":  2.80,
    "AL7075":  4.10,
    "AL2024":  3.90,
    "SS304":   3.50,
    "SS316L":  5.20,
    "SS17-4":  8.40,
    "MS_A36":  0.95,
    "MS_1018": 1.10,
    "TI6AL4V": 35.00,
    "PEEK":    82.00,
    "POM":     4.20,
    "PA66":    3.80,
    "ABS":     2.10,
    "PLA":     2.00,
    "PETG":    2.40,
    "CARBON_FIBER": 25.00,
    "INCONEL625": 55.00,
}

# In-memory price cache: {material_code: (price_usd_per_kg, date)}
_PRICE_CACHE: dict = {}
_CACHE_TTL_HOURS = 1


class MaterialPriceFetcher:
    """
    Retrieves current raw material spot prices.

    Caches results in memory with a 1-hour TTL to avoid hammering the price
    API on every inference request. If the cache is fresh, the price feed
    latency contribution to overall API response time is < 0.1 ms.

    If the external feed fails, falls back to historical averages and
    sets a staleness flag so the API can communicate uncertainty to the user.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_ttl_hours: int = _CACHE_TTL_HOURS,
    ):
        self._api_url = api_url or os.environ.get("PRICE_FEED_URL", "")
        self._api_key = api_key or os.environ.get("PRICE_FEED_API_KEY", "")
        self._cache_ttl = datetime.timedelta(hours=cache_ttl_hours)

    def get_price(self, material_code: str) -> Tuple[float, datetime.date]:
        """
        Returns (price_usd_per_kg, price_date).

        Attempts live fetch first; falls back to cache; falls back to
        historical average. Caller should check price_date staleness.
        """
        # Check memory cache
        cached = _PRICE_CACHE.get(material_code)
        if cached is not None:
            price, fetched_at, date = cached
            if datetime.datetime.utcnow() - fetched_at < self._cache_ttl:
                return price, date

        # Try live feed
        if self._api_url and self._api_key:
            try:
                price, date = self._fetch_live(material_code)
                _PRICE_CACHE[material_code] = (price, datetime.datetime.utcnow(), date)
                return price, date
            except Exception:
                # Feed failure is not fatal — fall through to historical
                pass

        # Historical fallback
        price = HISTORICAL_AVG_PRICES.get(material_code, 5.00)
        # Use a stale date so callers can detect the fallback
        stale_date = datetime.date.today() - datetime.timedelta(days=999)
        return price, stale_date

    def get_price_with_staleness(self, material_code: str) -> dict:
        """
        Convenience wrapper that also computes staleness_days.
        Returns a dict with price, date, and staleness_days.
        """
        price, date = self.get_price(material_code)
        staleness = (datetime.date.today() - date).days
        return {
            "price_usd_per_kg": price,
            "price_date": date.isoformat(),
            "staleness_days": staleness,
            "is_live": staleness < 7,
        }

    def _fetch_live(self, material_code: str) -> Tuple[float, datetime.date]:
        """
        Calls the internal price API. Raises on any error so the caller
        can fall back gracefully.
        """
        import urllib.request
        url = f"{self._api_url}/prices/{material_code}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        price = float(data["price_usd_per_kg"])
        date = datetime.date.fromisoformat(data["date"])
        return price, date

    @staticmethod
    def supported_materials() -> list:
        return sorted(HISTORICAL_AVG_PRICES.keys())

    @staticmethod
    def clear_cache() -> None:
        """Force cache expiry — used in tests."""
        _PRICE_CACHE.clear()
