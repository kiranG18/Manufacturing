# tests/test_price_fetcher.py
import datetime
import pytest
from src.features.price_fetcher import MaterialPriceFetcher, HISTORICAL_AVG_PRICES

def test_price_fetcher_historical_fallback():
    # When no API config is provided, it must fall back to historical prices
    fetcher = MaterialPriceFetcher(api_url=None, api_key=None)
    price, date = fetcher.get_price("AL6061")
    assert price == HISTORICAL_AVG_PRICES["AL6061"]
    
    # Staleness date should be far in the past (999 days)
    days_diff = (datetime.date.today() - date).days
    assert days_diff == 999

def test_price_fetcher_fallback_unknown_material():
    fetcher = MaterialPriceFetcher()
    price, date = fetcher.get_price("MY_SECRET_FABULOUS_METAL")
    # Default fallback price should be $5.00
    assert price == 5.00

def test_get_price_with_staleness():
    fetcher = MaterialPriceFetcher()
    res = fetcher.get_price_with_staleness("SS316L")
    assert "price_usd_per_kg" in res
    assert "price_date" in res
    assert res["staleness_days"] == 999
    assert res["is_live"] is False

def test_supported_materials():
    fetcher = MaterialPriceFetcher()
    materials = fetcher.supported_materials()
    assert "AL6061" in materials
    assert "SS316L" in materials
    assert "PEEK" in materials
