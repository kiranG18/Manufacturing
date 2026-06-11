# tests/test_coverage_checker.py
import pandas as pd
import pytest
from src.validation.coverage_checker import CoverageChecker

def test_coverage_checker_sufficient():
    checker = CoverageChecker(min_samples_per_process=5, warn_threshold=10)
    
    # Create df where cnc_milling has 12 rows (>= 10) and injection_molding has 11 rows (>= 10)
    df = pd.DataFrame(
        [{"process_code": "cnc_milling", "material_code": "AL6061", "price_staleness_days": 1}] * 12 +
        [{"process_code": "injection_molding", "material_code": "ABS", "price_staleness_days": 2}] * 11
    )
    
    is_valid, stats, warnings = checker.check_coverage(df)
    
    assert is_valid is True
    assert len(warnings) == 0
    assert stats["process_coverage"]["cnc_milling"] == 12

def test_coverage_checker_fail_threshold():
    # cnc_milling has 12 rows (>= 10), sand_casting has 3 rows (< 5)
    checker = CoverageChecker(min_samples_per_process=5, warn_threshold=10)
    
    df = pd.DataFrame(
        [{"process_code": "cnc_milling", "material_code": "AL6061", "price_staleness_days": 1}] * 12 +
        [{"process_code": "sand_casting", "material_code": "AL6061", "price_staleness_days": 1}] * 3
    )
    
    is_valid, stats, warnings = checker.check_coverage(df)
    
    assert is_valid is False
    assert len(warnings) > 0
    assert any("sand_casting" in w and "Below absolute minimum" in w for w in warnings)

def test_coverage_checker_stale_prices():
    checker = CoverageChecker(min_samples_per_process=5, warn_threshold=10)
    
    df = pd.DataFrame(
        [{"process_code": "cnc_milling", "material_code": "AL6061", "price_staleness_days": 10}] * 12
    )
    
    is_valid, stats, warnings = checker.check_coverage(df)
    
    # price staleness > 7 should produce a warning but not make it invalid (unless process counts fail)
    assert is_valid is True
    assert len(warnings) == 1
    assert "stale" in warnings[0]
