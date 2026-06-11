# tests/test_order_extractor.py
import datetime
import pandas as pd
import pytest
from src.ingestion.order_extractor import OrderExtractor, OrderRecord

@pytest.fixture
def sample_orders_df():
    return pd.DataFrame([
        {
            "order_id": "ORD-001",
            "part_id": "PART-001",
            "process_code": "cnc_milling",
            "material_code": "AL6061",
            "quantity": 10,
            "completion_date": datetime.date(2026, 6, 1),
            "facility_id": "FACILITY_01",
            "cost_total": 500.0,
            "cost_label_source": "estimated"
        },
        {
            "order_id": "ORD-002",
            "part_id": "PART-002",
            "process_code": "cnc_turning",
            "material_code": "SS316L",
            "quantity": 5,
            "completion_date": datetime.date(2026, 6, 5),
            "facility_id": "FACILITY_02",
            "cost_total": 800.0,
            "cost_label_source": "estimated"
        },
        {
            "order_id": "ORD-003",
            "part_id": "PART-003",
            "process_code": "sheet_metal_bending",
            "material_code": "MS_A36",
            "quantity": 20,
            "completion_date": datetime.date(2026, 6, 10),
            "facility_id": "FACILITY_01",
            "cost_total": 300.0,
            "cost_label_source": "exact"
        }
    ])

def test_extract_all(sample_orders_df):
    extractor = OrderExtractor.from_dataframe(sample_orders_df)
    records = extractor.extract()
    assert len(records) == 3
    assert all(isinstance(r, OrderRecord) for r in records)
    assert records[0].order_id == "ORD-001"
    assert records[1].material_code == "SS316L"

def test_extract_date_filter(sample_orders_df):
    extractor = OrderExtractor.from_dataframe(sample_orders_df)
    
    # Filter for completion dates between 2026-06-03 and 2026-06-12
    records = extractor.extract(
        start_date=datetime.date(2026, 6, 3),
        end_date=datetime.date(2026, 6, 12)
    )
    assert len(records) == 2
    assert {r.order_id for r in records} == {"ORD-002", "ORD-003"}

def test_extract_facility_filter(sample_orders_df):
    extractor = OrderExtractor.from_dataframe(sample_orders_df)
    records = extractor.extract(facility_ids=["FACILITY_02"])
    assert len(records) == 1
    assert records[0].order_id == "ORD-002"
