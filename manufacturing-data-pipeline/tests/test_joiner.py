# tests/test_joiner.py
import datetime
import pandas as pd
import pytest
from src.ingestion.order_extractor import OrderRecord
from src.ingestion.cad_metadata_loader import CADMetadataLoader
from src.ingestion.price_feed import PriceFeedClient
from src.ingestion.equipment_log_loader import EquipmentLogLoader
from src.transforms.joiner import OrderGeometryJoiner

@pytest.fixture
def mock_cad_df():
    return pd.DataFrame([
        {
            "part_id": "PART-001",
            "bounding_box_x": 100.0,
            "bounding_box_y": 50.0,
            "bounding_box_z": 10.0,
            "volume": 5000.0,
            "surface_area": 1200.0,
            "wall_thickness_min": 2.0,
            "wall_thickness_avg": 3.0,
            "aspect_ratio": 2.0,
            "hole_count": 2,
            "hole_diameter_min": 5.0,
            "undercut_flag": 0,
            "thin_wall_flag": 0,
            "curvature_complexity": 0.1,
            "symmetry_score": 0.8
        }
    ])

@pytest.fixture
def mock_equip_df():
    return pd.DataFrame([
        {
            "facility_id": "FACILITY_01",
            "machine_id": "M01",
            "process_code": "cnc_milling",
            "week_start": pd.to_datetime(datetime.date(2026, 5, 25)),
            "availability_pct": 90.0,
            "effective_hourly_rate_usd": 75.0,
            "downtime_hours": 0.0,
            "maintenance_notes": ""
        }
    ])

def test_joiner_happy_path(mock_cad_df, mock_equip_df):
    # Setup mock clients
    cad_loader = CADMetadataLoader.from_dataframe(mock_cad_df)
    
    price_feed = PriceFeedClient(api_url=None, api_key=None)
    # Patch get_price_for_date to return historical fallback or constant
    price_feed.get_price_for_date = lambda mat, dt: 2.80
    
    # Setup equipment loader
    equip_loader = EquipmentLogLoader(log_dir="dummy")
    equip_loader.load = lambda start_date=None, end_date=None, facility_ids=None: mock_equip_df
    
    joiner = OrderGeometryJoiner(cad_loader, price_feed, equip_loader)
    
    # Setup test order
    orders = [
        OrderRecord(
            order_id="ORD-001",
            part_id="PART-001",
            process_code="cnc_milling",
            material_code="AL6061",
            quantity=10,
            batch_size=10,
            surface_finish_ra=3.2,
            tolerance_band="IT8",
            order_date=datetime.date(2026, 6, 1),
            completion_date=datetime.date(2026, 6, 2),
            facility_id="FACILITY_01",
            cost_total=300.0,
            cost_machine_time=None,
            cost_setup=None,
            cost_tooling=None,
            cost_raw_material=None,
            cost_finishing=None,
            cost_label_source="estimated"
        )
    ]
    
    res_df = joiner.join(orders)
    
    assert len(res_df) == 1
    assert res_df.iloc[0]["part_id"] == "PART-001"
    assert res_df.iloc[0]["volume"] == 5000.0
    assert res_df.iloc[0]["material_price_per_kg"] == 2.80
    assert res_df.iloc[0]["effective_hourly_rate"] == 75.0

def test_joiner_missing_geometry_filtered(mock_equip_df):
    # Setup cad loader with empty cache
    cad_loader = CADMetadataLoader.from_dataframe(pd.DataFrame(columns=["part_id"]))
    price_feed = PriceFeedClient(api_url=None, api_key=None)
    price_feed.get_price_for_date = lambda mat, dt: 2.80
    
    equip_loader = EquipmentLogLoader(log_dir="dummy")
    equip_loader.load = lambda start_date=None, end_date=None, facility_ids=None: mock_equip_df
    
    joiner = OrderGeometryJoiner(cad_loader, price_feed, equip_loader)
    
    orders = [
        OrderRecord(
            order_id="ORD-001",
            part_id="PART-001",
            process_code="cnc_milling",
            material_code="AL6061",
            quantity=10,
            batch_size=10,
            surface_finish_ra=3.2,
            tolerance_band="IT8",
            order_date=datetime.date(2026, 6, 1),
            completion_date=datetime.date(2026, 6, 2),
            facility_id="FACILITY_01",
            cost_total=300.0,
            cost_machine_time=None,
            cost_setup=None,
            cost_tooling=None,
            cost_raw_material=None,
            cost_finishing=None,
            cost_label_source="estimated"
        )
    ]
    
    res_df = joiner.join(orders)
    # The order has no matching geometry, so it should be filtered out
    assert res_df.empty
