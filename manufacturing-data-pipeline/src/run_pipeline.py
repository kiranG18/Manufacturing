# src/run_pipeline.py
import argparse
import datetime
import logging
import os
import sys
from typing import Tuple, Dict, Any, List, Optional
import yaml
import pandas as pd
import numpy as np

from src.ingestion.order_extractor import OrderExtractor, OrderRecord
from src.ingestion.cad_metadata_loader import CADMetadataLoader
from src.ingestion.price_feed import PriceFeedClient
from src.ingestion.equipment_log_loader import EquipmentLogLoader

from src.transforms.joiner import OrderGeometryJoiner
from src.transforms.normalizer import DatasetNormalizer
from src.transforms.label_builder import CostLabelBuilder
from src.transforms.versioner import DatasetVersioner

from src.validation.schema_validator import SchemaValidator
from src.validation.coverage_checker import CoverageChecker
from src.validation.drift_detector import DriftDetector

from src.loaders.dataset_writer import DatasetWriter
from src.loaders.catalog_updater import CatalogUpdater

logger = logging.getLogger("run_pipeline")


def setup_logging(log_level_str: str, log_dir: str):
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_pipeline_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def generate_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates mock order, geometry, and equipment log DataFrames for pipeline testing."""
    logger.info("Generating mock data for pipeline validation run")
    np.random.seed(42)

    # 1. Orders
    n_orders = 100
    order_ids = [f"ORD-20260611-{i:03d}" for i in range(n_orders)]
    part_ids = [f"PART-SAMPLE-{i:03d}" for i in range(n_orders)]
    processes = ["cnc_milling", "cnc_turning", "sheet_metal_bending", "injection_molding", "additive_slm"]
    materials = ["AL6061", "SS316L", "MS_A36", "TI6AL4V", "PEEK", "ABS"]
    facilities = ["FACILITY_01", "FACILITY_02"]

    orders_rows = []
    for i in range(n_orders):
        qty = int(np.random.choice([10, 50, 100, 500, 1000]))
        orders_rows.append({
            "order_id": order_ids[i],
            "part_id": part_ids[i],
            "process_code": np.random.choice(processes),
            "material_code": np.random.choice(materials),
            "quantity": qty,
            "batch_size": qty,
            "surface_finish_ra": np.random.choice([3.2, 1.6, 0.8]),
            "tolerance_band": np.random.choice(["IT8", "IT9", "IT10"]),
            "order_date": datetime.date.today() - datetime.timedelta(days=int(np.random.randint(1, 30))),
            "completion_date": datetime.date.today(),
            "facility_id": np.random.choice(facilities),
            "cost_total": float(np.random.randint(100, 5000)),
            "cost_machine_time": None,  # let label builder fill these
            "cost_setup": None,
            "cost_tooling": None,
            "cost_raw_material": None,
            "cost_finishing": None,
            "cost_label_source": "estimated"
        })
    orders_df = pd.DataFrame(orders_rows)

    # 2. Geometry metadata
    geom_rows = []
    for pid in part_ids:
        geom_rows.append({
            "part_id": pid,
            "bounding_box_x": float(np.random.uniform(10, 500)),
            "bounding_box_y": float(np.random.uniform(10, 500)),
            "bounding_box_z": float(np.random.uniform(2, 200)),
            "volume": float(np.random.uniform(500, 500000)),
            "surface_area": float(np.random.uniform(1000, 50000)),
            "wall_thickness_min": float(np.random.uniform(0.5, 10)),
            "wall_thickness_avg": float(np.random.uniform(1, 15)),
            "aspect_ratio": float(np.random.uniform(1, 10)),
            "hole_count": int(np.random.randint(0, 20)),
            "hole_diameter_min": float(np.random.uniform(1.0, 20.0)),
            "undercut_flag": int(np.random.choice([0, 1], p=[0.8, 0.2])),
            "thin_wall_flag": 0,  # normalizer will derive
            "curvature_complexity": float(np.random.uniform(0, 0.5)),
            "symmetry_score": float(np.random.uniform(0.1, 1.0))
        })
    geom_df = pd.DataFrame(geom_rows)

    # 3. Equipment log
    equip_rows = []
    week_start = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=45))
    for fac in facilities:
        for proc in processes:
            equip_rows.append({
                "facility_id": fac,
                "machine_id": f"MACH_{proc.upper()}_01",
                "process_code": proc,
                "week_start": week_start,
                "availability_pct": 85.0,
                "effective_hourly_rate_usd": float(np.random.randint(30, 120)),
                "downtime_hours": 8.0,
                "maintenance_notes": "Routine servicing"
            })
    equip_df = pd.DataFrame(equip_rows)

    return orders_df, geom_df, equip_df


def main():
    parser = argparse.ArgumentParser(description="Matterize Manufacturing Data Pipeline")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Path to config YAML")
    parser.add_argument("--start-date", help="Start completion date for extraction (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End completion date for extraction (YYYY-MM-DD)")
    parser.add_argument("--validate-only", action="store_true", help="Validate data without writing to catalogs")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Force mock data generation for portfolio testing")

    args = parser.parse_args()

    # Load config
    try:
        config = load_pipeline_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        # Default fallback config structure
        config = {
            "pipeline": {
                "log_level": "INFO",
                "log_dir": "logs/pipeline_runs",
                "min_samples_per_class": 10
            }
        }

    # Setup Logging
    pipeline_cfg = config.get("pipeline", {})
    setup_logging(
        log_level_str=pipeline_cfg.get("log_level", "INFO"),
        log_dir=pipeline_cfg.get("log_dir", "logs/pipeline_runs")
    )

    logger.info("Initializing Manufacturing Data Pipeline Run")

    # Resolve dates
    start_date = None
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = None
    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    orders_df, geom_df, equip_df = None, None, None

    # Load data sources
    if args.test_mode or not os.environ.get("ORDER_DB_HOST"):
        logger.info("Running in offline/test mode. Generating synthetic dataset inputs.")
        orders_df, geom_df, equip_df = generate_mock_data()
        
        # Save mock equipment log for equipment loader to find
        os.makedirs("data/raw", exist_ok=True)
        equip_df.to_csv("data/raw/equipment_log_latest.csv", index=False)

        # Initialize loaders from static dataframes
        order_extractor = OrderExtractor.from_dataframe(orders_df)
        cad_loader = CADMetadataLoader.from_dataframe(geom_df)
        price_feed = PriceFeedClient(api_url=None, api_key=None, snapshot_path="data/processed/material_price_snapshots.csv")
        equipment_loader = EquipmentLogLoader(log_dir="data/raw")
    else:
        logger.info("Running in production mode. Fetching from database and APIs.")
        # DB connection string from config or env
        conn_str = os.environ.get("ORDER_DB_CONN_STR") or "postgresql://postgres@localhost:5432/matterize_orders"
        order_extractor = OrderExtractor(conn_str)
        cad_loader = CADMetadataLoader(
            api_url=os.environ.get("CAD_STORE_URL", ""),
            api_key=os.environ.get("CAD_STORE_API_KEY", "")
        )
        price_feed = PriceFeedClient(
            api_url=os.environ.get("PRICE_FEED_URL", ""),
            api_key=os.environ.get("PRICE_FEED_API_KEY", "")
        )
        equipment_loader = EquipmentLogLoader(log_dir="data/raw")

    # 1. Ingestion
    logger.info("Stage 1: Ingesting completed orders...")
    orders = order_extractor.extract(start_date=start_date, end_date=end_date)
    if not orders:
        logger.error("No orders extracted. Aborting pipeline.")
        sys.exit(1)

    # 2. Join
    logger.info("Stage 2: Joining orders, CAD features, prices, and equipment rates...")
    joiner = OrderGeometryJoiner(cad_loader, price_feed, equipment_loader)
    joined_df = joiner.join(orders)
    if joined_df.empty:
        logger.error("Join resulted in empty dataset. Aborting.")
        sys.exit(1)

    # 3. Normalize
    logger.info("Stage 3: Normalizing features, encoding categories...")
    normalizer = DatasetNormalizer()
    normalized_df = normalizer.normalize(joined_df)

    # 4. Label building
    logger.info("Stage 4: Building cost component labels...")
    label_builder = CostLabelBuilder()
    labeled_df = label_builder.build_labels(normalized_df)

    # 5. Validation
    logger.info("Stage 5: Performing schema validation...")
    validator = SchemaValidator()
    clean_df, val_metrics = validator.validate(labeled_df)
    
    # Coverage check
    logger.info("Stage 5b: Checking process coverage...")
    min_samples = pipeline_cfg.get("min_samples_per_class", 20)
    coverage_checker = CoverageChecker(min_samples_per_process=min_samples)
    is_covered, cov_stats, cov_warnings = coverage_checker.check_coverage(clean_df)

    # Drift check
    logger.info("Stage 5c: Checking for data drift...")
    drift_detector = DriftDetector()
    drift_detected, drift_stats, drift_warnings = drift_detector.detect_drift(clean_df)

    # Combine pipeline run metrics
    pipeline_metrics = {
        "validation": val_metrics,
        "coverage": cov_stats,
        "drift": drift_stats,
        "warnings": cov_warnings + drift_warnings,
        "drift_detected": drift_detected,
        "coverage_passed": is_covered,
    }

    # 6. Load/Write (unless validate_only)
    if args.validate_only:
        logger.info("Validate-only run complete. Skipping dataset output writes.")
        print("Validation report summary:")
        print(f"  Pre-val rows: {val_metrics.get('total_rows_pre_validation')}")
        print(f"  Post-val rows: {val_metrics.get('total_rows_post_validation')}")
        print(f"  Dropped rows: {val_metrics.get('dropped_rows_count')}")
        print(f"  Coverage passed: {is_covered}")
        print(f"  Drift detected: {drift_detected}")
        print("Pipeline execution finished.")
        return

    logger.info("Stage 6: Persisting versioned dataset...")
    writer = DatasetWriter()
    csv_path, json_path = writer.write(clean_df, pipeline_metrics)
    
    # Update catalog
    catalog_updater = CatalogUpdater()
    version = pipeline_metrics["validation"].get("version", 1) # Note: versioner returns path with version
    # Retrieve the version number from path or versioner next version minus 1
    version = writer.versioner.get_next_version() - 1
    
    catalog_updater.update_catalog(
        version=version,
        csv_path=csv_path,
        row_count=len(clean_df),
        columns_count=len(clean_df.columns),
        pass_rate_pct=val_metrics.get("pass_rate_pct", 100.0)
    )

    logger.info("Pipeline executed successfully and datasets updated!")


if __name__ == "__main__":
    main()
