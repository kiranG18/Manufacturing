# src/transforms/joiner.py
import logging
import pandas as pd
from typing import List, Dict, Any
from ..ingestion.order_extractor import OrderRecord
from ..ingestion.cad_metadata_loader import CADMetadataLoader
from ..ingestion.price_feed import PriceFeedClient
from ..ingestion.equipment_log_loader import EquipmentLogLoader

logger = logging.getLogger(__name__)

class OrderGeometryJoiner:
    """
    Joins order database records, CAD geometry features, material pricing, and machine logs.
    Performs a left join on orders, logging and filtering out rows where geometry is missing.
    """

    def __init__(
        self,
        cad_loader: CADMetadataLoader,
        price_feed: PriceFeedClient,
        equipment_loader: EquipmentLogLoader,
    ):
        self.cad_loader = cad_loader
        self.price_feed = price_feed
        self.equipment_loader = equipment_loader

    def join(self, orders: List[OrderRecord]) -> pd.DataFrame:
        """
        Merge the list of OrderRecords with geometry features, material prices, and equipment logs.
        """
        if not orders:
            logger.warning("No orders provided to joiner")
            return pd.DataFrame()

        # 1. Convert orders list to DataFrame
        orders_df = pd.DataFrame([vars(o) for o in orders])
        logger.info(f"Starting join with {len(orders_df)} order records")

        # 2. Bulk load CAD geometry features
        part_ids = list(orders_df["part_id"].unique())
        geom_df = self.cad_loader.load_batch(part_ids)

        if geom_df.empty:
            logger.error("No geometry features found for any part_ids. Joining is impossible.")
            return pd.DataFrame()

        # 3. Perform Left Join on orders with geometry features
        # Filter out orders that have no matching geometry
        joined_df = pd.merge(orders_df, geom_df, on="part_id", how="left")
        
        # Log missing geometry parts
        missing_geom = joined_df[joined_df["bounding_box_x"].isna()]
        if not missing_geom.empty:
            missing_ids = missing_geom["part_id"].unique()
            logger.warning(
                f"Filtered out {len(missing_geom)} order rows because their part_ids "
                f"had no geometry feature vectors: {list(missing_ids)[:10]}"
            )
            joined_df = joined_df[joined_df["bounding_box_x"].notna()].copy()

        # 4. Fetch equipment hourly rates contemporaneous logs
        # First load all logs
        min_date = orders_df["order_date"].min()
        max_date = orders_df["order_date"].max()
        equip_df = self.equipment_loader.load(start_date=min_date, end_date=max_date)

        # 5. Populate pricing and hourly rate per order row
        material_prices = []
        hourly_rates = []

        for idx, row in joined_df.iterrows():
            # contemporaneous price feed lookup
            mat_price = self.price_feed.get_price_for_date(row["material_code"], row["order_date"])
            material_prices.append(mat_price)

            # contemporaneous machine rate lookup
            rate = self.equipment_loader.get_rate_for_order(
                row["process_code"], row["order_date"], row["facility_id"], equip_df
            )
            hourly_rates.append(rate)

        joined_df["material_price_per_kg"] = material_prices
        joined_df["effective_hourly_rate"] = hourly_rates

        # Log completion
        logger.info(f"Join completed successfully. Output shape: {joined_df.shape}")
        return joined_df
