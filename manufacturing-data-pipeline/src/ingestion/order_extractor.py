# src/ingestion/order_extractor.py
#
# Pulls completed manufacturing orders from the internal order database.
# Orders are the primary source of truth for cost labels — every row
# in the training data corresponds to a fulfilled order with an invoice.

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    order_id: str
    part_id: str
    process_code: str
    material_code: str
    quantity: int
    batch_size: int
    surface_finish_ra: float
    tolerance_band: str
    order_date: date
    completion_date: date
    facility_id: str
    cost_total: float
    cost_machine_time: Optional[float] = None
    cost_setup: Optional[float] = None
    cost_tooling: Optional[float] = None
    cost_raw_material: Optional[float] = None
    cost_finishing: Optional[float] = None
    cost_label_source: str = "estimated"


class OrderExtractor:
    """
    Extracts completed orders from the manufacturing order database.

    In production, this connects to a Postgres instance via SQLAlchemy.
    For testing, it accepts a pre-loaded DataFrame (via from_dataframe()).
    Both modes produce the same list of OrderRecord objects downstream.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self._conn_str = connection_string
        self._engine = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "OrderExtractor":
        """Testing constructor — wraps a DataFrame as if it were the database."""
        inst = cls(connection_string=None)
        inst._static_df = df
        return inst

    def extract(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        facility_ids: Optional[List[str]] = None,
    ) -> List[OrderRecord]:
        """
        Pull completed orders. Both start_date and end_date are inclusive.
        Returns orders sorted by completion_date ascending.
        """
        if hasattr(self, "_static_df"):
            return self._extract_from_df(
                self._static_df, start_date, end_date, facility_ids
            )
        return self._extract_from_db(start_date, end_date, facility_ids)

    def _extract_from_db(self, start_date, end_date, facility_ids):
        """Live database extraction. Requires SQLAlchemy + psycopg2."""
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise RuntimeError(
                "SQLAlchemy is required for database extraction. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )

        if not self._engine:
            self._engine = create_engine(self._conn_str)

        conditions = ["status = 'completed'"]
        params = {}
        if start_date:
            conditions.append("completion_date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            conditions.append("completion_date <= :end_date")
            params["end_date"] = end_date
        if facility_ids:
            conditions.append("facility_id = ANY(:facilities)")
            params["facilities"] = facility_ids

        where = " AND ".join(conditions)
        query = f"""
            SELECT
                o.order_id, o.part_id, o.process_code, o.material_code,
                o.quantity, o.batch_size, o.surface_finish_ra, o.tolerance_band,
                o.order_date, o.completion_date, o.facility_id,
                i.cost_total, i.cost_machine_time, i.cost_setup,
                i.cost_tooling, i.cost_raw_material, i.cost_finishing,
                i.label_source AS cost_label_source
            FROM orders o
            LEFT JOIN invoices i ON o.order_id = i.order_id
            WHERE {where}
            ORDER BY o.completion_date ASC
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        logger.info(f"Extracted {len(rows)} orders from database")
        return [OrderRecord(**dict(row._mapping)) for row in rows]

    def _extract_from_df(self, df, start_date, end_date, facility_ids):
        """Extract from pre-loaded DataFrame (test/offline mode)."""
        mask = pd.Series([True] * len(df), index=df.index)
        if start_date and "completion_date" in df.columns:
            mask &= pd.to_datetime(df["completion_date"]).dt.date >= start_date
        if end_date and "completion_date" in df.columns:
            mask &= pd.to_datetime(df["completion_date"]).dt.date <= end_date
        if facility_ids and "facility_id" in df.columns:
            mask &= df["facility_id"].isin(facility_ids)

        filtered = df[mask].copy()
        logger.info(f"Loaded {len(filtered)} orders from DataFrame")
        records = []
        for _, row in filtered.iterrows():
            rec = OrderRecord(
                order_id=str(row.get("order_id", "")),
                part_id=str(row.get("part_id", "")),
                process_code=str(row.get("process_code", "")),
                material_code=str(row.get("material_code", "")),
                quantity=int(row.get("quantity", 1)),
                batch_size=int(row.get("batch_size", row.get("quantity", 1))),
                surface_finish_ra=float(row.get("surface_finish_ra", 3.2)),
                tolerance_band=str(row.get("tolerance_band", "IT8")),
                order_date=row.get("order_date", date.today()),
                completion_date=row.get("completion_date", date.today()),
                facility_id=str(row.get("facility_id", "FACILITY_01")),
                cost_total=float(row.get("cost_total", 0)),
                cost_machine_time=row.get("cost_machine_time"),
                cost_setup=row.get("cost_setup"),
                cost_tooling=row.get("cost_tooling"),
                cost_raw_material=row.get("cost_raw_material"),
                cost_finishing=row.get("cost_finishing"),
                cost_label_source=str(row.get("cost_label_source", "estimated")),
            )
            records.append(rec)
        return records
