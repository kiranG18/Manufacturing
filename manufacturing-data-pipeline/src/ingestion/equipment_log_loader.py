# src/ingestion/equipment_log_loader.py
#
# Loads machine availability and hourly rate data from weekly CSV dumps.
# Equipment logs are used to attach actual machine rate to each order
# rather than using a static rate table.

import logging
import os
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EquipmentLogLoader:
    """
    Reads equipment availability and cost-rate logs from CSV dump files.

    The operations team exports a weekly CSV from the MES (Manufacturing
    Execution System) with columns:
        facility_id, machine_id, process_code, week_start,
        availability_pct, effective_hourly_rate_usd,
        downtime_hours, maintenance_notes
    """

    def __init__(self, log_dir: str = "data/raw"):
        self._log_dir = log_dir

    def load(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        facility_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load equipment log entries. Returns combined DataFrame from all
        matching CSV files in log_dir.
        """
        files = self._find_log_files(start_date, end_date)
        if not files:
            logger.warning(f"No equipment log files found in {self._log_dir}")
            return pd.DataFrame()

        dfs = []
        for fpath in files:
            try:
                df = pd.read_csv(fpath, parse_dates=["week_start"])
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read equipment log {fpath}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        if facility_ids:
            combined = combined[combined["facility_id"].isin(facility_ids)]

        logger.info(f"Loaded {len(combined)} equipment log rows")
        return combined

    def get_rate_for_order(
        self,
        process_code: str,
        order_date: date,
        facility_id: str,
        equipment_df: pd.DataFrame,
    ) -> Optional[float]:
        """
        Look up the effective hourly machine rate for a specific process
        on a given date at a given facility. Returns None if no matching entry.
        """
        if equipment_df.empty:
            return None

        mask = (
            (equipment_df["process_code"] == process_code) &
            (equipment_df["facility_id"] == facility_id)
        )
        candidates = equipment_df[mask].copy()
        if candidates.empty:
            return None

        # Find the week containing the order date
        candidates["week_start"] = pd.to_datetime(candidates["week_start"]).dt.date
        candidates = candidates[candidates["week_start"] <= order_date]
        if candidates.empty:
            return None

        latest = candidates.sort_values("week_start").iloc[-1]
        return float(latest["effective_hourly_rate_usd"])

    def _find_log_files(self, start_date, end_date) -> List[str]:
        if not os.path.isdir(self._log_dir):
            return []
        files = []
        for fname in os.listdir(self._log_dir):
            if fname.startswith("equipment_log_") and fname.endswith(".csv"):
                files.append(os.path.join(self._log_dir, fname))
        return sorted(files)
