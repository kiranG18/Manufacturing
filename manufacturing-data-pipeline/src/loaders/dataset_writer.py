# src/loaders/dataset_writer.py
import logging
import os
import shutil
import pandas as pd
from typing import Dict, Any, Tuple
from ..transforms.versioner import DatasetVersioner

logger = logging.getLogger(__name__)


class DatasetWriter:
    """
    Handles persisting the final clean training dataset to the processed data folder
    and copying it to training directories in other portfolio repositories.
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        cost_model_dir: str = "../matterize-cost-model/data/processed",
        classifier_dir: str = "../cad-process-recommender/data/processed",
    ):
        self.output_dir = output_dir
        self.cost_model_dir = cost_model_dir
        self.classifier_dir = classifier_dir
        self.versioner = DatasetVersioner(output_dir=output_dir)

    def write(
        self,
        df: pd.DataFrame,
        pipeline_metrics: Dict[str, Any],
        copy_to_repos: bool = True,
    ) -> Tuple[str, str]:
        """
        Writes the clean DataFrame to a new incremental version and updates copies.
        """
        # Save incremental version
        csv_path, json_path = self.versioner.save_version(df, pipeline_metrics)

        if not copy_to_repos:
            return csv_path, json_path

        # Resolve paths relative to pipeline root
        pipeline_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 1. Copy to matterize-cost-model
        target_cost_dir = self.cost_model_dir
        if not os.path.isabs(target_cost_dir):
            target_cost_dir = os.path.normpath(os.path.join(pipeline_root, target_cost_dir))
        
        if os.path.isdir(target_cost_dir):
            try:
                dest = os.path.join(target_cost_dir, "training_data_v2.csv")
                shutil.copy2(csv_path, dest)
                logger.info(f"Copied latest dataset to cost model path: {dest}")
            except Exception as e:
                logger.warning(f"Failed to copy to cost model dir: {e}")

        # 2. Copy to cad-process-recommender
        target_class_dir = self.classifier_dir
        if not os.path.isabs(target_class_dir):
            target_class_dir = os.path.normpath(os.path.join(pipeline_root, target_class_dir))

        if os.path.isdir(target_class_dir):
            try:
                # Recommender consumes features_v2.csv
                dest = os.path.join(target_class_dir, "features_v2.csv")
                shutil.copy2(csv_path, dest)
                logger.info(f"Copied latest dataset to classifier path: {dest}")
            except Exception as e:
                logger.warning(f"Failed to copy to classifier dir: {e}")

        return csv_path, json_path
