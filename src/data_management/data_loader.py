"""Data loading utilities for essay scoring system."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of essay data, clusters, and samples."""
    
    def __init__(self, data_root: Optional[Path] = None):
        """Initialize with optional custom data root path."""
        self.data_root = data_root or (PROJECT_ROOT / "src" / "data")
        self.cluster_samples_dir = self.data_root / "cluster_samples"
        self.train_clusters_dir = self.data_root / "train_clusters"
        
    def load_cluster_summary(self) -> pd.DataFrame:
        """Load the cluster summary data."""
        summary_path = self.cluster_samples_dir / "sampling_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Cluster summary not found at {summary_path}")
            
        try:
            df = pd.read_csv(summary_path)
            logger.info(f"Loaded cluster summary with {len(df)} clusters")
            return df
        except Exception as e:
            logger.error(f"Error loading cluster summary: {e}")
            raise
            
    def load_cluster_data(self, cluster_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both sample and train data for a specific cluster.
        
        Returns:
            Tuple of (sample_df, train_df)
        """
        # Load sample data
        sample_path = self.cluster_samples_dir / f"{cluster_name}_sample.csv"
        if not sample_path.exists():
            # Try optimized version
            sample_path = self.cluster_samples_dir / f"{cluster_name}_optimized.csv"
            
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample data not found for cluster {cluster_name}")
            
        # Load train data
        train_path = self.train_clusters_dir / f"{cluster_name}.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Train data not found for cluster {cluster_name}")
            
        try:
            sample_df = pd.read_csv(sample_path)
            train_df = pd.read_csv(train_path)
            
            logger.info(f"Loaded cluster {cluster_name}: {len(sample_df)} samples, {len(train_df)} train essays")
            return sample_df, train_df
            
        except Exception as e:
            logger.error(f"Error loading cluster data for {cluster_name}: {e}")
            raise
            
    def get_available_clusters(self) -> List[str]:
        """Get list of available cluster names."""
        clusters = set()
        
        # Check sample files
        for file_path in self.cluster_samples_dir.glob("*_sample.csv"):
            cluster_name = file_path.stem.replace("_sample", "")
            clusters.add(cluster_name)
            
        # Check optimized files
        for file_path in self.cluster_samples_dir.glob("*_optimized.csv"):
            cluster_name = file_path.stem.replace("_optimized", "")
            clusters.add(cluster_name)
            
        # Filter to only include clusters that have train data
        available_clusters = []
        for cluster in clusters:
            train_path = self.train_clusters_dir / f"{cluster}.csv"
            if train_path.exists():
                available_clusters.append(cluster)
                
        logger.info(f"Found {len(available_clusters)} available clusters")
        return sorted(available_clusters)
        
    def validate_cluster_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that cluster data has required columns."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True