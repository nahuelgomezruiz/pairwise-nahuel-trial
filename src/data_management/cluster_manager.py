"""Cluster management for essay scoring system."""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages essay clusters and their associated data."""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        """Initialize with optional custom data loader."""
        self.data_loader = data_loader or DataLoader()
        self._cluster_cache = {}
        
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get the cluster summary with caching."""
        if 'summary' not in self._cluster_cache:
            self._cluster_cache['summary'] = self.data_loader.load_cluster_summary()
        return self._cluster_cache['summary']
        
    def get_cluster_data(self, cluster_name: str, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get cluster sample and train data with optional caching."""
        cache_key = f"cluster_{cluster_name}"
        
        if use_cache and cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]
            
        sample_df, train_df = self.data_loader.load_cluster_data(cluster_name)
        
        # Validate required columns
        required_columns = ['essay_id', 'full_text', 'score']
        if not self.data_loader.validate_cluster_data(sample_df, required_columns):
            raise ValueError(f"Sample data for cluster {cluster_name} missing required columns")
        if not self.data_loader.validate_cluster_data(train_df, required_columns):
            raise ValueError(f"Train data for cluster {cluster_name} missing required columns")
            
        if use_cache:
            self._cluster_cache[cache_key] = (sample_df, train_df)
            
        return sample_df, train_df
        
    def get_available_clusters(self) -> List[str]:
        """Get list of available cluster names."""
        return self.data_loader.get_available_clusters()
        
    def prepare_sample_essays(self, sample_df: pd.DataFrame) -> List[Dict]:
        """Prepare sample essays for comparison."""
        sample_essays = []
        for _, row in sample_df.iterrows():
            sample_essays.append({
                'essay_id': row['essay_id'],
                'text': row['full_text'],
                'score': row['score']
            })
        return sample_essays
        
    def filter_test_essays(self, train_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        """Filter and limit test essays for grading."""
        if limit > 0:
            return train_df.head(limit)
        return train_df
        
    def clear_cache(self):
        """Clear the cluster cache."""
        self._cluster_cache.clear()