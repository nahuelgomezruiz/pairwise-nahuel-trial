"""
Main FeatureExtractor class for computing essay grading features.
Implements all features from the essay grading feature list.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

from .rule_based_features import RuleBasedFeatureExtractor
from .model_based_features import ModelBasedFeatureExtractor
from .resource_manager import ResourceManager


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction"""
    # Model configuration
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-5-nano"
    
    # Feature selection
    include_rule_based: bool = True
    include_model_based: bool = True
    
    # Resource paths
    resources_dir: str = "resources"
    
    # Processing options
    normalize_per_100_words: bool = True
    include_raw_counts: bool = True
    
    # Parallelism and rate control
    parallelism: int = 8  # number of parallel essays to process
    max_model_concurrency: int = 256  # cap concurrent model HTTP calls
    requests_per_minute_limit: Optional[int] = 30000  # Tier 5 RPM
    tokens_per_minute_limit: Optional[int] = 180_000_000  # Tier 5 TPM


class FeatureExtractor:
    """
    Main class for extracting essay grading features.
    
    Computes both rule-based and model-based features according to the
    essay grading feature list specification.
    """
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(config.resources_dir)
        
        # Initialize feature extractors
        if config.include_rule_based:
            self.rule_based_extractor = RuleBasedFeatureExtractor(
                self.resource_manager, 
                normalize_per_100_words=config.normalize_per_100_words,
                include_raw_counts=config.include_raw_counts
            )
        
        if config.include_model_based:
            self.model_based_extractor = ModelBasedFeatureExtractor(
                api_key=config.openai_api_key,
                model_name=config.model_name,
                max_concurrency=config.max_model_concurrency,
                rpm_limit=config.requests_per_minute_limit,
                tpm_limit=config.tokens_per_minute_limit
            )
    
    def extract_features(
        self, 
        essay_text: str, 
        prompt_text: Optional[str] = None,
        source_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract all features from an essay.
        
        Args:
            essay_text: The student's essay text
            prompt_text: The assignment prompt (optional)
            source_texts: Source passages provided to students (optional)
            
        Returns:
            Dictionary mapping feature names to values
        """
        features = {}
        
        try:
            # Extract rule-based features
            if self.config.include_rule_based:
                rule_features = self.rule_based_extractor.extract_all_features(
                    essay_text, prompt_text, source_texts
                )
                features.update(rule_features)
            
            # Extract model-based features
            if self.config.include_model_based:
                model_features = self.model_based_extractor.extract_all_features(
                    essay_text, prompt_text, source_texts
                )
                features.update(model_features)
                
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
        
        return features
    
    def extract_features_batch(
        self,
        essays: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from a batch of essays.
        
        Args:
            essays: List of dictionaries with keys 'essay_text', 
                   optionally 'prompt_text' and 'source_texts'
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with features for each essay
        """
        from tqdm import tqdm
        
        results: List[Dict[str, Any]] = []
        total = len(essays)
        if total == 0:
            return pd.DataFrame()

        # Chunk essays to avoid creating too many futures at once
        chunk_size = max(1, self.config.parallelism)
        num_chunks = ceil(total / chunk_size)

        pbar = tqdm(total=total, desc="Extracting features") if show_progress else None

        def _task(idx: int, essay_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                features = self.extract_features(
                    essay_text=essay_data['essay_text'],
                    prompt_text=essay_data.get('prompt_text'),
                    source_texts=essay_data.get('source_texts')
                )
            except Exception as exc:
                self.logger.error(f"Error processing essay {idx}: {exc}")
                features = {f'feature_{j}': np.nan for j in range(100)}
            features['essay_id'] = essay_data.get('essay_id', idx)
            return features

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, total)
            batch = essays[start:end]
            with ThreadPoolExecutor(max_workers=self.config.parallelism) as executor:
                future_to_idx = {
                    executor.submit(_task, start + i, essay): start + i
                    for i, essay in enumerate(batch)
                }
                for future in as_completed(future_to_idx):
                    res = future.result()
                    results.append(res)
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Keep deterministic order by essay_id if present, otherwise original idx
        df = pd.DataFrame(results)
        if 'essay_id' in df.columns:
            return df.sort_values('essay_id').reset_index(drop=True)
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted."""
        feature_names = []
        
        if self.config.include_rule_based:
            feature_names.extend(self.rule_based_extractor.get_feature_names())
        
        if self.config.include_model_based:
            feature_names.extend(self.model_based_extractor.get_feature_names())
        
        return feature_names
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features."""
        descriptions = {}
        
        if self.config.include_rule_based:
            descriptions.update(self.rule_based_extractor.get_feature_descriptions())
        
        if self.config.include_model_based:
            descriptions.update(self.model_based_extractor.get_feature_descriptions())
        
        return descriptions