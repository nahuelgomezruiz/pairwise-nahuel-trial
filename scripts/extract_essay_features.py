#!/usr/bin/env python3
"""
Main script to extract essay grading features from CSV data.

This script processes essay data and extracts comprehensive feature vectors
using both rule-based and model-based approaches.

Usage:
    python extract_essay_features.py --input data.csv --output features.csv [options]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_essay_data(input_file: str) -> pd.DataFrame:
    """Load essay data from CSV file."""
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} essays from {input_file}")
        
        # Validate required columns
        required_columns = ['essay_id', 'full_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try alternative column names
            if 'full_text' not in df.columns and 'text' in df.columns:
                df['full_text'] = df['text']
            elif 'full_text' not in df.columns and 'essay_text' in df.columns:
                df['full_text'] = df['essay_text']
            
            if 'essay_id' not in df.columns:
                df['essay_id'] = df.index
        
        # Ensure we have the required columns now
        if 'full_text' not in df.columns:
            raise ValueError("No essay text column found. Expected 'full_text', 'text', or 'essay_text'")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {input_file}: {e}")
        raise


def prepare_essay_data(df: pd.DataFrame, prompt_file: str = None, 
                      sources_file: str = None) -> List[Dict[str, Any]]:
    """Prepare essay data for feature extraction."""
    logger = logging.getLogger(__name__)
    
    essays = []
    
    # Load prompt if provided
    prompt_text = None
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        logger.info(f"Loaded prompt from {prompt_file}")
    
    # Load sources if provided
    source_texts = None
    if sources_file and os.path.exists(sources_file):
        try:
            sources_df = pd.read_csv(sources_file)
            if 'text' in sources_df.columns:
                source_texts = sources_df['text'].tolist()
            elif 'source_text' in sources_df.columns:
                source_texts = sources_df['source_text'].tolist()
            logger.info(f"Loaded {len(source_texts)} source texts from {sources_file}")
        except Exception as e:
            logger.warning(f"Could not load sources from {sources_file}: {e}")
    
    # Prepare essay data
    for _, row in df.iterrows():
        essay_data = {
            'essay_id': row['essay_id'],
            'essay_text': row['full_text'],
            'prompt_text': prompt_text,
            'source_texts': source_texts
        }
        
        # Include score if available (for training data)
        if 'score' in row:
            essay_data['score'] = row['score']
        
        essays.append(essay_data)
    
    return essays


def extract_features_batch(extractor: FeatureExtractor, essays: List[Dict[str, Any]], 
                          batch_size: int = 10) -> pd.DataFrame:
    """Extract features in batches to manage memory and API limits."""
    logger = logging.getLogger(__name__)
    
    all_features = []
    
    for i in range(0, len(essays), batch_size):
        batch = essays[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(essays) + batch_size - 1)//batch_size}")
        
        try:
            batch_features = extractor.extract_features_batch(batch, show_progress=True)
            all_features.append(batch_features)
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Create empty features for failed batch
            empty_features = pd.DataFrame([
                {**{f'feature_{j}': np.nan for j in range(100)}, 
                 'essay_id': essay['essay_id']} 
                for essay in batch
            ])
            all_features.append(empty_features)
    
    # Combine all batches
    if all_features:
        result_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"Extracted features for {len(result_df)} essays")
        return result_df
    else:
        logger.warning("No features extracted")
        return pd.DataFrame()


def save_features(features_df: pd.DataFrame, output_file: str, 
                 original_df: pd.DataFrame = None) -> None:
    """Save extracted features to CSV file."""
    logger = logging.getLogger(__name__)
    
    try:
        # Merge with original data if provided (to include scores)
        if original_df is not None and 'score' in original_df.columns:
            features_df = features_df.merge(
                original_df[['essay_id', 'score']], 
                on='essay_id', 
                how='left'
            )
        
        # Save to CSV
        features_df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")
        
        # Save metadata
        metadata_file = output_file.replace('.csv', '.metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Feature extraction completed: {datetime.now()}\n")
            f.write(f"Total essays: {len(features_df)}\n")
            f.write(f"Total features: {len(features_df.columns) - 1}\n")  # -1 for essay_id
            f.write(f"Feature columns: {', '.join([col for col in features_df.columns if col != 'essay_id'])}\n")
        
        logger.info(f"Saved metadata to {metadata_file}")
        
    except Exception as e:
        logger.error(f"Error saving features to {output_file}: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract essay grading features from CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with rule-based features only
    python extract_essay_features.py --input train.csv --output features.csv --no-model-features
    
    # Full feature extraction with GPT-5-nano
    python extract_essay_features.py --input train.csv --output features.csv --api-key YOUR_API_KEY
    
    # With prompt and source texts
    python extract_essay_features.py --input train.csv --output features.csv \\
        --prompt prompt.txt --sources sources.csv --api-key YOUR_API_KEY
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with essay data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file for features')
    
    # Optional arguments
    parser.add_argument('--api-key', 
                       help='OpenAI API key for model-based features')
    parser.add_argument('--model', default='gpt-5-nano',
                       help='Model name to use (default: gpt-5-nano)')
    parser.add_argument('--prompt',
                       help='Text file containing the assignment prompt')
    parser.add_argument('--sources',
                       help='CSV file containing source texts')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing (default: 10)')
    parser.add_argument('--resources-dir', default='resources',
                       help='Directory for linguistic resources (default: resources)')
    parser.add_argument('--parallelism', type=int, default=8,
                       help='Number of essays to process in parallel (default: 8)')
    parser.add_argument('--max-model-concurrency', type=int, default=256,
                       help='Max concurrent model calls (default: 256)')
    parser.add_argument('--rpm-limit', type=int, default=30000,
                       help='Requests per minute soft limit for model calls (Tier 5 default: 30000)')
    parser.add_argument('--tpm-limit', type=int, default=180_000_000,
                       help='Tokens per minute soft limit (Tier 5 default: 180,000,000)')
    # Resume options
    parser.add_argument('--resume', action='store_true',
                       help='Resume and append to an existing output CSV, skipping essay_ids already processed')
    parser.add_argument('--resume-from-output', default=None,
                       help='Path to an existing output CSV to resume from (defaults to --output)')
    
    # Feature selection
    parser.add_argument('--no-rule-features', action='store_true',
                       help='Skip rule-based features')
    parser.add_argument('--no-model-features', action='store_true',
                       help='Skip model-based features')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Skip per-100-words normalization')
    parser.add_argument('--no-raw-counts', action='store_true',
                       help='Skip raw count features')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting essay feature extraction")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    
    try:
        # Load data
        df = load_essay_data(args.input)
        
        # Prepare essay data
        essays = prepare_essay_data(df, args.prompt, args.sources)
        
        # Configure feature extraction
        config = FeatureExtractionConfig(
            openai_api_key=args.api_key,
            model_name=args.model,
            include_rule_based=not args.no_rule_features,
            include_model_based=not args.no_model_features,
            resources_dir=args.resources_dir,
            normalize_per_100_words=not args.no_normalize,
            include_raw_counts=not args.no_raw_counts,
            parallelism=args.parallelism,
            max_model_concurrency=args.max_model_concurrency,
            requests_per_minute_limit=args.rpm_limit,
            tokens_per_minute_limit=args.tpm_limit
        )
        
        # Initialize feature extractor
        extractor = FeatureExtractor(config)
        
        # Resume support: determine already processed essay_ids
        processed_ids = set()
        resume_path = args.resume_from_output or args.output
        first_write = True
        if args.resume and os.path.exists(resume_path):
            try:
                tmp = pd.read_csv(resume_path, usecols=['essay_id'])
                processed_ids = set(tmp['essay_id'].astype(str))
                first_write = False
                logger.info(f"Resuming: {len(processed_ids)} essays already in {resume_path}")
            except Exception as e:
                logger.warning(f"Could not read existing output for resume: {e}")
        
        # Filter essays to those not processed
        if processed_ids:
            essays = [e for e in essays if str(e.get('essay_id')) not in processed_ids]
            logger.info(f"Remaining essays to process: {len(essays)}")
        
        total = len(essays)
        if total == 0 and args.resume:
            logger.info("Nothing to do; all essays already processed. Exiting.")
            sys.exit(0)
        
        # Stream batches and append to CSV for robust resume
        for i in range(0, total, args.batch_size):
            batch = essays[i:i + args.batch_size]
            logger.info(f"Processing batch {i//args.batch_size + 1}/{(total + args.batch_size - 1)//args.batch_size}")
            batch_features = extractor.extract_features_batch(batch, show_progress=True)
            if len(batch_features) == 0:
                continue
            # Merge score if available
            if 'score' in df.columns:
                batch_features = batch_features.merge(
                    df[['essay_id', 'score']].astype({'essay_id': batch_features['essay_id'].dtype}),
                    on='essay_id', how='left'
                )
            # Append to CSV
            write_header = first_write
            batch_features.to_csv(args.output, mode='a', index=False, header=write_header)
            first_write = False
        
        if not os.path.exists(args.output):
            logger.error("No features were extracted")
            sys.exit(1)
        
        # Write metadata summary
        try:
            meta_file = args.output.replace('.csv', '.metadata.txt')
            out_df = pd.read_csv(args.output)
            with open(meta_file, 'w') as f:
                f.write(f"Feature extraction completed: {datetime.now()}\n")
                f.write(f"Total essays: {len(out_df)}\n")
                f.write(f"Total features: {len(out_df.columns) - 1}\n")
                f.write(f"Feature columns: {', '.join([c for c in out_df.columns if c != 'essay_id'])}\n")
            logger.info(f"Saved metadata to {meta_file}")
        except Exception as e:
            logger.warning(f"Could not write metadata: {e}")
        
        logger.info("Feature extraction completed successfully")
        
        # Print summary
        out_df = pd.read_csv(args.output)
        print(f"\nFeature Extraction Summary:")
        print(f"  Essays processed: {len(out_df)}")
        print(f"  Features extracted: {len(out_df.columns) - 1}")
        print(f"  Output file: {args.output}")
        
        if 'score' in out_df.columns:
            print(f"  Score range: {out_df['score'].min():.1f} - {out_df['score'].max():.1f}")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()