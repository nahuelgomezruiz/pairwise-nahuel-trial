#!/usr/bin/env python3
"""
Cross-dataset evaluation:
1. Train RF on 30 samples from existing Kaggle dataset features
2. Test on 100 essays from training_set_rel3.tsv
3. Push results to Google Sheets with QWK
"""

import os
import sys
import json
import base64
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Add project src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT))

from src.sheets_integration.sheets_client import SheetsClient
from src.feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate QWK score."""
    y_true = np.asarray(y_true).astype(int)
    min_rating = int(np.min(y_true))
    max_rating = int(np.max(y_true))
    y_pred_round = np.rint(np.clip(y_pred, min_rating, max_rating)).astype(int)

    num_ratings = max_rating - min_rating + 1
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=float)
    for a, b in zip(y_true, y_pred_round):
        conf_mat[a - min_rating, b - min_rating] += 1

    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    for i in range(num_ratings):
        hist_true[i] = np.sum(y_true == (i + min_rating))
        hist_pred[i] = np.sum(y_pred_round == (i + min_rating))

    E = np.outer(hist_true, hist_pred) / np.sum(hist_true)
    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2) if num_ratings > 1 else 0.0

    O = conf_mat / np.sum(conf_mat)
    E = E / np.sum(E)

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)
    if denominator == 0:
        return 0.0
    return float(1.0 - numerator / denominator)


def convert_tsv_to_csv(tsv_path: str, output_csv: str, limit: int = 100):
    """Convert first N rows of TSV to CSV format expected by feature extractor."""
    logger.info(f"Converting first {limit} essays from {tsv_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(tsv_path, sep='\t', nrows=limit, quoting=1, encoding=encoding)
            logger.info(f"Successfully read TSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read TSV file with any of the encodings: {encodings}")
    
    # Create output dataframe with expected columns
    output_df = pd.DataFrame({
        'essay_id': df['essay_id'].astype(str),
        'full_text': df['essay'],
        'score': df['rater1_domain1'].astype(float)
    })
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(output_df)} essays to {output_csv}")
    return output_df


def extract_features_for_tsv_essays(csv_path: str, output_features: str, api_key: str = None):
    """Extract features for essays from TSV (converted to CSV)."""
    logger.info(f"Extracting features for {csv_path}")
    
    # Load essay data
    df = pd.read_csv(csv_path)
    
    # Prepare essays for extraction
    essays = []
    for _, row in df.iterrows():
        essays.append({
            'essay_id': row['essay_id'],
            'essay_text': row['full_text'],
            'score': row['score']
        })
    
    # Configure extractor (rule-based only for speed)
    config = FeatureExtractionConfig(
        openai_api_key=api_key,
        include_rule_based=True,
        include_model_based=bool(api_key),  # Only if API key provided
        resources_dir='resources',
        parallelism=8
    )
    
    # Extract features
    extractor = FeatureExtractor(config)
    features_df = extractor.extract_features_batch(essays, show_progress=True)
    
    # Add scores
    features_df = features_df.merge(
        df[['essay_id', 'score']],
        on='essay_id',
        how='left'
    )
    
    # Save features
    features_df.to_csv(output_features, index=False)
    logger.info(f"Saved features to {output_features}")
    return features_df


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--train-features", default="exports/features_train_80_full.csv", 
                       help="Training features from Kaggle dataset")
    parser.add_argument("--train-size", type=int, default=30, help="Training sample size")
    parser.add_argument("--tsv-path", default="training_set_rel3.tsv", 
                       help="TSV file with test essays")
    parser.add_argument("--test-size", type=int, default=100, help="Test sample size from TSV")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (optional, for model features)")
    parser.add_argument("--worksheet-name", default=None, 
                       help="Worksheet name (default: cross-dataset-<timestamp>)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-extraction", action="store_true", 
                       help="Skip feature extraction if test features already exist")
    args = parser.parse_args()
    
    # Step 1: Convert TSV to CSV
    test_csv = "exports/test_essays_tsv.csv"
    test_df = convert_tsv_to_csv(args.tsv_path, test_csv, limit=args.test_size)
    
    # Step 2: Extract features for test essays
    test_features_path = "exports/test_features_tsv.csv"
    if args.skip_extraction and os.path.exists(test_features_path):
        logger.info(f"Using existing features from {test_features_path}")
        test_features_df = pd.read_csv(test_features_path)
    else:
        test_features_df = extract_features_for_tsv_essays(
            test_csv, test_features_path, api_key=args.api_key
        )
    
    # Step 3: Load and sample training features
    logger.info(f"Loading training features from {args.train_features}")
    train_df = pd.read_csv(args.train_features)
    
    # Sample training data
    if len(train_df) < args.train_size:
        raise ValueError(f"Need {args.train_size} training samples but only have {len(train_df)}")
    
    train_sample = train_df.sample(n=args.train_size, random_state=args.random_state)
    
    # Get feature columns (exclude essay_id and score)
    feature_cols = [c for c in train_sample.columns if c not in ('essay_id', 'score')]
    
    # Align test features to training features
    missing_cols = [c for c in feature_cols if c not in test_features_df.columns]
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} features in test data, filling with 0")
        for col in missing_cols:
            test_features_df[col] = 0
    
    test_features_df = test_features_df[['essay_id'] + feature_cols + ['score']]
    
    # Prepare data for training
    X_train = train_sample[feature_cols].fillna(0).values
    y_train = train_sample['score'].values
    X_test = test_features_df[feature_cols].fillna(0).values
    y_test = test_features_df['score'].values
    essay_ids_test = test_features_df['essay_id'].astype(str).tolist()
    
    # Step 4: Train RandomForest
    logger.info(f"Training RandomForest on {len(X_train)} samples")
    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [12, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1],
        'max_features': ['sqrt', 0.5]
    }
    
    search = GridSearchCV(
        rf, param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    
    # Step 5: Make predictions and calculate QWK
    y_pred = model.predict(X_test)
    qwk = quadratic_weighted_kappa(y_test, y_pred)
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Training: {args.train_size} samples from Kaggle dataset")
    print(f"Testing: {args.test_size} samples from TSV dataset")
    print(f"QWK Score: {qwk:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("="*60)
    
    # Step 6: Push to Google Sheets
    scores = []
    for i in range(len(essay_ids_test)):
        scores.append({
            'essay_id': essay_ids_test[i],
            'total_score': float(y_pred[i]),
            'reasoning': f'Cross-dataset RF ({args.train_size} Kaggle → {args.test_size} TSV)',
            'model_used': 'RF-CrossDataset',
            'metadata': {
                'train_dataset': 'Kaggle',
                'test_dataset': 'TSV',
                'train_size': args.train_size,
                'test_size': args.test_size
            }
        })
    
    # Initialize Sheets client
    sheets_id = os.getenv("GOOGLE_SHEETS_ID")
    if not sheets_id:
        logger.warning("GOOGLE_SHEETS_ID not set, skipping Sheets upload")
        return
    
    credentials_dict = None
    if os.getenv("SHEETS_CREDENTIALS_BASE64"):
        try:
            credentials_dict = json.loads(base64.b64decode(os.getenv("SHEETS_CREDENTIALS_BASE64")))
        except Exception as e:
            logger.warning(f"Could not decode credentials: {e}")
    
    if credentials_dict:
        sheets_client = SheetsClient(credentials_dict=credentials_dict)
    else:
        sheets_client = SheetsClient()
    
    # Generate worksheet name
    worksheet_name = args.worksheet_name or f"cross-dataset-{datetime.now().strftime('%H%M%S')}"
    run_id = datetime.now().strftime("%H:%M:%S")
    
    # Write to Sheets
    ok = sheets_client.write_scores_to_sheet(
        scores=scores,
        spreadsheet_id=sheets_id,
        worksheet_name=worksheet_name,
        run_id=run_id,
        actual_scores=list(y_test.astype(float)),
        component_categories=None
    )
    
    if ok:
        print(f"\n✅ Pushed {len(scores)} results to Google Sheets worksheet '{worksheet_name}'")
        print(f"   QWK: {qwk:.4f}")
    else:
        print("\n⚠️ Failed to write to Google Sheets")


if __name__ == "__main__":
    main()