#!/usr/bin/env python3
"""
Example usage of the essay feature extraction system.

This script demonstrates how to:
1. Extract features from essay data
2. Split data for training/testing
3. Train a tree regressor on the features
4. Evaluate the model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# Matplotlib imports moved to plot_results function to handle optional dependency
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig


def load_and_extract_features(csv_file: str, api_key: str = None, 
                            use_model_features: bool = False) -> pd.DataFrame:
    """
    Load essay data and extract features.
    
    Args:
        csv_file: Path to CSV file with essay data
        api_key: OpenAI API key (optional, for model-based features)
        use_model_features: Whether to include GPT-5-nano features
        
    Returns:
        DataFrame with extracted features
    """
    print(f"Loading data from {csv_file}...")
    
    # Load the data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} essays")
    
    # Configure feature extraction
    config = FeatureExtractionConfig(
        openai_api_key=api_key,
        model_name="gpt-5-nano",
        include_rule_based=True,
        include_model_based=use_model_features,
        normalize_per_100_words=True,
        include_raw_counts=True
    )
    
    # Initialize extractor
    extractor = FeatureExtractor(config)
    
    # Prepare essay data
    essays = []
    for _, row in df.iterrows():
        essays.append({
            'essay_id': row.get('essay_id', row.name),
            'essay_text': row['full_text'],
            'score': row.get('score')
        })
    
    print("Extracting features...")
    features_df = extractor.extract_features_batch(essays, show_progress=True)
    
    print(f"Extracted {len(features_df.columns)} features")
    return features_df


def train_and_evaluate_model(features_df: pd.DataFrame, test_size: float = 0.2):
    """
    Train a Random Forest regressor and evaluate performance.
    
    Args:
        features_df: DataFrame with features and scores
        test_size: Fraction of data to use for testing
    """
    # Prepare features and target
    feature_columns = [col for col in features_df.columns 
                      if col not in ['essay_id', 'score']]
    
    X = features_df[feature_columns]
    y = features_df['score']
    
    print(f"Training with {len(feature_columns)} features on {len(X)} samples")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest regressor...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = rf.predict(X_train_scaled)
    y_pred_test = rf.predict(X_test_scaled)
    
    # Evaluate performance
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    return rf, scaler, feature_importance, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae,
        'y_test': y_test, 'y_pred_test': y_pred_test
    }


def plot_results(results: dict, save_path: str = None):
    """Plot model results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        ax1.scatter(results['y_test'], results['y_pred_test'], alpha=0.6)
        ax1.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Score')
        ax1.set_ylabel('Predicted Score')
        ax1.set_title(f'Actual vs Predicted (R² = {results["test_r2"]:.3f})')
        
        # Residuals
        residuals = results['y_test'] - results['y_pred_test']
        ax2.scatter(results['y_pred_test'], residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Score')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Main example function."""
    print("Essay Feature Extraction and Modeling Example")
    print("=" * 50)
    
    # Configuration
    csv_file = "src/data/learning-agency-lab-automated-essay-scoring-2/train.csv"
    use_model_features = False  # Set to True if you have OpenAI API key
    api_key = None  # Set your OpenAI API key here if using model features
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        print("Please ensure the data file exists or update the path.")
        return
    
    try:
        # Extract features
        features_df = load_and_extract_features(
            csv_file, 
            api_key=api_key, 
            use_model_features=use_model_features
        )
        
        # Check if we have scores for training
        if 'score' not in features_df.columns or features_df['score'].isna().all():
            print("No scores found in data. Cannot train model.")
            print("Features extracted successfully. Check the output CSV.")
            return
        
        # Train and evaluate model
        model, scaler, importance, results = train_and_evaluate_model(features_df)
        
        # Plot results
        plot_results(results, save_path="model_results.png")
        
        # Save feature importance
        importance.to_csv("feature_importance.csv", index=False)
        print(f"\nFeature importance saved to feature_importance.csv")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()