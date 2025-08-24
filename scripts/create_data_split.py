#!/usr/bin/env python3
"""Script to create training/validation splits from the main dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_stratified_split(input_csv_path, output_dir, train_ratio=0.8, random_state=42):
    """
    Create stratified train/validation split maintaining score distribution.
    
    Args:
        input_csv_path: Path to the main CSV file
        output_dir: Directory to save the split files
        train_ratio: Proportion for training (0.8 = 80% train, 20% validation)
        random_state: Random seed for reproducibility
    """
    
    # Load the data
    print(f"Loading data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Total essays: {len(df)}")
    
    # Print score distribution
    print("\nOriginal score distribution:")
    print(df['score'].value_counts().sort_index())
    
    # Create stratified split based on score
    train_df, val_df = train_test_split(
        df, 
        test_size=(1 - train_ratio),
        stratify=df['score'],  # Maintain score distribution
        random_state=random_state
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_path = output_path / "train_split.csv"
    val_path = output_path / "validation_split.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\nSaved splits:")
    print(f"Training set: {train_path} ({len(train_df)} essays)")
    print(f"Validation set: {val_path} ({len(val_df)} essays)")
    
    # Print split score distributions
    print(f"\nTraining set score distribution:")
    print(train_df['score'].value_counts().sort_index())
    
    print(f"\nValidation set score distribution:")
    print(val_df['score'].value_counts().sort_index())
    
    # Calculate percentages
    print(f"\nSplit percentages:")
    print(f"Training: {len(train_df)/len(df)*100:.1f}%")
    print(f"Validation: {len(val_df)/len(df)*100:.1f}%")
    
    return train_path, val_path


def create_small_development_set(input_csv_path, output_dir, n_samples=100, random_state=42):
    """
    Create a small balanced development set for quick testing.
    
    Args:
        input_csv_path: Path to the main CSV file
        output_dir: Directory to save the development set
        n_samples: Total number of samples (will be distributed across scores)
        random_state: Random seed for reproducibility
    """
    
    # Load the data
    df = pd.read_csv(input_csv_path)
    
    # Get proportional samples from each score
    dev_samples = []
    score_counts = df['score'].value_counts().sort_index()
    
    for score in score_counts.index:
        # Calculate proportional number of samples for this score
        score_proportion = score_counts[score] / len(df)
        score_samples = max(1, int(n_samples * score_proportion))  # At least 1 sample per score
        
        # Sample from this score group
        score_df = df[df['score'] == score].sample(
            n=min(score_samples, score_counts[score]), 
            random_state=random_state
        )
        dev_samples.append(score_df)
    
    # Combine all samples
    dev_df = pd.concat(dev_samples, ignore_index=True)
    
    # Shuffle the final dataset
    dev_df = dev_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save development set
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dev_path = output_path / "development_set.csv"
    
    dev_df.to_csv(dev_path, index=False)
    
    print(f"\nCreated development set: {dev_path} ({len(dev_df)} essays)")
    print("Development set score distribution:")
    print(dev_df['score'].value_counts().sort_index())
    
    return dev_path


def main():
    """Main function to create data splits."""
    
    # Configuration
    input_csv = "src/data/learning-agency-lab-automated-essay-scoring-2/train.csv"
    output_directory = "src/data/splits"
    
    print("=" * 60)
    print("Creating Data Splits")
    print("=" * 60)
    
    # Create main train/validation split (50/50)
    print("\n1. Creating main train/validation split (50/50)...")
    train_path, val_path = create_stratified_split(
        input_csv_path=input_csv,
        output_dir=output_directory,
        train_ratio=0.5,
        random_state=42
    )
    
    print("\n" + "=" * 60)
    
    # Create small development set for quick testing
    print("\n2. Creating small development set (100 essays)...")
    dev_path = create_small_development_set(
        input_csv_path=input_csv,
        output_dir=output_directory,
        n_samples=100,
        random_state=42
    )
    
    print("\n" + "=" * 60)
    print("Data splitting completed!")
    print("\nRecommended usage:")
    print(f"- Use '{train_path.name}' for training your grading approach")
    print(f"- Use '{val_path.name}' for final validation/evaluation")
    print(f"- Use '{dev_path.name}' for quick development and testing")
    print("\nNote: The original test.csv appears to be a duplicate of some train data.")
    print("Use validation_split.csv as your true holdout test set.")


if __name__ == "__main__":
    main() 