#!/usr/bin/env python3
"""
Sample 30 training rows and 100 validation rows from a features CSV,
train a RandomForestRegressor, compute QWK on the 100, and write the
per-essay predictions plus an aggregated row to Google Sheets using the
existing Sheets integration format.

Requires env vars:
- GOOGLE_SHEETS_ID
- SHEETS_CREDENTIALS_BASE64 (preferred) or GOOGLE_SHEETS_CREDENTIALS_PATH
"""

import os
import sys
import json
import base64
import argparse
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


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def main():
    parser = argparse.ArgumentParser(description="Push 30/100 RF experiment to Google Sheets")
    parser.add_argument("--input-features", default="exports/features_train_80_full.csv", help="Features CSV with essay_id and score")
    parser.add_argument("--train-size", type=int, default=30, help="Training sample size")
    parser.add_argument("--val-size", type=int, default=100, help="Validation sample size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--drop-incomplete", action="store_true", help="Drop rows with any missing feature values")
    parser.add_argument("--worksheet-name", default=None, help="Worksheet name (default: ml-30x100-<timestamp>)")
    parser.add_argument("--out-preds", default="exports/experiment_30_100_predictions.csv", help="Where to save predictions CSV")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_features)
    if "essay_id" not in df.columns or "score" not in df.columns:
        raise ValueError("Input features must include 'essay_id' and 'score' columns")

    feature_cols = [c for c in df.columns if c not in ("essay_id", "score")]
    if args.drop_incomplete:
        before = len(df)
        df = df.dropna(subset=feature_cols + ["score"]).copy()
        print(f"Dropped {before - len(df)} rows with incomplete feature vectors")
    else:
        df[feature_cols] = df[feature_cols].fillna(0.0)

    total_needed = args.train_size + args.val_size
    if len(df) < total_needed:
        raise ValueError(f"Need {total_needed} rows but only have {len(df)} in {args.input_features}")

    sampled = df.sample(n=total_needed, random_state=args.random_state).reset_index(drop=True)
    train_df = sampled.iloc[:args.train_size].copy()
    val_df = sampled.iloc[args.train_size:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["score"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score"].values
    essay_ids_val = val_df["essay_id"].astype(str).tolist()

    # Train RF with a compact grid
    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    param_grid = {
        "n_estimators": [300],
        "max_depth": [12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1],
        "max_features": ["sqrt", 0.5],
    }
    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred_val = model.predict(X_val)
    qwk = quadratic_weighted_kappa(y_val, y_pred_val)
    print(f"Val QWK (30 train, 100 val): {qwk:.4f}")

    # Save predictions CSV
    out_path = Path(args.out_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "essay_id": essay_ids_val,
        "pred": y_pred_val,
        "actual": y_val,
    }).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    # Prepare scores payload for Sheets
    scores = []
    for i in range(len(essay_ids_val)):
        scores.append({
            "essay_id": essay_ids_val[i],
            "total_score": float(y_pred_val[i]),
            "reasoning": "RandomForest on extracted features (30-train/100-test)",
            "model_used": "RF-Features",
            # Optional fields not used for this experiment:
            # "category_scores": {},
            # "metadata": {"experiment": "30_train_100_val"}
        })

    # Initialize Sheets client
    sheets_id = os.getenv("GOOGLE_SHEETS_ID")
    if not sheets_id:
        raise RuntimeError("GOOGLE_SHEETS_ID is not set in environment")

    credentials_dict = None
    if os.getenv("SHEETS_CREDENTIALS_BASE64"):
        try:
            credentials_dict = json.loads(base64.b64decode(os.getenv("SHEETS_CREDENTIALS_BASE64")))
        except Exception:
            credentials_dict = None

    if credentials_dict:
        sheets_client = SheetsClient(credentials_dict=credentials_dict)
    else:
        # Fallback to credentials file path via config setting
        sheets_client = SheetsClient()

    # Worksheet name
    worksheet_name = args.worksheet_name
    if not worksheet_name:
        worksheet_name = f"ml-30x100-{datetime.now().strftime('%H%M%S')}"

    # Run ID
    run_id = datetime.now().strftime("%H:%M:%S")

    # Write to Google Sheets using existing format
    ok = sheets_client.write_scores_to_sheet(
        scores=scores,
        spreadsheet_id=sheets_id,
        worksheet_name=worksheet_name,
        run_id=run_id,
        actual_scores=list(y_val.astype(float)),
        component_categories=None,
    )

    if ok:
        print(f"Pushed {len(scores)} rows to Google Sheets worksheet '{worksheet_name}' with QWK={qwk:.4f}")
    else:
        print("Failed to write to Google Sheets", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

