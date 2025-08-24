#!/usr/bin/env python3
"""
Train RandomForest regressors with varying small training sizes (10,15,...,50)
and evaluate on a fixed 100-sample test set from the same dataset.
Push each run's predictions to Google Sheets with worksheet names:
  nahuel-tree-Nx100
where N is the number of training samples.

Default features source: exports/features_train_80_full.csv
Requires env:
  - GOOGLE_SHEETS_ID
  - SHEETS_CREDENTIALS_BASE64 (preferred) or GOOGLE_SHEETS_CREDENTIALS_PATH
"""

import os
import sys
import json
import base64
import time
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
    parser = argparse.ArgumentParser(description="Run tree regressors with varying train sizes and push to Sheets")
    parser.add_argument("--features", default="exports/features_train_80_full.csv", help="Features CSV (must include essay_id, score)")
    parser.add_argument("--train-sizes", nargs="*", type=int, default=[10,15,20,25,30,35,40,45,50], help="Training sizes to evaluate")
    parser.add_argument("--test-size", type=int, default=100, help="Fixed test set size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--sheet-prefix", default="nahuel-tree", help="Worksheet name prefix")
    parser.add_argument("--grid-small", action="store_true", help="Use smaller hyperparameter grid for speed")
    args = parser.parse_args()

    # Load features
    df = pd.read_csv(args.features)
    if "essay_id" not in df.columns or "score" not in df.columns:
        raise ValueError("Features CSV must have essay_id and score columns")

    feature_cols = [c for c in df.columns if c not in ("essay_id", "score")]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Sample fixed test set
    rng = np.random.RandomState(args.random_state)
    if len(df) < args.test_size + max(args.train_sizes):
        # Not strictly required to be disjoint, but we prefer it if possible
        pass

    all_indices = np.arange(len(df))
    test_indices = rng.choice(all_indices, size=min(args.test_size, len(df)), replace=False)
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_indices] = True

    test_df = df.iloc[test_indices].copy()
    X_test = test_df[feature_cols].values
    y_test = test_df["score"].values
    test_ids = test_df["essay_id"].astype(str).tolist()

    # Sheets client
    sheets_id = os.getenv("GOOGLE_SHEETS_ID")
    if not sheets_id:
        raise RuntimeError("GOOGLE_SHEETS_ID is not set in environment")

    credentials_dict = None
    if os.getenv("SHEETS_CREDENTIALS_BASE64"):
        try:
            credentials_dict = json.loads(base64.b64decode(os.getenv("SHEETS_CREDENTIALS_BASE64")))
        except Exception:
            credentials_dict = None

    sheets_client = SheetsClient(credentials_dict=credentials_dict) if credentials_dict else SheetsClient()

    # Evaluate for each train size
    for n in args.train_sizes:
        # Choose train indices disjoint from test if possible
        candidate_indices = np.where(~test_mask)[0]
        if len(candidate_indices) < n:
            # Fall back to sampling from all
            candidate_indices = all_indices
        train_indices = rng.choice(candidate_indices, size=min(n, len(candidate_indices)), replace=False)
        train_df = df.iloc[train_indices].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df["score"].values

        # Model & grid
        rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
        if args.grid_small:
            param_grid = {
                "n_estimators": [200],
                "max_depth": [12, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1],
                "max_features": ["sqrt", 0.5],
            }
        else:
            param_grid = {
                "n_estimators": [300, 600],
                "max_depth": [12, 18, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.5],
            }

        search = GridSearchCV(
            rf, param_grid=param_grid, scoring="neg_mean_squared_error", cv=3, n_jobs=-1, verbose=0
        )

        start_time = time.time()
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fit_seconds = time.time() - start_time

        y_pred = model.predict(X_test)
        qwk = quadratic_weighted_kappa(y_test, y_pred)

        # Prepare rows for Sheets
        scores = []
        for i in range(len(test_ids)):
            scores.append({
                "essay_id": test_ids[i],
                "total_score": float(y_pred[i]),
                "reasoning": f"RF n={n}, grid_small={args.grid_small}, fit_s={fit_seconds:.2f}",
                "model_used": "RF-Features",
            })

        worksheet_name = f"{args.sheet_prefix}-{n}x{args.test_size}"
        run_id = datetime.now().strftime("%H:%M:%S")

        ok = sheets_client.write_scores_to_sheet(
            scores=scores,
            spreadsheet_id=sheets_id,
            worksheet_name=worksheet_name,
            run_id=run_id,
            actual_scores=list(y_test.astype(float)),
            component_categories=None,
        )

        print(f"{worksheet_name}: QWK={qwk:.4f}, fit_s={fit_seconds:.2f}, pushed={ok}")


if __name__ == "__main__":
    main()

