#!/usr/bin/env python3
"""
Train a tree-based regressor on extracted features and evaluate with QWK.

Inputs:
  - --train-features: CSV produced from training split (includes score)
  - --val-features:   CSV produced from validation split (includes score)

Outputs:
  - exports/model_rf.joblib
  - exports/feature_importance_rf.csv
  - metrics printed to stdout (QWK, RMSE, MAE, R2)
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from joblib import dump


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating=None, max_rating=None) -> float:
    """Compute Quadratic Weighted Kappa (QWK) for integer ratings.
    We round predictions to nearest integer within observed [min,max] range.
    """
    y_true = np.asarray(y_true).astype(int)
    if min_rating is None:
        min_rating = int(np.min(y_true))
    if max_rating is None:
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
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    O = conf_mat / np.sum(conf_mat)
    E = E / np.sum(E)

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'essay_id' in df.columns:
        df = df.drop(columns=['essay_id'])
    if 'score' not in df.columns:
        raise ValueError(f"score column missing in {path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train tree regressor with QWK evaluation")
    parser.add_argument('--train-features', required=True, help='CSV from training split')
    parser.add_argument('--val-features', required=True, help='CSV from validation split')
    parser.add_argument('--outdir', default='exports', help='Output directory')
    parser.add_argument('--train-sample-size', type=int, default=None, help='Optional: sample N rows for training')
    parser.add_argument('--val-sample-size', type=int, default=None, help='Optional: sample N rows for validation')
    parser.add_argument('--drop-incomplete', action='store_true', help='Drop rows with any missing feature values (full vectors only)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for sampling and model')
    args = parser.parse_args()

    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    # Determine if we are sampling from a single file (when paths are equal)
    use_single_source = os.path.abspath(args.train_features) == os.path.abspath(args.val_features)

    if use_single_source and (args.train_sample_size or args.val_sample_size):
        base_df = load_features(args.train_features)
        feature_cols = [c for c in base_df.columns if c != 'score']
        if args.drop_incomplete:
            before = len(base_df)
            base_df = base_df.dropna(subset=feature_cols + ['score'])
            after = len(base_df)
            print(f"Dropped {before - after} rows with incomplete feature vectors (single source)")
        else:
            base_df[feature_cols] = base_df[feature_cols].fillna(0.0)

        n_train = args.train_sample_size or 30
        n_val = args.val_sample_size or 100
        total_needed = n_train + n_val
        if len(base_df) < total_needed:
            raise ValueError(f"Not enough rows in {args.train_features} to sample {total_needed} disjoint rows (have {len(base_df)})")

        sampled = base_df.sample(n=total_needed, random_state=args.random_state)
        train_df = sampled.iloc[:n_train].copy()
        val_df = sampled.iloc[n_train:].copy()
    else:
        train_df = load_features(args.train_features)
        val_df = load_features(args.val_features)
        feature_cols = [c for c in train_df.columns if c != 'score']
        # Align validation columns to train feature set
        val_df = val_df.reindex(columns=feature_cols + ['score'])
        if args.drop_incomplete:
            before_t = len(train_df)
            train_df = train_df.dropna(subset=feature_cols + ['score'])
            after_t = len(train_df)
            before_v = len(val_df)
            val_df = val_df.dropna(subset=feature_cols + ['score'])
            after_v = len(val_df)
            print(f"Dropped {before_t - after_t} train and {before_v - after_v} val rows with incomplete feature vectors")
        else:
            train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
            val_df[feature_cols] = val_df[feature_cols].fillna(0.0)

        if args.train_sample_size:
            if len(train_df) < args.train_sample_size:
                raise ValueError(f"Requested train-sample-size {args.train_sample_size} > available {len(train_df)}")
            train_df = train_df.sample(n=args.train_sample_size, random_state=args.random_state)
        if args.val_sample_size:
            if len(val_df) < args.val_sample_size:
                raise ValueError(f"Requested val-sample-size {args.val_sample_size} > available {len(val_df)}")
            val_df = val_df.sample(n=args.val_sample_size, random_state=args.random_state)

    # Features and targets
    feature_cols = [c for c in train_df.columns if c != 'score']
    X_train = train_df[feature_cols].values
    y_train = train_df['score'].values
    X_val = val_df[feature_cols].reindex(columns=feature_cols, fill_value=0.0).values
    y_val = val_df['score'].values

    rf = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    param_grid = {
        'n_estimators': [300, 600],
        'max_depth': [12, 18, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 0.5]
    }
    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    min_rating = int(min(np.min(y_train), np.min(y_val)))
    max_rating = int(max(np.max(y_train), np.max(y_val)))
    qwk_val = quadratic_weighted_kappa(y_val, y_pred_val, min_rating=min_rating, max_rating=max_rating)

    dump(model, os.path.join(args.outdir, 'model_rf.joblib'))
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}) \
        .sort_values('importance', ascending=False)
    importance.to_csv(os.path.join(args.outdir, 'feature_importance_rf.csv'), index=False)

    print("\nRESULTS")
    print("======")
    print(f"Best Params: {search.best_params_}")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Val RMSE:   {rmse_val:.4f}")
    print(f"Val MAE:    {mae_val:.4f}")
    print(f"Val R2:     {r2_val:.4f}")
    print(f"Val QWK:    {qwk_val:.4f}")

    if qwk_val >= 0.7:
        print("\nPASS: QWK >= 0.7")
    else:
        print("\nWARN: QWK < 0.7. Consider enabling model-based features or tuning grid further.")


if __name__ == '__main__':
    main()

