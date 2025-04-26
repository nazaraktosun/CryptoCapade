# trainers/ridge_trainer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import sys
import os
from datetime import datetime, timedelta

# --- Add parent directory to Python path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End of path modification ---

from utils.data_fetcher import DataFetcher
from utils.featureBuilder import FeatureBuilder


def train_ridge_model(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    n_lags: int = 5
):
    """
    Fetches data via DataFetcher, builds features, trains a Ridge model,
    evaluates it, and plots results.
    """
    # 1) Fetch OHLCV + Log Returns in one go
    fetcher = DataFetcher()
    try:
        data = fetcher.get_crypto_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            compute_log_returns=True,
            n_lags=n_lags
        )
    except ValueError as e:
        print(f"Failed to fetch data: {e}")
        return

    if data.empty:
        print("No data returned from DataFetcher.")
        return

    print(f"Data head after fetching & cleaning:\n{data.head()}\n")

    # 2) Feature engineering
    fb = FeatureBuilder(data, target_col='Log Returns', n_lags=n_lags)
    X, y = fb.get_features_and_target()

    if X.empty or y.empty:
        print("Error: No data left after feature engineering. Check your FeatureBuilder logic.")
        return

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape:   {y.shape}")
    print(f"Features head:\n{X.head()}\n")

    # 3) Ridge with TimeSeriesSplit + GridSearchCV
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    ridge = Ridge()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    print("Starting Ridge parameter grid search...")
    grid = GridSearchCV(ridge, param_grid, scoring=mse_scorer, cv=tscv, n_jobs=-1)
    grid.fit(X, y)
    print("GridSearchCV finished.\n")

    best_model = grid.best_estimator_
    print(f"Best alpha: {grid.best_params_['alpha']}")
    print(f"Best CV score (neg. MSE): {grid.best_score_:.6f}\n")

    # 4) Evaluate on final split
    train_idx, test_idx = None, None
    for train_idx, test_idx in tscv.split(X):
        pass  # after loop, these are the last split

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("Ridge Test Set Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2 : {r2:.6f}\n")

    # 5) Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test,  label='Actual Log Return', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Log Return (Ridge)',
             linestyle='--', alpha=0.8)
    plt.title(f"{symbol} Ridge Regression: Actual vs Predicted Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    default_symbol     = "BTC"
    default_end_date   = datetime.today()
    default_start_date = default_end_date - timedelta(days=365*2)
    default_n_lags     = 5

    print(
        f"Running Ridge trainer for {default_symbol}"
        f" from {default_start_date:%Y-%m-%d}"
        f" to {default_end_date:%Y-%m-%d}\n"
    )
    train_ridge_model(
        symbol=default_symbol,
        start_date=default_start_date,
        end_date=default_end_date,
        n_lags=default_n_lags
    )
