import sys
import os

# --- Add parent directory to Python path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End of path modification ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Use DataFetcher to handle data retrieval and preprocessing
from utils.data_fetcher import DataFetcher
from utils.featureBuilder import FeatureBuilder


def train_linear_regression_model(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    n_lags: int = 5
):
    """
    Fetches OHLCV + log returns via DataFetcher, builds lag features, trains a Linear Regression model,
    performs time-series cross-validation, evaluates on final split, and plots results.

    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC').
        start_date: Start date for data fetching.
        end_date: End date for data fetching.
        n_lags: Number of lag features to create.
    """
    fetcher = DataFetcher()
    try:
        # Get data with log returns precomputed
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

    # Feature engineering
    fb = FeatureBuilder(data, target_col='Log Returns', n_lags=n_lags)
    X, y = fb.get_features_and_target()

    if X.empty or y.empty:
        print("Error: No data left after feature engineering. Check your FeatureBuilder logic.")
        return

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features head:\n{X.head()}\n")

    # Linear Regression with TimeSeriesSplit + cross_val_score
    model = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    print("Performing Time Series Cross-Validation...")
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=mse_scorer, n_jobs=-1)
    print("Cross-Validation scores (neg. MSE):", cv_scores)
    print("Mean CV neg. MSE:", np.mean(cv_scores), "\n")

    # Evaluate on final split
    train_idx, test_idx = None, None
    for train_idx, test_idx in tscv.split(X):
        pass  # last split

    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("Linear Regression Test Set Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2 : {r2:.6f}\n")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Log Return', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Log Return (Linear)', linestyle='--', alpha=0.8)
    plt.title(f"{symbol} Linear Regression: Actual vs Predicted Log Returns (Final Split)")
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
        f"Running Linear Regression trainer for {default_symbol}"
        f" from {default_start_date:%Y-%m-%d} to {default_end_date:%Y-%m-%d}\n"
    )
    train_linear_regression_model(
        symbol=default_symbol,
        start_date=default_start_date,
        end_date=default_end_date,
        n_lags=default_n_lags
    )
