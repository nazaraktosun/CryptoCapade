# trainers/xgboost_trainer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from datetime import datetime, timedelta

# --- Add parent directory to Python path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End of path modification ---

from utils.data_fetcher import DataFetcher
from utils.featureBuilder import FeatureBuilder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_xgboost_model(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    n_lags: int = 5,
    n_splits: int = 5,
    output_dir: str = "trained_models"
):
    """
    Fetches data via DataFetcher, builds features, trains an XGBoost model,
    evaluates it, and plots results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Fetch OHLCV + Log Returns
    fetcher = DataFetcher()
    data = fetcher.get_crypto_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        compute_log_returns=True,
        n_lags=n_lags
    )
    if data.empty:
        print("No data returned from DataFetcher.")
        return
    print(f"Data head after fetching & cleaning:\n{data.head()}\n")

    # 2) Feature engineering
    fb = FeatureBuilder(data, target_col='Log Returns', n_lags=n_lags)
    X, y = fb.get_features_and_target()
    if X.empty or y.empty:
        print("No data left after feature engineering.")
        return

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # 3) XGBoost with GridSearchCV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
    }
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=mse_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    print("Starting XGBoost grid search...")
    grid.fit(X, y)
    print("GridSearchCV finished.\n")

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV neg. MSE: {grid.best_score_:.6f}\n")

    # 4) Final split evaluation
    train_idx, test_idx = None, None
    for train_idx, test_idx in tscv.split(X):
        pass
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print("Final Test Set Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2 : {r2:.6f}\n")

    # 5) Save model
    model_path = os.path.join(output_dir, f"{symbol}_xgb_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}\n")

    # 6) Diagnostics: Feature importances
    fi = best_model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': fi
    }).sort_values('importance', ascending=False).head(15)
    print("Top 15 feature importances:")
    print(fi_df)

    plt.figure(figsize=(8,6))
    plt.barh(fi_df['feature'], fi_df['importance'])
    plt.gca().invert_yaxis()
    plt.title(f"{symbol} XGBoost Feature Importances")
    plt.tight_layout()
    plt.show()

    # 7) Plot Actual vs Predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--', alpha=0.8)
    plt.title(f"{symbol} XGBoost: Actual vs Predicted")
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    default_symbol     = "BTC"
    default_end_date   = datetime.today()
    default_start_date = default_end_date - timedelta(days=365*2)
    default_n_lags     = 5
    default_n_splits   = 5
    default_output_dir = os.path.join(parent_dir, 'trained_models')

    print(
        f"Running XGBoost trainer for {default_symbol}"
        f" from {default_start_date:%Y-%m-%d} to {default_end_date:%Y-%m-%d}\n"
    )
    train_xgboost_model(
        symbol=default_symbol,
        start_date=default_start_date,
        end_date=default_end_date,
        n_lags=default_n_lags,
        n_splits=default_n_splits,
        output_dir=default_output_dir
    )


if __name__ == "__main__":
    main()
