# trainers/xgboost_trainer.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
import joblib

from utils.featureBuilder import FeatureBuilder

class XGBoostTrainer:
    """
    Trainer class for XGBoost regression on cryptocurrency log returns using lag features.

    Interface:
      - fit(df): fits model via TimeSeriesSplit and GridSearchCV
      - predict_historical(df): returns actual vs. predicted on test split
      - forecast_future(df, days): iterative future log-return forecasts
      - summary(): returns text summary of best params and test metrics
    """
    def __init__(
        self,
        n_lags: int = 5,
        n_splits: int = 5,
        param_grid: dict = None,
        test_size: float = 0.2
    ):
        self.n_lags = n_lags
        self.n_splits = n_splits
        self.test_size = test_size
        # default hyperparameter grid
        self.param_grid = param_grid or {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        self.best_model = None
        self.best_params_ = {}
        self.metrics_ = {}
        self.train_index = None
        self.test_index = None

    def fit(self, df: pd.DataFrame):
        # compute log returns
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        # build lag features
        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        X, y = fb.get_features_and_target()

        # time-series split and grid search
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=self.param_grid,
            scoring=scorer,
            cv=tscv,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X, y)
        self.best_model = grid.best_estimator_
        self.best_params_ = grid.best_params_

        # final split evaluation
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        self.train_index = train_idx
        self.test_index = test_idx

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # fit final
        self.best_model.fit(X_train, y_train)

        # predict
        y_pred = self.best_model.predict(X_test)
        self.metrics_['mse'] = mean_squared_error(y_test, y_pred)
        self.metrics_['mae'] = mean_absolute_error(y_test, y_pred)
        self.metrics_['r2']  = r2_score(y_test, y_pred)

        return self

    def predict_historical(self, df: pd.DataFrame):
        # return actual vs predicted log returns for test split
        # assume fit() has been called
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        X, y = fb.get_features_and_target()

        actual = y.iloc[self.test_index].values
        predicted = self.best_model.predict(X.iloc[self.test_index])
        return actual, predicted

    def forecast_future(self, df: pd.DataFrame, days: int):
        # naive iterative forecast of future log returns
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        X, y = fb.get_features_and_target()
        # start from last known features
        last_feats = X.iloc[-1].values.reshape(1, -1)
        forecasts = []
        for _ in range(days):
            pred = self.best_model.predict(pd.DataFrame(last_feats, columns=X.columns))[0]
            forecasts.append(pred)
            # roll features: drop oldest lag, append new pred
            last_feats = np.roll(last_feats, -1)
            last_feats[0, -1] = pred
        return np.array(forecasts)

    def summary(self) -> str:
        if self.best_model is None:
            return "XGBoost model not yet fit."
        return (
            f"params={self.best_params_}, "
            f"MSE={self.metrics_['mse']:.4f}, "
            f"MAE={self.metrics_['mae']:.4f}, "
            f"R2={self.metrics_['r2']:.4f}"
        )

# Usage example:
# trainer = XGBoostTrainer()
# trainer.fit(df)
# print(trainer.summary())
