# trainers/ridge_trainer.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from utils.featureBuilder import FeatureBuilder

class RidgeTrainer:
    """
    Trainer class for Ridge regression on cryptocurrency log returns using lag features.

    Interface:
      - fit(df): fits model and stores results
      - predict_historical(df): returns actual vs. predicted on test split
      - forecast_future(df, days): returns future log-return forecasts
      - summary(): returns text summary of best parameters and metrics
    """
    def __init__(self,
                 n_lags: int = 5,
                 alphas: list[float] = [0.001, 0.01, 0.1, 1, 10, 100],
                 n_splits: int = 5,
                 test_size: float = 0.2):
        self.n_lags = n_lags
        self.alphas = alphas
        self.n_splits = n_splits
        self.test_size = test_size
        self.best_model = None
        self.best_params_ = {}
        self.metrics_ = {}
        self.train_index = None
        self.test_index = None

    def fit(self, df: pd.DataFrame):
        # compute log returns and build lag features
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        fb = FeatureBuilder(data, target_col='Log Returns', n_lags=self.n_lags)
        X, y = fb.get_features_and_target()

        # time-series grid search
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        ridge = Ridge()
        grid = GridSearchCV(ridge, {'alpha': self.alphas}, scoring=scorer,
                            cv=tscv, n_jobs=-1)
        grid.fit(X, y)

        # store best
        self.best_model = grid.best_estimator_
        self.best_params_ = grid.best_params_

        # final split
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        self.train_index = train_idx
        self.test_index = test_idx

        # fit on last split
        x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        self.best_model.fit(x_train, y_train)
        y_pred = self.best_model.predict(x_test)

        # metrics
        self.metrics_['mse'] = mean_squared_error(y_test, y_pred)
        self.metrics_['mae'] = mean_absolute_error(y_test, y_pred)
        self.metrics_['r2']  = r2_score(y_test, y_pred)

        return self

    def predict_historical(self, df: pd.DataFrame):
        # return actual vs predicted log returns for test split
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
        # iterative forecast of future log returns
        data = df.copy()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)
        returns = list(data['Log Returns'].values)

        forecasts = []
        for _ in range(days):
            last = returns[-self.n_lags:]
            X_new = pd.DataFrame([last], columns=[f'lag_{i+1}' for i in range(self.n_lags)])
            pred = self.best_model.predict(X_new)[0]
            forecasts.append(pred)
            returns.append(pred)
        return np.array(forecasts)

    def summary(self) -> str:
        if self.best_model is None:
            return "Ridge model not yet fit."
        return (
            f"alpha={self.best_params_['alpha']}, "
            f"MSE={self.metrics_['mse']:.4f}, "
            f"MAE={self.metrics_['mae']:.4f}, "
            f"R2={self.metrics_['r2']:.4f}"
        )
