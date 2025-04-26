# trainers/ARIMA.py
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ARIMATrainer:
    """
    Trainer class for ARIMA modeling on cryptocurrency prices (log returns).

    Exposes a consistent interface:
      - fit(df): fits the model and evaluates on a hold-out
      - predict_historical(df): returns actual vs. predicted on test split
      - forecast_future(df, days): returns future forecasts
      - summary(): returns a brief text summary of fit results
    """

    def __init__(self,
                 p_values=[0, 1, 2],
                 d_values=[0],
                 q_values=[0, 1, 2],
                 test_size=0.2):
        self.p_values = p_values
        self.d_values = d_values
        self.q_values = q_values
        self.test_size = test_size
        self.best_order = None
        self.result = None
        self.train_series = None
        self.test_series = None
        self.metrics_ = {}

    def fit(self, df: pd.DataFrame):
        """
        Fit the ARIMA model on the log-return series, performing grid search
        over (p,d,q) and storing the best result.
        """
        data = df.copy()
        # Calculate log returns
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Log Returns'], inplace=True)

        series = data['Log Returns']
        n = len(series)
        split_idx = int(n * (1 - self.test_size))
        self.train_series = series.iloc[:split_idx]
        self.test_series = series.iloc[split_idx:]

        # Grid search for best (p,d,q)
        best_score = float('inf')
        for p, d, q in itertools.product(self.p_values,
                                         self.d_values,
                                         self.q_values):
            try:
                model = ARIMA_model(self.train_series, order=(p, d, q))
                res = model.fit()
                forecast = res.forecast(steps=len(self.test_series))
                mse = mean_squared_error(self.test_series, forecast)
                if mse < best_score:
                    best_score = mse
                    self.best_order = (p, d, q)
            except Exception:
                continue

        # Refit on train with best_order
        model = ARIMA_model(self.train_series, order=self.best_order)
        self.result = model.fit()

        # Evaluate on test
        forecast = self.result.forecast(steps=len(self.test_series))
        self.metrics_['mse'] = mean_squared_error(self.test_series, forecast)
        self.metrics_['mae'] = mean_absolute_error(self.test_series, forecast)
        self.metrics_['r2']  = r2_score(self.test_series, forecast)
        return self

    def predict_historical(self, df: pd.DataFrame):
        """
        Return the actual vs. predicted log returns on the hold-out test set.
        """
        forecast = self.result.forecast(steps=len(self.test_series))
        actual = self.test_series.values
        predicted = forecast.values
        return actual, predicted

    def forecast_future(self, df: pd.DataFrame, days: int):
        """
        Return forecasted log returns for the next `days` days.
        """
        future = self.result.forecast(steps=days)
        return future.values

    def summary(self) -> str:
        """
        Return a one-line summary of best order and test metrics.
        """
        if self.best_order is None:
            return "ARIMA model has not been fit yet."
        return (
            f"Order={self.best_order}, "
            f"MSE={self.metrics_['mse']:.4f}, "
            f"MAE={self.metrics_['mae']:.4f}, "
            f"R2={self.metrics_['r2']:.4f}"
        )

