# trainers/ARIMA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import sys
import os
from datetime import datetime, timedelta

# --- Add parent directory to Python path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End of path modification ---

# Import DataFetcher from utils
try:
    from utils.data_fetcher import DataFetcher
except ModuleNotFoundError:
    print(f"Error: Could not import DataFetcher. Current sys.path: {sys.path}")
    print(f"Attempted to add parent directory: {parent_dir}")
    sys.exit(1) # Exit if import fails


def train_arima_model(symbol: str, start_date: datetime, end_date: datetime):
    """
    Fetches data, trains an ARIMA model on log returns, evaluates it, and plots results.

    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC').
        start_date: Start date for data fetching.
        end_date: End date for data fetching.
    """
    fetcher = DataFetcher()
    try:
        # Fetch data using DataFetcher
        data = fetcher.get_crypto_data(symbol=symbol, start_date=start_date, end_date=end_date)
    except ValueError as e:
        print(f"Failed to fetch data: {e}")
        return

    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    

    # Compute Log Returns
    data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=["Log Returns"], inplace=True)

    if data.empty:
        print("Not enough data after computing log returns and dropping NaNs.")
        return

    print(f"Data head after preprocessing:\n{data.head()}")

    # plot log returns
    plt.figure(figsize=(10,6))
    plt.plot(data.index,data["Log Returns"])
    plt.title(f"{symbol} Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.show()

    # We check how stationary data is using Dickey-Fuller test as ARIMA requries a stationary series
    adf_result = adfuller(data["Log Returns"])

    print(f"\nADF Statistics: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("Log return series is stationary")
    else:
        print("Log return series are not stationary. ARIMA might not be the best fit or differencing is needed.")


    # Identify ARIMA parameters (p and q) with ACF and PACF plots
    # ACF gives insghts into the number of moving average(q) terms needed
    # PACF determines autoregressive terms (p)
    fig,ax = plt.subplots( 2, 1, figsize = (12,8))
    plot_acf(data["Log Returns"], ax = ax[0])
    plot_pacf(data["Log Returns"],ax= ax[1])
    plt.show()

    train_size = int(len(data)*0.8)
    train, test = data["Log Returns"].iloc[:train_size],data["Log Returns"].iloc[train_size:]

    # Define possible parameter ranges for grid search
    # d=0 is used because log returns are usually stationary (checked by ADF test)
    p_values = [0, 1, 2]
    d_values = [0] # Assuming log returns are stationary, d=0
    q_values = [0, 1, 2]

    best_score = float("inf")
    best_order = None

    print("\nStarting ARIMA parameter grid search...")
    # Grid search over all combinations
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p,d,q))
            result = model.fit()
            forecast = result.forecast(steps=len(test))
            mse = mean_squared_error(test, forecast)
            if mse < best_score:
                best_score = mse
                best_order = (p,d,q)
            print(f" ARIMA({p},{d},{q}) MSE: {mse:.5f}")
        except Exception as e:
            # print(f"Could not fit ARIMA({p},{d},{q}): {e}") # Uncomment for detailed errors
            continue

    if best_order is None:
        print("Could not find a valid ARIMA order.")
        return

    # Report best order
    print(f"\nBest ARIMA order: {best_order} with MSE: {best_score:.5f}")

    # Refit the model using the best found (p, d, q) on the full training set
    p, d, q = best_order
    model = ARIMA(train, order=(p,d,q))
    result = model.fit()
    forecast = result.forecast(steps=len(test))

    # Plot the actual log returns and forecasted log returns
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, forecast, label='Forecasted Log Returns', linestyle='--')
    plt.legend()
    plt.title(f"{symbol} ARIMA Forecast on Test Data (Log Returns)")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.show()

    # Evaluate metrics on the test set
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    r2 = r2_score(test, forecast)
    print(f"\nARIMA Test Set Metrics:")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    print(f" R2: {r2:.4f}")




if __name__ == "__main__":
    
    default_symbol = "BTC"
    default_end_date = datetime.today()
    default_start_date = default_end_date - timedelta(days=365*2) # Fetch 2 years of data

    print(f"Running ARIMA trainer for {default_symbol} from {default_start_date.strftime('%Y-%m-%d')} to {default_end_date.strftime('%Y-%m-%d')}")
    train_arima_model(symbol=default_symbol, start_date=default_start_date, end_date=default_end_date)
