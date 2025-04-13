from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv("/Users/nazaraktosun/CryptoCapade/trainers/sample_data/BTC-USD_data.csv", skiprows=[1])
data.rename(columns={'Price': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
data.set_index('Date', inplace=True)
cols_to_convert = ['Open', 'Close', 'High', 'Low', 'Volume']
for col in cols_to_convert:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    else:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
        
        
data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["Log Returns"], inplace=True)
print(data.head())

# plot log returns
plt.figure(figsize=(10,6))
plt.plot(data.index,data["Log Returns"])
plt.title("BTC Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.show()

#We check how stationary data is using Dickey-Fuller test as ARIMA requries a stationary series
adf_result = adfuller(data["Log Returns"])

print(f"ADF Statistics: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("Log return series is stationary")
else:
    print("Log return series are not stationary.")
    
    
#  Identify ARIMA parameters (p and q) with ACF and PACF plots
# ACF gives insghts into the number of moving average(q) terms needed
# PACF determines autoregressive terms (p)
fig,ax = plt.subplots( 2, 1, figsize = (12,8))
plot_acf(data["Log Returns"], ax = ax[0])
plot_pacf(data["Log Returns"],ax= ax[1])
plt.show()

#Based on plots we choose based on ACF/PACF cutoff
train_size = int(len(data)*0.8)
train, test = data.iloc[:train_size],data.iloc[train_size:]

# Define possible parameter ranges
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

best_score = float("inf")
best_order = None

# Grid search over all combinations
for p, d, q in itertools.product(p_values, d_values, q_values):
    try:
        model = ARIMA(train["Log Returns"], order=(p,d,q))
        result = model.fit()
        forecast = result.forecast(steps=len(test))
        mse = mean_squared_error(test["Log Returns"], forecast)
        if mse < best_score:
            best_score = mse
            best_order = (p,d,q)
    except:
        continue

# Report best order
print(f"Best ARIMA order: {best_order} with MSE: {best_score:.5f}")
# Refit the model using the best found (p, d, q)
p, d, q = best_order
model = ARIMA(train["Log Returns"], order=(p,d,q))
result = model.fit()
forecast = result.forecast(steps=len(test))

# Plot the actual log returns and forecasted log returns
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Log Returns'], label='Training Data')
plt.plot(test.index, test['Log Returns'], label='Test Data')
plt.plot(test.index, forecast, label='Forecasted Log Returns', linestyle='--')
plt.legend()
plt.title("ARIMA Forecast on Test Data (Log Returns)")
plt.show()


mae = mean_absolute_error(test["Log Returns"],forecast)
mse = mean_squared_error(test["Log Returns"],forecast)
r2 = r2_score(test["Log Returns"],forecast)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")