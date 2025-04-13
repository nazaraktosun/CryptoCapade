import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression

from utils.featureBuilder import FeatureBuilder  

data = pd.read_csv("/Users/nazaraktosun/CryptoCapade/trainers/sample_data/BTC-USD_data.csv", skiprows=[1])
data.rename(columns={'Price': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce') 
data.set_index('date', inplace=True)

# Convert columns with commas
cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']    
for col in cols_to_convert:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    else:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Compute Log Returns
data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["Log Returns"], inplace=True)


fb = FeatureBuilder(data, target_col='Log Returns', n_lags=5)
X, y = fb.get_features_and_target()


model = LinearRegression()
tscv = TimeSeriesSplit(n_splits=5)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

mse_list = []
mae_list = []
r2_list = []

cross_val_scores = cross_val_score(model, X, y, cv=tscv, scoring=mse_scorer)

for train_index, test_index in tscv.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mse_list.append(mean_squared_error(y_test, y_pred))
    mae_list.append(mean_absolute_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))


print(f"Avg MSE: {np.mean(mse_list):.6f}")
print(f"Avg MAE: {np.mean(mae_list):.6f}")
print(f"Avg R²: {np.mean(r2_list):.6f}")
print("Cross-Validation MSEs (negative):", cross_val_scores)
print("Mean Cross-Validation MSE (negative):", -np.mean(cross_val_scores))

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual Log Return', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Log Return', alpha=0.7)
plt.legend()
plt.title("Linear Regression: Actual vs Predicted Log Returns")
plt.tight_layout()
plt.show()
