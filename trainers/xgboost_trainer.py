import numpy as np
import pandas as pd
from utils.featureBuilder import FeatureBuilder
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor


# --- Load and clean BTC data ---
data = pd.read_csv("sample_data/BTC-USD_data.csv", skiprows=[1])
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


fb = FeatureBuilder()
X , y = fb.get_features_and_target()

#Time series split
tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_squared_error,greater_is_better=False)
