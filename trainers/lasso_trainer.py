import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from utils.featureBuilder import FeatureBuilder

# --------------------------------------
# 1. Load and clean BTC data
# --------------------------------------
data = pd.read_csv("sample_data/BTC-USD_data.csv", skiprows=[1])
data.rename(columns={'Price': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
data.set_index('date', inplace=True)

cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_convert:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    else:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["Log Returns"], inplace=True)

fb = FeatureBuilder(data, target_col= 'Log Returns', n_lags=5)
X,y = fb.get_features_and_target()

##### Lasso with Grid Search
tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_squared_error,greater_is_better=False)

lasso = Lasso(max_iter=10000)
param_grid = {'alpha': [0.001,0.01,0.1,1,10]}
grid_search = GridSearchCV(lasso,param_grid,scoring=scorer, cv = tscv)
grid_search.fit(X,y)

