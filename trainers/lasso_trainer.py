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
data = pd.read_csv("/Users/nazaraktosun/CryptoCapade/trainers/sample_data/BTC-USD_data.csv", skiprows=[1])
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

best_model = grid_search.best_estimator_
print('Best alpha', grid_search.best_params_['alpha'])


#Evaluate
for train_index , test_index in tscv.split(X):
    pass # last fold

x_train, x_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

best_model.fit(x_train,y_train)
y_pred = best_model.predict(x_test)

#####Metrics#####
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f'MSE : {mse:.6f}')
print(f'MAE : {mae:.6f}')
print(f'R2 : {r2:.6f}')

#Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual Log Return', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Log Return', alpha=0.7)
plt.title("Lasso Regression: Actual vs Predicted Log Returns")
plt.legend()
plt.tight_layout()
plt.show()

#Selected features
coefs = pd.Series(best_model.coef_ , index =X.columns)

selected = coefs[coefs != 0].sort_values(key = lambda x:abs(x), ascending=False)
print("Selected Features: ")
print(selected)

