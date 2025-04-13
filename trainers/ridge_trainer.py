import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

from utils.featureBuilder import FeatureBuilder

##### Load and clean data

data = pd.read_csv("/Users/nazaraktosun/CryptoCapade/trainers/sample_data/BTC-USD_data.csv",skiprows= [1])
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


#### Feature engineering
fb = FeatureBuilder(data, target_col= 'Log Returns',n_lags=5)
X, y = fb.get_features_and_target()

### Ridge Regression with Grid Search
tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_squared_error,greater_is_better=False)


param_grid = {'alpha': [0.001,0.01,0.1,1,10,100]}
ridge = Ridge()
grid_search = GridSearchCV(ridge,param_grid,scoring=scorer,cv  = tscv)
grid_search.fit(X,y)

# best ridge model

best_ridge = grid_search.best_estimator_
print("Best alpha is: ", grid_search.best_params_["alpha"])

for train_index,test_index in tscv.split(X):
    pass # get last split

X_train , X_test = X.iloc[train_index],X.iloc[test_index]
y_train, y_test = y.iloc[train_index],y.iloc[test_index]

best_ridge.fit(X_train,y_train)
y_pred = best_ridge.predict(X_test)

###### Metrics
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f" MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f" R²: {r2:.6f}")


# --------------------------------------
# 6. Plot Actual vs Predicted
# --------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual Log Return', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Log Return', alpha=0.7)
plt.title("Ridge Regression: Actual vs Predicted Log Returns")
plt.legend()
plt.tight_layout()
plt.show()