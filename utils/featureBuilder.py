import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self, data, target_col='Log Returns', n_lags=5):
        """
        Initialize the FeatureBuilder with the dataset and parameters.

        Args:
            data (pd.DataFrame): The time series data, must have a datetime index
                                 and 'Close' column. 'Volume' is optional but needed for some indicators.
            target_col (str): The column name for the target variable (often 'Log Returns').
            n_lags (int): Number of lag features to create for the target column.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        self.df = data.copy()
        self.target_col = target_col
        self.n_lags = n_lags

        # Ensure target column exists or calculate it if it's 'Log Returns'
        if self.target_col == 'Log Returns' and 'Log Returns' not in self.df.columns:
            self.df['Log Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential inf values

    def add_lag_features(self):
        """Create lag features for the target column."""
        if self.target_col not in self.df.columns:
             raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        for lag in range(1, self.n_lags + 1):
            self.df[f'{self.target_col}_lag_{lag}'] = self.df[self.target_col].shift(lag)

    def add_technical_indicators(self, volatility_window=20):
        """
        Adds common technical indicators to the DataFrame.

        Args:
            volatility_window (int): The rolling window size for volatility calculation.
        """
        # Ensure Log Returns exist for volatility calculation
        if 'Log Returns' not in self.df.columns:
             self.df['Log Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
             self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # --- Volatility ---
        # Calculate rolling standard deviation of log returns
        self.df[f'Volatility_{volatility_window}'] = self.df['Log Returns'].rolling(window=volatility_window).std() * np.sqrt(volatility_window) # Optional: Annualize/Scale

        # --- Other Indicators ---
        if 'Close' in self.df.columns:
            # SMA (Simple Moving Average)
            self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
            self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()

            # EMA (Exponential Moving Average)
            self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
            self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()

            # MACD (Moving Average Convergence Divergence)
            self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
            self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

            # RSI (14-period)
            delta = self.df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # Avoid division by zero for RSI
            rs = gain / loss
            rs[loss == 0] = np.inf # Handle cases where loss is zero
            self.df['RSI_14'] = 100 - (100 / (1 + rs))
            self.df['RSI_14'].fillna(100, inplace=True) # Fill initial NaNs or cases where loss was 0

            # Bollinger Bands (20-period, 2 std)
            ma = self.df['Close'].rolling(window=20).mean()
            std = self.df['Close'].rolling(window=20).std()
            self.df['BB_upper'] = ma + (2 * std)
            self.df['BB_lower'] = ma - (2 * std)

        if 'Volume' in self.df.columns and 'Close' in self.df.columns:
            # OBV (On-Balance Volume) - Vectorized approach for efficiency
            signed_volume = self.df['Volume'] * np.sign(self.df['Close'].diff()).fillna(0)
            self.df['OBV'] = signed_volume.cumsum()


    def clean(self):
        """Removes rows with NaN values introduced by feature calculations."""
        self.df.dropna(inplace=True)

    def get_features_and_target(self, add_lags=True, add_indicators=True, volatility_window=20):
        """
        Generates features and target variable.

        Args:
            add_lags (bool): Whether to add lag features.
            add_indicators (bool): Whether to add technical indicators.
            volatility_window (int): Window for volatility calculation if add_indicators is True.

        Returns:
            pd.DataFrame: Features (X)
            pd.Series: Target variable (y)
        """
        if add_lags:
            self.add_lag_features()
        if add_indicators:
            self.add_technical_indicators(volatility_window=volatility_window)

        self.clean() # Remove NaNs after all features are added

        if self.target_col not in self.df.columns:
             raise ValueError(f"Target column '{self.target_col}' is not available after feature generation and cleaning.")

        # Define features (all columns except the target)
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Ensure X does not contain the target column if it was also a base feature (e.g., 'Close' if target is 'Close')
        if self.target_col in X.columns:
             X = X.drop(columns=[self.target_col])


        return X, y
