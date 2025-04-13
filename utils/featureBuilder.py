import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self, df, target_col='Log Returns', n_lags=5):
        self.df = df.copy()
        self.target_col = target_col
        self.n_lags = n_lags

    def add_lag_features(self):
        for lag in range(1, self.n_lags + 1):
            self.df[f'lag_{lag}'] = self.df[self.target_col].shift(lag)
        return self

    def add_rolling_features(self, window=5):
        self.df[f'ma_{window}'] = self.df[self.target_col].rolling(window).mean()
        self.df[f'std_{window}'] = self.df[self.target_col].rolling(window).std()
        self.df[f'z_score_{window}'] = (
            (self.df[self.target_col] - self.df[f'ma_{window}']) / self.df[f'std_{window}']
        )
        return self

    def add_technical_indicators(self):
        if 'Close' in self.df.columns:
            # EMA
            self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
            self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()

            # MACD and Signal Line
            self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
            self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

            # RSI (14-period)
            delta = self.df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.df['RSI_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands (20-period, 2 std)
            ma = self.df['Close'].rolling(window=20).mean()
            std = self.df['Close'].rolling(window=20).std()
            self.df['BB_upper'] = ma + (2 * std)
            self.df['BB_lower'] = ma - (2 * std)

        if 'Volume' in self.df.columns:
            # OBV (On-Balance Volume)
            obv = [0]  # first day starts at 0
            for i in range(1, len(self.df)):
                if self.df['Close'].iloc[i] > self.df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] + self.df['Volume'].iloc[i])
                elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] - self.df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            self.df['OBV'] = obv


    def clean(self):
        self.df.dropna(inplace=True)
        return self

    def get_features_and_target(self):
        self.add_lag_features()
        self.add_rolling_features()
        self.add_technical_indicators()
        self.clean()

        feature_cols = [col for col in self.df.columns if col.startswith('lag_') or
                        col.startswith('ma_') or col.startswith('std_') or
                        col.startswith('z_score') or
                        col in ['RSI_14', 'MACD', 'MACD_signal', 'EMA_12', 'EMA_26',
                                'BB_upper', 'BB_lower', 'OBV']]

        X = self.df[feature_cols]
        y = self.df[self.target_col]
        return X, y
