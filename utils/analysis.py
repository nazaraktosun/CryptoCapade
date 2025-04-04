# utils/analysis.py
import pandas as pd

class CryptoAnalysis:
    """
    CryptoAnalysis performs statistical analyses on crypto data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with crypto data.
        :param data: DataFrame with Date and Close columns.
        """
        self.data = data

    def calculate_moving_average(self, window: int = 7) -> pd.Series:
        """
        Calculate a simple moving average over the given window.
        :param window: Number of periods to average.
        :return: Pandas Series with moving average.
        """
        return self.data["Close"].rolling(window=window).mean()

    def calculate_daily_returns(self) -> pd.Series:
        """
        Calculate daily percentage returns.
        :return: Pandas Series with daily returns.
        """
        return self.data["Close"].pct_change() * 100
