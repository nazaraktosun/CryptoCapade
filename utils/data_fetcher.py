# utils/data_fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataFetcher:
    """
    DataFetcher fetches or simulates cryptocurrency data.
    """

    def __init__(self):
        pass

    def get_crypto_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Simulate fetching historical crypto data for the past given number of days.
        In a production version, replace this simulation with an API call.

        :param symbol: Cryptocurrency symbol (e.g. BTC, ETH)
        :param days: Number of days to simulate
        :return: DataFrame with date and closing price
        """
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()
        # Simulate closing prices with random walk
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(days)) + 100  # Starting around 100
        df = pd.DataFrame({"Date": dates, "Close": prices})
        return df
