# utils/data_fetcher.py
import pandas as pd
from datetime import datetime
import yfinance as yf


class DataFetcher:
    """
    DataFetcher fetches cryptocurrency data using yfinance.
    """

    def __init__(self):
        pass

    def get_crypto_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical crypto data from start_date to end_date using yfinance.

        :param symbol: Cryptocurrency symbol (e.g. BTC, ETH)
        :param start_date: Start date for analysis (datetime)
        :param end_date: End date for analysis (datetime)
        :return: DataFrame with Date and Close price columns
        """
        # Convert symbol to the yfinance ticker format (e.g. BTC -> BTC-USD)
        ticker = f"{symbol}-USD"
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please verify the symbol and try again.")

        # Reset the index to include the Date column and return only Date and Close columns
        data.reset_index(inplace=True)
        return data[['Date', 'Close']]
