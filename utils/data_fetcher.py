# utils/data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


class DataFetcher:
    """
    DataFetcher fetches cryptocurrency data using yfinance.
    """

    def __init__(self):
        pass

    def get_crypto_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical crypto data for the past given number of days using yfinance.

        :param symbol: Cryptocurrency symbol (e.g. BTC, ETH)
        :param days: Number of days to fetch data for
        :return: DataFrame with Date and Close price columns
        """
        # Convert symbol to the yfinance ticker format (e.g. BTC -> BTC-USD)
        ticker = f"{symbol}-USD"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Fetch historical data from yfinance
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please verify the symbol and try again.")

        # Reset the index to turn the Date index into a column and return only Date and Close columns
        data.reset_index(inplace=True)
        return data[['Date', 'Close']]
