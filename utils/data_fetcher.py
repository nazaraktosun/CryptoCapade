# utils/data_fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf


class DataFetcher:
    """
    DataFetcher fetches cryptocurrency data using yfinance
    and returns a clean DataFrame ready for feature building.
    """

    def __init__(self):
        pass

    def get_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        compute_log_returns: bool = True,
        n_lags: int = 5
    ) -> pd.DataFrame:
        """
        Fetch historical crypto data from start_date to end_date, 
        coerce types, set the Date index, and optionally add log‐returns.

        :param symbol:      Cryptocurrency symbol (e.g. "BTC", "ETH")
        :param start_date:  Start date for analysis
        :param end_date:    End date for analysis
        :param compute_log_returns: Whether to add a "Log Returns" column
        :param n_lags:      If compute_log_returns, how many lags your FeatureBuilder will need
        :return:            DataFrame, indexed by Date, with numeric Open, High, Low, Close, Volume
                            (and Log Returns if requested)
        """
        # 1) Download full OHLCV
        ticker = f"{symbol}-USD"
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please verify the symbol and try again.")
        
        if isinstance(df.columns, pd.MultiIndex) and "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")

        df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]

        # 3) Ensure all columns are numeric (in case yfinance gives strings with commas)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4) Drop any rows with missing prices or volume
        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

        # 5) Optionally compute log returns (and drop the first n_lags rows)
        if compute_log_returns:
            df["Log Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # if downstream FeatureBuilder uses n_lags, it's cleaner to pre‐drop those
            df.dropna(subset=["Log Returns"], inplace=True)

        # 6) Make sure Date is the index and sorted
        df.index.name = "Date"
        df.sort_index(inplace=True)

        return df
